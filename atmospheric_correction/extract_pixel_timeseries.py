import os, glob, re
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
CSV_uncorrected_FOLDER = os.path.join(DATA_DIR, "decoded_images")
CSV_corrected_FOLDER   = os.path.join(DATA_DIR, "corrected")
EMISSIVITY_PATH        = os.path.join(DATA_DIR, "corrected", "daily_emissivity.csv")
METEO_PATH             = os.path.join(DATA_DIR, "CSVs", "murtel_met_qc.csv")
OUT_CSV                = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")

# Validation inputs
RADIO_RIDGE_PATH  = os.path.join(DATA_DIR, "validation_data", "radiometer_ridge.csv")
RADIO_FURROW_PATH = os.path.join(DATA_DIR, "validation_data", "radiometer_furrow.csv")

BH1_PATH = os.path.join(DATA_DIR, "validation_data", "BH_1.csv")
BH2_PATH = os.path.join(DATA_DIR, "validation_data", "BH_2.csv")
BH3_PATH = os.path.join(DATA_DIR, "validation_data", "BH_3.csv")

GST_PATH = os.path.join(DATA_DIR, "validation_data", "GST.csv")

START_DATE = "2021-01-01"
END_DATE   = "2023-08-31"

RADIUS = 2
W, H = 336, 252
SIGMA = 5.670374419e-8

MATCH_TOL_MIN = 60
RADIO_COL = "Corrected_Target_Ambient_LW_Lab"

# Measurement points (pixel x, y)
POINTS = {
    "Meteo_Station": (132, 94),
    "GST_1": (284, 162),
    "GST_2": (192, 154),
    "GST_3": (169, 102),
    "GST_4": (136, 91),
    "GST_5": (101, 77),
    "BH_1":  (132, 94),
    "BH_2":  (123, 88),
    "BH_3":  (96, 75),
    "radio_furrow": (268, 140),
    "radio_ridge": (279, 138),
}

TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

def name_to_datetime(fname: str) -> datetime | None:
    m = TS_RE.search(fname)
    if not m:
        return None
    yy, MM, DD, hh, mm = map(int, m.groups())
    return datetime(2000 + yy, MM, DD, hh, mm)

def extract_ts_key(fname: str) -> str | None:
    m = TS_RE.search(fname)
    if not m:
        return None
    yy, MM, DD, hh, mm = m.groups()
    return f"m{yy}{MM}{DD}{hh}{mm}"


# Validation helpers
def read_csv_flexible_any(path: str) -> pd.DataFrame:
    na = ["NA", "NaN", "nan", "", "null", "None"]
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False, na_values=na, keep_default_na=True)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False, na_values=na, keep_default_na=True)
    except ValueError:
        return pd.read_csv(path, sep=None, engine="python", na_values=na, keep_default_na=True)

def find_time_col(df: pd.DataFrame) -> str:
    candidates = ["Combined_Date_Time", "Time", "time", "TimeStamp", "timestamp", "Timestamp", "DateTime", "datetime", "Date", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc:
            return c
    return df.columns[0]

def load_validation_df(path: str) -> tuple[pd.DataFrame, str]:
    df = read_csv_flexible_any(path)
    tcol = find_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    return df, tcol

def nearest_row_to_tir(df: pd.DataFrame, time_col: str, target_time: datetime, tol_minutes: int) -> pd.Series | None:
    if df is None or df.empty or pd.isna(target_time):
        return None

    tol = pd.Timedelta(minutes=tol_minutes)
    t = pd.Timestamp(target_time)

    # sorted in load_validation_df — binary search is valid here
    pos = df[time_col].searchsorted(t)
    candidates = []
    if 0 <= pos < len(df):
        candidates.append(pos)
    if 0 <= pos - 1 < len(df):
        candidates.append(pos - 1)

    best_idx = None
    best_dt = None
    for idx in candidates:
        dt = abs(df.at[idx, time_col] - t)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_idx = idx

    if best_idx is None or best_dt is None or best_dt > tol:
        return None
    return df.loc[best_idx]

def attach_validation(rec: dict, prefix: str, df: pd.DataFrame, tcol: str, ts: datetime, tol_minutes: int):
    row = nearest_row_to_tir(df, tcol, ts, tol_minutes)
    if row is None:
        rec[f"{prefix}_Time"] = ""
        return

    rec[f"{prefix}_Time"] = row[tcol].strftime("%Y-%m-%d %H:%M")
    for c in df.columns:
        if c == tcol:
            continue
        rec[f"{prefix}_{c}"] = row[c]

def series_to_numeric_robust(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s2 = s.astype("string")
    s2 = s2.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})
    s2 = s2.str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^0-9eE\.\-\+]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def make_daily_means(df: pd.DataFrame, tcol: str, value_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame(columns=["date", "daily_mean"])

    d = df[[tcol, value_col]].copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna(subset=[tcol]).sort_values(tcol)
    d[value_col] = series_to_numeric_robust(d[value_col])

    d = d.set_index(tcol)
    daily = d[value_col].resample("D").mean()
    out = daily.reset_index()
    out = out.rename(columns={tcol: "date", value_col: "daily_mean"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


# Load validation data
print("Loading validation datasets (radiometers, boreholes, GST)...")

validation_ridge_full, ridge_tcol = load_validation_df(RADIO_RIDGE_PATH)
validation_furrow_full, furrow_tcol = load_validation_df(RADIO_FURROW_PATH)

# Daily means from full radiometer datasets
ridge_daily = make_daily_means(validation_ridge_full, ridge_tcol, RADIO_COL)
furrow_daily = make_daily_means(validation_furrow_full, furrow_tcol, RADIO_COL)

# Keep only the requested radiometer column for nearest-match attachment
def keep_only_radiocol(df: pd.DataFrame, tcol: str, label: str) -> pd.DataFrame:
    if RADIO_COL not in df.columns:
        print(f"Warning: {label}: column '{RADIO_COL}' not found — keeping all columns.")
        return df
    return df[[tcol, RADIO_COL]].copy()

validation_ridge  = keep_only_radiocol(validation_ridge_full,  ridge_tcol,  "ridge")
validation_furrow = keep_only_radiocol(validation_furrow_full, furrow_tcol, "furrow")

validation_BH_1, bh1_tcol = load_validation_df(BH1_PATH)
validation_BH_2, bh2_tcol = load_validation_df(BH2_PATH)
validation_BH_3, bh3_tcol = load_validation_df(BH3_PATH)

gst_all, gst_tcol = load_validation_df(GST_PATH)

# All GST sensors share the same source file
validation_GST_1 = gst_all
validation_GST_2 = gst_all
validation_GST_3 = gst_all
validation_GST_4 = gst_all
validation_GST_5 = gst_all

print(f"Radiometers: ridge={len(validation_ridge)}, furrow={len(validation_furrow)}")
print(f"Radiometers daily: ridge_days={len(ridge_daily)}, furrow_days={len(furrow_daily)}")
print(f"Boreholes: BH1={len(validation_BH_1)}, BH2={len(validation_BH_2)}, BH3={len(validation_BH_3)}")
print(f"GST: {len(gst_all)}")


# Index uncorrected files by timestamp key
uncorr_paths = glob.glob(os.path.join(CSV_uncorrected_FOLDER, "*.csv"))
uncorr_by_key = {}
for p in uncorr_paths:
    key = extract_ts_key(os.path.basename(p))
    if key is None:
        continue
    if key not in uncorr_by_key:
        uncorr_by_key[key] = p

print(f"Indexed {len(uncorr_by_key)} uncorrected CSVs by timestamp key")


# Load emissivity
print(f"Loading emissivity values from {EMISSIVITY_PATH}")
emis_df = pd.read_csv(EMISSIVITY_PATH, sep=";", decimal=".")
emis_df["Date"] = pd.to_datetime(emis_df["Date"], errors="coerce")
emis_df = emis_df.dropna(subset=["Date", "E"]).sort_values("Date").reset_index(drop=True)
print(f"Loaded {len(emis_df)} emissivity records ({emis_df['Date'].min().date()} - {emis_df['Date'].max().date()})")

def get_emissivity(target_time: datetime) -> float:
    date_only = pd.Timestamp(target_time.date())
    emis_df["Δt"] = (emis_df["Date"] - date_only).abs()
    idx = emis_df["Δt"].idxmin()
    return float(emis_df.at[idx, "E"])


# Load meteo
print(f"Loading meteo data from {METEO_PATH}")
met_df = pd.read_csv(METEO_PATH, sep=",", low_memory=False)
met_df["TimeStamp"] = pd.to_datetime(met_df["TimeStamp"], errors="coerce")
met_df = met_df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").reset_index(drop=True)

# SWRup NaNs set to 0 (sensor reports no radiation during night/shade)
met_df["SWRup"] = pd.to_numeric(met_df["SWRup"], errors="coerce")
n_swrup_nan = int(met_df["SWRup"].isna().sum())
met_df["SWRup"] = met_df["SWRup"].fillna(0.0)
print(f"Meteo data loaded ({met_df['TimeStamp'].min()} - {met_df['TimeStamp'].max()})")
print(f"Replaced {n_swrup_nan} NaN values in SWRup with 0")

def read_nearest_met(met_df, target_time, emissivity):
    if pd.isna(target_time):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, pd.NaT

    met_df["Δt"] = (met_df["TimeStamp"] - target_time).abs()
    nearest = met_df.loc[met_df["Δt"].idxmin()]

    ta        = float(nearest.get("TA", np.nan))
    rh        = float(nearest.get("RH", np.nan))
    vwnd      = float(nearest.get("VWND1", np.nan))
    dwnd      = float(nearest.get("DWND1", np.nan))
    lwr_up    = float(nearest.get("LWRup", np.nan))
    lwr_down  = float(nearest.get("LWRdown", np.nan))
    swr_up    = float(nearest.get("SWRup", np.nan))
    swr_down  = float(nearest.get("SWRdown", np.nan))
    hs        = float(nearest.get("HS1", np.nan))
    tss       = float(nearest.get("TSS", np.nan))

    lw_temp = (lwr_down / (emissivity * SIGMA))
    if lw_temp > 0:
        lw_temp = lw_temp ** 0.25 - 273.15
    else:
        lw_temp = np.nan

    return ta, rh, vwnd, dwnd, lwr_down, lwr_up, swr_down, swr_up, hs, tss, lw_temp, nearest["TimeStamp"]


# Patch mean
def mean_patch(arr, x, y, radius=RADIUS):
    if not (0 <= x < W and 0 <= y < H):
        return np.nan
    y1, y2 = max(0, y - radius), min(H, y + radius + 1)
    x1, x2 = max(0, x - radius), min(W, x + radius + 1)
    patch = arr[y1:y2, x1:x2]
    if patch.size == 0 or np.all(np.isnan(patch)):
        return np.nan
    return float(np.nanmean(patch))


# Process corrected files
corr_files = sorted(glob.glob(os.path.join(CSV_corrected_FOLDER, "*_corr.csv")))
print(f"\nFound {len(corr_files)} corrected CSVs in {CSV_corrected_FOLDER}")

records = []
start_dt = datetime.fromisoformat(START_DATE)
end_dt   = datetime.fromisoformat(END_DATE)

# Daily lookup dicts for speed
ridge_daily_map = dict(zip(ridge_daily["date"], ridge_daily["daily_mean"])) if not ridge_daily.empty else {}
furrow_daily_map = dict(zip(furrow_daily["date"], furrow_daily["daily_mean"])) if not furrow_daily.empty else {}

daily_suffix = "_daily"

for f_corr in corr_files:
    fname_corr = os.path.basename(f_corr)
    ts = name_to_datetime(fname_corr)
    if ts is None:
        print(f"Skipping (no timestamp): {fname_corr}")
        continue

    if not (start_dt <= ts <= end_dt):
        continue

    key = extract_ts_key(fname_corr)
    if key is None:
        print(f"Skipping (no timestamp key): {fname_corr}")
        continue

    f_uncorr = uncorr_by_key.get(key)
    if not f_uncorr or not os.path.exists(f_uncorr):
        print(f"Missing uncorrected file for {fname_corr} (key={key})")
        continue

    try:
        arr_corr = pd.read_csv(
            f_corr, sep=";", header=None, decimal=".", skiprows=1
        ).to_numpy(dtype=np.float32)

        arr_uncorr = pd.read_csv(
            f_uncorr, sep=";", header=None, decimal=".", skiprows=8
        ).to_numpy(dtype=np.float32)

    except Exception as e:
        print(f"Error reading {fname_corr}: {e}")
        continue

    if arr_corr.shape != arr_uncorr.shape:
        print(f"Shape mismatch in {fname_corr}: corr={arr_corr.shape}, uncorr={arr_uncorr.shape}")
        continue

    E = get_emissivity(ts)
    ta, rh, vwnd, dwnd, lwr_down, lwr_up, swr_down, swr_up, hs, tss, lw_temp, ts_meteo = read_nearest_met(met_df, ts, E)

    rec = {
        "Time": ts.strftime("%Y-%m-%d %H:%M"),
        "Emissivity": E,
        "TA": ta,
        "RH": rh,
        "VWND": vwnd,
        "DWND": dwnd,
        "LWRdown": lwr_down,
        "LWRup": lwr_up,
        "SWRdown": swr_down,
        "SWRup": swr_up,
        "HS": hs,
        "TSS": tss,
        "LW_temp": lw_temp,
        "Date Meteo Station": ts_meteo.strftime("%Y-%m-%d %H:%M") if pd.notna(ts_meteo) else ""
    }

    # Patch means at all measurement points
    for name, (x, y) in POINTS.items():
        rec[f"{name}_uncorr"] = mean_patch(arr_uncorr, x, y)
        rec[f"{name}_corr"]   = mean_patch(arr_corr, x, y)

    # Attach reference datasets (nearest within tolerance)
    attach_validation(rec, "validation_ridge",  validation_ridge,  ridge_tcol,  ts, MATCH_TOL_MIN)
    attach_validation(rec, "validation_furrow", validation_furrow, furrow_tcol, ts, MATCH_TOL_MIN)

    attach_validation(rec, "validation_BH_1", validation_BH_1, bh1_tcol, ts, MATCH_TOL_MIN)
    attach_validation(rec, "validation_BH_2", validation_BH_2, bh2_tcol, ts, MATCH_TOL_MIN)
    attach_validation(rec, "validation_BH_3", validation_BH_3, bh3_tcol, ts, MATCH_TOL_MIN)

    attach_validation(rec, "validation_GST", gst_all, gst_tcol, ts, MATCH_TOL_MIN)

    # Attach radiometer daily means
    dkey = ts.date()
    rec[f"validation_ridge_{RADIO_COL}{daily_suffix}"]  = ridge_daily_map.get(dkey, np.nan)
    rec[f"validation_furrow_{RADIO_COL}{daily_suffix}"] = furrow_daily_map.get(dkey, np.nan)

    records.append(rec)
    print(f"Processed: {ts.strftime('%Y-%m-%d %H:%M')}")

# Save output
out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False, sep=";", float_format="%.3f")

print(f"\nSaved {len(records)} records to {OUT_CSV}")