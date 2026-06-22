from __future__ import annotations
import numpy as np
import pandas as pd
import os, glob, re
import matplotlib.pyplot as plt
import math

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
RAW_FOLDER   = os.path.join(DATA_DIR, "decoded_images_filtered")
OUT_FOLDER   = os.path.join(DATA_DIR, "corrected")
OD_PATH      = os.path.join(DATA_DIR, "CSVs", "murtel_distance_filled.csv")
METEO_PATH   = os.path.join(DATA_DIR, "CSVs", "murtel_met_qc.csv")

# Date range filter (set to None to disable)
START_DATE = None   # YYYY-MM-DD
START_TIME = None        # HH:MM
END_DATE   = None
END_TIME   = None

W, H = 336, 252

THRESH_HS  = 10.0
MIN_DAYS   = 3
TRANS_DAYS = 11
HALF_WIN   = 5

E_SNOW  = 0.98
E_BARE  = 0.94
DELTA_E = (E_SNOW - E_BARE) / TRANS_DAYS

# Debug
PROCESS_ONLY_FIRST_N_FILES = None   # set None to process all files

os.makedirs(OUT_FOLDER, exist_ok=True)

# Helpers
SIGMA = 5.670374419e-8  # W m^-2 K^-4

def lwr_to_temp_c(lwr_wm2: float) -> float:
    if not np.isfinite(lwr_wm2) or lwr_wm2 <= 0:
        return np.nan
    return float((lwr_wm2 / SIGMA) ** 0.25 - 273.15)


def read_nearest_met(csv_path, date_str, time_str):
    target = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
    if pd.isna(target):
        return np.nan, np.nan, np.nan, np.nan, pd.NaT

    df = pd.read_csv(csv_path, sep=",", low_memory=False)
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")
    df["Δt"] = (df["TimeStamp"] - target).abs()
    row = df.loc[df["Δt"].idxmin()]

    TA   = float(pd.to_numeric(row.get("TA",   np.nan), errors="coerce"))
    RH   = float(pd.to_numeric(row.get("RH",   np.nan), errors="coerce"))
    HS1  = float(pd.to_numeric(row.get("HS1",  np.nan), errors="coerce"))
    LWRu = float(pd.to_numeric(row.get("LWRup", np.nan), errors="coerce"))
    LW_temp = lwr_to_temp_c(LWRu)

    return TA, RH, HS1, LW_temp, row["TimeStamp"]



# Physical constants
h = 6.62607015e-34       # Planck constant [J s]
c = 299792458.0         # speed of light [m/s]
kB = 1.380649e-23       # Boltzmann constant [J/K]


# Planck spectral radiance

def planck_L_lambda(T_K: np.ndarray, lam_m: np.ndarray) -> np.ndarray:
    lam = lam_m.astype(np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    T = np.maximum(T, 1e-9)

    exponent = (h * c) / (lam[None, ...] * kB * T[..., None])
    denom = np.expm1(exponent)

    L = (2.0 * h * c**2) / (lam[None, ...]**5) / denom
    return L


# Temperature to band-integrated radiance

def tempC_to_radiance_band(
    temp_C,
    lambda_min_um: float = 7.5,
    lambda_max_um: float = 13.5,
    n_lambda: int = 400,
):
    # Scalar path
    if np.isscalar(temp_C):
        if not np.isfinite(temp_C):
            return np.nan
        T_K = float(temp_C) + 273.15
        lam = np.linspace(lambda_min_um * 1e-6, lambda_max_um * 1e-6, n_lambda)
        # planck_L_lambda expects array-like T_K; make it 1-element
        L = planck_L_lambda(np.array([T_K], dtype=np.float64), lam)  # (1, Nlam)
        rad = float(np.trapezoid(L[0], lam))
        return rad

    # Array path
    temp_C = np.asarray(temp_C, dtype=np.float64)

    if temp_C.ndim != 2:
        raise ValueError(f"temp_C must be scalar or 2D array, got shape {temp_C.shape}")

    out = np.full_like(temp_C, np.nan, dtype=np.float64)

    valid = np.isfinite(temp_C)
    if not np.any(valid):
        return out

    T_K = temp_C[valid] + 273.15
    lam = np.linspace(lambda_min_um * 1e-6, lambda_max_um * 1e-6, n_lambda)

    L = planck_L_lambda(T_K, lam)          # (Npix, Nlam)
    rad_valid = np.trapezoid(L, lam, axis=-1)  # integrate over wavelength

    out[valid] = rad_valid
    return out


# Band-integrated radiance to temperature (numerical inversion)

def _band_radiance_from_T(
    T_K: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    L = planck_L_lambda(T_K, lam)
    return np.trapezoid(L, lam, axis=-1)


def radiance_band_to_tempC(
    radiance: np.ndarray,
    lambda_min_um: float = 7.5,
    lambda_max_um: float = 13.5,
    n_lambda: int = 400,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    if radiance.ndim != 2:
        raise ValueError(f"radiance must be 2D, got shape {radiance.shape}")

    radiance = np.asarray(radiance, dtype=np.float64)
    out = np.full_like(radiance, np.nan, dtype=np.float64)

    valid = np.isfinite(radiance)
    if not np.any(valid):
        return out

    lam = np.linspace(lambda_min_um * 1e-6, lambda_max_um * 1e-6, n_lambda)

    R = radiance[valid]

    # Reasonable physical bounds for rock glacier surface
    T_low = np.full_like(R, 150.0)   # 150 K  (~ -123 °C)
    T_high = np.full_like(R, 400.0)  # 400 K  (~ 127 °C)

    for _ in range(max_iter):
        T_mid = 0.5 * (T_low + T_high)
        R_mid = _band_radiance_from_T(T_mid, lam)

        mask = R_mid > R
        T_high[mask] = T_mid[mask]
        T_low[~mask] = T_mid[~mask]

        if np.max(np.abs(R_mid - R)) < tol:
            break

    T_final = 0.5 * (T_low + T_high)
    out[valid] = T_final - 273.15
    return out

# Atmospheric functions

def pressure_at_elevation(h):
    P0 = 101325
    T0 = 288.15
    L = 0.0065
    g = 9.80665
    M = 0.0289644
    R = 8.31447

    exponent = (g * M) / (R * L)
    P = P0 * (1 - (L * h) / T0) ** exponent
    return P

def vapor_density(T_air, RH, P):
    a = 610.94
    b = 17.625
    c = 243.04

    es_SL = a * math.exp(b * T_air / (c + T_air))
    es = es_SL * (P / 101325)
    ea = es * (RH / 100)

    M_w = 18.015
    R = 8.314472
    T_K = T_air + 273.15

    rho_v = (ea * M_w) / (R * T_K)
    return rho_v

def atm_trans(dist, rho_v):
    alpha1 = 0.00656899996101856
    alpha2 = 0.0126200001686811
    beta1 = -0.00227600010111928
    beta2 = -0.00667000003159046
    X = 1.89999997

    dist = np.asarray(dist, dtype=np.float32)
    dist = np.maximum(dist, 0.0)

    sdist = np.sqrt(dist)
    srho  = np.sqrt(max(float(rho_v), 0.0))

    term1 = X * np.exp(-sdist * (alpha1 + beta1 * srho))
    term2 = (1.0 - X) * np.exp(-sdist * (alpha2 + beta2 * srho))
    tau = term1 + term2

    return np.clip(tau, 0.0, 1.0)

# Core correction
def raw2temp_arr(raw, E=0.94, OD=1, ATemp=20, LW_down=np.nan, IRT=0.9, RH=50):

    IRWTemp = ATemp
    RTemp = LW_down
    emiss_wind = 1 - IRT
    refl_wind = 0

    P = pressure_at_elevation(2702)
    rho = vapor_density(ATemp, RH, P)
    tau1 = atm_trans(OD, rho)
    tau2 = tau1
    print("INPUT raw stats (before any conversion):",
      float(np.nanmin(raw)), float(np.nanmean(raw)), float(np.nanmax(raw)))
    
    raw = tempC_to_radiance_band(raw)

    raw_refl1 = tempC_to_radiance_band(RTemp)
    raw_refl1_attn = (1 - E) / E * raw_refl1

    raw_atm1 = tempC_to_radiance_band(ATemp)
    raw_atm1_attn = (1 - tau1) / (E * tau1) * raw_atm1

    raw_wind = tempC_to_radiance_band(IRWTemp)
    raw_wind_attn = emiss_wind / (E * tau1 * IRT) * raw_wind

    raw_refl2 = raw_refl1
    raw_refl2_attn = refl_wind / (E * tau1 * IRT) * raw_refl2

    raw_atm2 = raw_atm1
    raw_atm2_attn = (1 - tau2) / (E * tau1 * IRT * tau2) * raw_atm2

    raw_obj = (raw / (E * tau1 * IRT * tau2) -
               raw_atm1_attn - raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)
    
    temp_celsius = radiance_band_to_tempC(raw_obj)
    print("OUTPUT stats:",
      float(np.nanmin(temp_celsius)), float(np.nanmean(temp_celsius)), float(np.nanmax(temp_celsius)))
    return temp_celsius

# Daily snow depth and emissivity
met = pd.read_csv(METEO_PATH, sep=",", low_memory=False)
met["TimeStamp"] = pd.to_datetime(met["TimeStamp"], errors="coerce")
met["HS1"] = pd.to_numeric(met["HS1"], errors="coerce")
met = met.dropna(subset=["TimeStamp", "HS1"])
met["Date"] = met["TimeStamp"].dt.floor("D")

daily = (
    met.groupby("Date", as_index=False)["HS1"].mean()
      .rename(columns={"HS1": "HS1_mean"})
      .sort_values("Date")
      .reset_index(drop=True)
)

state = np.where(daily["HS1_mean"] >= THRESH_HS, 1, 0)

state_filt = state.copy()
for i in range(1, len(state)):
    if state[i] != state_filt[i-1]:
        end = min(len(state), i + MIN_DAYS)
        if np.all(state[i:end] == state[i]):
            state_filt[i] = state[i]
        else:
            state_filt[i] = state_filt[i-1]
    else:
        state_filt[i] = state_filt[i-1]

switches = []
for i in range(1, len(state_filt)):
    if state_filt[i] != state_filt[i-1]:
        switches.append(i)

print("\nDetected valid HS1 switches:")
for i in switches:
    prev = "snow" if state_filt[i-1] == 1 else "no snow"
    now  = "snow" if state_filt[i] == 1 else "no snow"
    print(f"  {daily.at[i,'Date'].date()}: {prev} → {now}")

E = np.zeros(len(daily), dtype=np.float32)
E[0] = E_SNOW if state_filt[0] == 1 else E_BARE

daily_delta = np.zeros(len(daily), dtype=np.float32)
for i in switches:
    sign = +1 if state_filt[i] == 1 else -1
    for j in range(i - HALF_WIN, i + HALF_WIN + 1):
        if 0 <= j < len(daily):
            daily_delta[j] += sign * DELTA_E

for i in range(1, len(E)):
    E[i] = E[i-1] + daily_delta[i]
    E[i] = min(E_SNOW, max(E_BARE, float(E[i])))

daily["E"] = E

emissivity_csv = os.path.join(OUT_FOLDER, "daily_emissivity.csv")
daily.to_csv(emissivity_csv, sep=";", index=False, float_format="%.4f")
print(f"\nSaved daily emissivity CSV: {emissivity_csv}")

plt.figure(figsize=(9, 4))
plt.plot(daily["Date"], daily["E"], marker="o", ms=3)
plt.ylabel("Emissivity")
plt.title("Daily emissivity evolution (21-day symmetric transitions)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# Process images
distances = pd.read_csv(OD_PATH).to_numpy(dtype=np.float32).reshape(H, W)
E_by_date = dict(zip(daily["Date"], daily["E"]))

files = sorted(glob.glob(os.path.join(RAW_FOLDER, "*.csv")))
print(f"\nFound {len(files)} TIR CSVs.")

# Apply date range filter
def parse_dt_from_filename(fname: str) -> pd.Timestamp | None:
    m = re.search(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", fname)
    if not m:
        return None
    yy, MM, DD, hh, mm = map(int, m.groups())
    return pd.Timestamp(f"20{yy:02d}-{MM:02d}-{DD:02d} {hh:02d}:{mm:02d}")

def to_ts(d, t):
    if d is None:
        return None
    if t is None:
        t = "00:00"
    return pd.to_datetime(f"{d} {t}", errors="coerce")

start_ts = to_ts(START_DATE, START_TIME)
end_ts   = to_ts(END_DATE, END_TIME)

if start_ts is not None:
    before = len(files)
    files = [f for f in files
             if (parse_dt_from_filename(os.path.basename(f)) is not None and
                 parse_dt_from_filename(os.path.basename(f)) >= start_ts)]
    print(f"Applied START filter >= {start_ts}: {before} -> {len(files)} files")

if end_ts is not None:
    before = len(files)
    files = [f for f in files
             if (parse_dt_from_filename(os.path.basename(f)) is not None and
                 parse_dt_from_filename(os.path.basename(f)) <= end_ts)]
    print(f"Applied END filter <= {end_ts}: {before} -> {len(files)} files")

if PROCESS_ONLY_FIRST_N_FILES is not None:
    files = files[:int(PROCESS_ONLY_FIRST_N_FILES)]
    print(f"Limiting to first {len(files)} files.")

for f in files:
    fname = os.path.basename(f)
    m = re.search(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", fname)
    if not m:
        continue

    yy, MM, DD, hh, mm = map(int, m.groups())
    dt = pd.Timestamp(f"20{yy:02d}-{MM:02d}-{DD:02d} {hh:02d}:{mm:02d}")
    d = dt.floor("D")

    if d not in E_by_date:
        idx = (daily["Date"] - d).abs().idxmin()
        E_use = float(daily.at[idx, "E"])
    else:
        E_use = float(E_by_date[d])

    TA, RH, _, LW_temp, _ = read_nearest_met(METEO_PATH, d.date().isoformat(), f"{hh:02d}:{mm:02d}")

    raw = pd.read_csv(f, header=None, sep=";", skiprows=8).to_numpy(dtype=np.float32).reshape(H, W)

    print(f"\nProcessing {fname}  (E={E_use:.4f}, TA={TA:.2f}, RH={RH:.1f})")
    temp_corr = raw2temp_arr(raw, E=E_use, OD=distances, ATemp=TA, LW_down=LW_temp, RH=RH)

    out_csv = os.path.join(OUT_FOLDER, os.path.splitext(fname)[0] + "_corr.csv")
    pd.DataFrame(temp_corr).to_csv(out_csv, sep=";", index=False, float_format="%.6f", decimal=".")
    print(f"{dt.date()}  E={E_use:.4f}  saved: {os.path.basename(out_csv)}")

print("\nAll files processed.")
