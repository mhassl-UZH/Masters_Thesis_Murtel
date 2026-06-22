import os
import glob
import re
import shutil
from datetime import datetime, timedelta
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

TIR_CSV_FOLDER = os.path.join(DATA_DIR, "corrected")
METEO_CSV      = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")

ZNORM_ROOT = os.path.join(DATA_DIR, "analysis", "anomaly_persistence")
TIR_MASK_PATH = os.path.join(DATA_DIR, "TIR_mask.png")

OUT_ROOT_BASE = os.path.join(DATA_DIR, "analysis", "anomaly_persistence_regression")

H, W = 252, 336
ROLLING_WINDOW_DAYS = 5
MERGE_TOLERANCE_MIN = 10
MIN_POINTS_PER_WINDOW = 10
PRINT_EVERY_N_IMAGES = 50
PRINT_FIRST_N_SKIPS_PER_WINDOW = 5

MATCH_TOLERANCE_MIN_DAYNIGHT = 10
NIGHT_THRESHOLD = 10.0   # SWRup < 10 => night, else day

BASE_VARS = ["TA", "RH", "VWND", "LWRup", "SWRup", "DWND"]

# "ALL" or specific family names e.g. ["Snow_NoSnow"]
ANALYSIS_ORDER = [
    "Day_Night",
    "Rain_NoRain",
    "Snow_NoSnow",
    "Transition",
]

GIF_NAME = "daily_group_masks.gif"
GIF_FRAMES_SUBDIR = "gif_frames"
GIF_FPS = 1.0

# Analysis sets

ANALYSIS_CONFIGS = {
    "Snow_NoSnow": {
        "split_day_night": False,
        "ranges": [
            {"label": "2021_NoSnow", "title": "NoSnow 2021", "start": "05.08.21", "end": "23.08.21"},
            {"label": "2022_NoSnow", "title": "NoSnow 2022", "start": "07.06.22", "end": "19.07.22"},
            {"label": "2022_Snow", "title": "Snow 2022", "start": "26.02.22", "end": "01.04.22"},
        ],
    },


    "Transition": {
        "split_day_night": False,
        "ranges": [
            {"label": "2021_Spring", "title": "Spring 2021", "start": "14.06.21", "end": "25.07.21"},
            {"label": "2021_Autumn", "title": "Autumn 2021", "start": "13.09.21", "end": "14.11.21"},
            {"label": "2022_Spring", "title": "Spring 2022", "start": "09.05.22", "end": "26.06.22"},
            {"label": "2022_Autumn", "title": "Autumn 2022", "start": "31.10.22", "end": "18.12.22"},
            {"label": "2023_Spring", "title": "Spring 2023", "start": "15.05.23", "end": "02.07.23"},
        ],
    },

    "Rain_NoRain": {
        "split_day_night": False,
        "ranges": [
            {"label": "2022_NoSnow_NoRain", "title": "NoSnow NoRain 2022", "start": "05.07.22", "end": "22.07.22"},
            {"label": "2022_NoSnow_Rain", "title": "NoSnow Rain 2022", "start": "23.07.22", "end": "08.08.22"},
            {"label": "2022_Snow_NoRain", "title": "Snow NoRain 2022", "start": "24.02.22", "end": "28.03.22"},
            {"label": "2022_Snow_Rain", "title": "Snow Rain 2022", "start": "29.01.22", "end": "23.02.22"},
        ],
    },

    "Day_Night": {
        "split_day_night": True,
        "ranges": [
            {"label": "2022_NoSnow", "title": "NoSnow 2022", "start": "07.06.22", "end": "19.07.22"},
            {"label": "2022_Snow", "title": "Snow 2022", "start": "26.02.22", "end": "14.03.22"},
        ],
    },
}

# Helpers

TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

def name_to_dt(path_or_name: str):
    m = TS_RE.search(os.path.basename(path_or_name))
    if not m:
        return None
    yy, MM, DD, hh, mm = map(int, m.groups())
    return datetime(2000 + yy, MM, DD, hh, mm)

def parse_date_flexible(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {s}")

def build_daily_targets(start_dt: datetime, end_dt: datetime):
    targets = []
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end_day:
        targets.append(cur)
        cur += timedelta(days=1)
    return targets

def get_window_bounds(target_day: datetime, window_days: int):
    day_start = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = day_start - timedelta(days=window_days - 1)
    window_end = day_start.replace(hour=23, minute=59, second=59, microsecond=0)
    window_label = f"{window_start.strftime('%Y-%m-%d')}_to_{day_start.strftime('%Y-%m-%d')}"
    target_label = day_start.strftime("%Y-%m-%d")
    target_label_display = day_start.strftime("%d.%m.%Y")
    window_label_display = f"{window_start.strftime('%d.%m.%y')} - {day_start.strftime('%d.%m.%y')}"
    return window_start, window_end, window_label, target_label, target_label_display, window_label_display

def robust_read_meteo_csv(path: str) -> tuple[pd.DataFrame, str]:
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if "Time" in df.columns:
                return df, sep
        except Exception:
            pass
    raise RuntimeError("Could not read METEO_CSV or missing 'Time' column.")

def read_grid_csv(path: str, H: int, W: int, sep: str = ";") -> np.ndarray | None:
    try:
        df0 = pd.read_csv(path, sep=sep, header=None, skiprows=1, engine="python")
    except Exception as e:
        print(f"[READ FAIL] {os.path.basename(path)}: {e}")
        return None
    arr = df0.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if arr.shape != (H, W):
        print(f"[SHAPE FAIL] {os.path.basename(path)} got {arr.shape}, expected {(H, W)}")
        return None
    return arr

def zscore_df(X: pd.DataFrame, eps: float = 1e-12):
    means = X.mean(axis=0, skipna=True)
    stds  = X.std(axis=0, skipna=True, ddof=0)
    dropped = stds[stds <= eps].index.tolist()
    keep_cols = [c for c in X.columns if c not in dropped]
    Xk = X[keep_cols].copy()
    mk = means[keep_cols]
    sk = stds[keep_cols]
    Xz = (Xk - mk) / sk
    return Xz, means, stds, dropped

def safe_mean_over_mask(grid: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    vals = grid[mask]
    n = int(np.isfinite(vals).sum())
    if n == 0:
        return np.nan, 0
    return float(np.nanmean(vals)), n

def get_param_value(series: pd.Series, key: str, default=np.nan):
    return float(series[key]) if key in series.index else default

def find_count_above_target_csv(range_folder: str, effective_label: str, target_label: str) -> Optional[str]:
    patt = os.path.join(
        range_folder,
        f"count_above_0_{effective_label}_target_{target_label}_window_*_n*.csv"
    )
    matches = sorted(glob.glob(patt))
    if not matches:
        return None
    return matches[0]

def load_day_night_table_from_meteo(met: pd.DataFrame) -> pd.DataFrame:
    df = met[["Time", "SWRup"]].dropna(subset=["Time", "SWRup"]).copy()
    df = df.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df

def attach_day_night_flag(df_files: pd.DataFrame, df_ref: pd.DataFrame, tol_min: int, night_threshold: float) -> pd.DataFrame:
    out = pd.merge_asof(
        df_files.sort_values("Time"),
        df_ref.rename(columns={"Time": "ref_time"}).sort_values("ref_time"),
        left_on="Time",
        right_on="ref_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tol_min),
    )
    out = out.dropna(subset=["SWRup"]).copy()
    out["day_night"] = np.where(out["SWRup"] < night_threshold, "night", "day")
    return out

def load_group_masks_from_counts(count_above_csv: str, tir_mask: np.ndarray) -> tuple[dict, dict]:
    count_above = pd.read_csv(count_above_csv, sep=";", header=None, engine="python").apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    if count_above.shape != (H, W):
        raise RuntimeError(f"[COUNTS] {os.path.basename(count_above_csv)} has shape {count_above.shape}, expected {(H, W)}")

    valid = tir_mask & np.isfinite(count_above)
    vals = count_above[valid]

    if vals.size < 10:
        raise RuntimeError("[COUNTS] Too few valid pixels to form quantiles.")

    q25 = float(np.nanquantile(vals, 0.25))
    q75 = float(np.nanquantile(vals, 0.75))

    mask_cold = valid & (count_above <= q25)
    mask_warm = valid & (count_above >= q75)
    mask_mid  = valid & (~mask_cold) & (~mask_warm)

    if mask_cold.sum() == 0 or mask_mid.sum() == 0 or mask_warm.sum() == 0:
        raise RuntimeError("[GROUPS] One of the quantile masks is empty.")

    groups = {
        "Tcold": mask_cold,
        "Tmid":  mask_mid,
        "Twarm": mask_warm,
    }
    meta = {
        "q25": q25,
        "q75": q75,
        "cold_pixels": int(mask_cold.sum()),
        "mid_pixels": int(mask_mid.sum()),
        "warm_pixels": int(mask_warm.sum()),
    }
    return groups, meta

def save_bundle(out_dir: str, results: list, timeseries_list: list, quality: list):
    os.makedirs(out_dir, exist_ok=True)

    res_df = pd.DataFrame(results)
    res_path = os.path.join(out_dir, "daily_window_group_regression_qtiles.csv")
    res_df.to_csv(res_path, index=False)
    print("[SAVE]", res_path, f"({len(res_df)} rows)")

    if timeseries_list:
        ts_df = pd.concat(timeseries_list, ignore_index=True)
        ts_path = os.path.join(out_dir, "daily_window_group_timeseries_qtiles.csv")
        ts_df.to_csv(ts_path, index=False)
        print("[SAVE]", ts_path, f"({len(ts_df)} rows)")
    else:
        print("[WARN] No timeseries to save into:", out_dir)

    q_df = pd.DataFrame(quality)
    q_path = os.path.join(out_dir, "daily_window_quality_report_qtiles.csv")
    q_df.to_csv(q_path, index=False)
    print("[SAVE]", q_path, f"({len(q_df)} rows)")

def build_group_code_array(groups: dict, tir_mask: np.ndarray) -> np.ndarray:
    arr = np.full((H, W), np.nan, dtype=np.float32)
    arr[tir_mask] = 0
    arr[groups["Tcold"]] = 1
    arr[groups["Tmid"]]  = 2
    arr[groups["Twarm"]] = 3
    return arr

def render_group_frame_png(code_arr: np.ndarray, target_label: str, window_label: str, out_png: str,
                           target_label_display: str = None, window_label_display: str = None):
    cmap = ListedColormap([
        "#E6E6E6",  # valid but unclassified background
        "#2C7BB6",  # cold
        "#FFFFBF",  # mid
        "#D7191C",  # warm
    ])

    display_arr = np.where(np.isnan(code_arr), -1, code_arr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(display_arr, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")

    t_disp = target_label_display if target_label_display else target_label
    w_disp = window_label_display if window_label_display else window_label
    ax.set_title(f"Persistence Classes (5-Day Window, {t_disp})")
    ax.set_xticks([])
    ax.set_yticks([])

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#2C7BB6", edgecolor="black", label="Persistently Cold"),
        Patch(facecolor="#FFFFBF", edgecolor="black", label="Intermediate"),
        Patch(facecolor="#D7191C", edgecolor="black", label="Persistently Warm"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_group_gif(frames_info: list, out_gif: str):
    if not frames_info:
        print("[GIF] No frames available, GIF not created:", out_gif)
        return

    images = []
    for item in frames_info:
        try:
            images.append(imageio.imread(item["png"]))
        except Exception as e:
            print(f"[GIF] Could not read frame {item['png']}: {e}")

    if not images:
        print("[GIF] No readable images for GIF:", out_gif)
        return

    duration = 1.0 / GIF_FPS if GIF_FPS > 0 else 1.0
    imageio.mimsave(out_gif, images, duration=duration, loop=0)
    print("[SAVE]", out_gif)

# Load TIR mask

mask_img = plt.imread(TIR_MASK_PATH)
if mask_img.ndim == 3:
    mask_img = mask_img[..., 0]
tir_mask = mask_img > 0.5

if tir_mask.shape != (H, W):
    raise RuntimeError(f"TIR mask has shape {tir_mask.shape}, expected {(H, W)}")

print(f"[MASK] Valid pixels: {int(tir_mask.sum())} / {H*W}")

# Load meteo

print("\n=== LOADING METEO ===")
met, met_sep = robust_read_meteo_csv(METEO_CSV)
print(f"[METEO] Read {len(met)} rows with sep='{met_sep}'")

met["Time"] = pd.to_datetime(met["Time"], errors="coerce")
met = met.dropna(subset=["Time"]).copy()

missing = [c for c in BASE_VARS if c not in met.columns]
if missing:
    raise RuntimeError(f"[METEO] Missing required columns: {missing}")

for c in BASE_VARS:
    met[c] = pd.to_numeric(met[c], errors="coerce")

rad = np.deg2rad(met["DWND"])
met["DWND_sin"] = np.sin(rad)
met["DWND_cos"] = np.cos(rad)

METEO_VARS = ["TA", "RH", "VWND", "LWRup", "SWRup", "DWND_sin", "DWND_cos"]
met = met.sort_values("Time").reset_index(drop=True)
print("[METEO] Time range:", met["Time"].min(), "->", met["Time"].max())

df_daynight_ref = load_day_night_table_from_meteo(met)
print(f"[DAY/NIGHT REF] Loaded {len(df_daynight_ref)} reference rows from meteo Time + SWRup.")

# Index TIR grids

print("\n=== INDEXING TIR GRIDS ===")
tir_files = sorted(glob.glob(os.path.join(TIR_CSV_FOLDER, "*_corr.csv")))
print(f"[TIR] Found {len(tir_files)} *_corr.csv files in {TIR_CSV_FOLDER}")

tir_records = []
no_ts = 0
for p in tir_files:
    dt = name_to_dt(p)
    if dt is None:
        no_ts += 1
        continue
    tir_records.append((p, dt))

tir_df_all = pd.DataFrame(tir_records, columns=["path", "Time"])
print(f"[TIR] Parsed {len(tir_df_all)} files. Unparsed timestamps: {no_ts}")
if len(tir_df_all) == 0:
    raise RuntimeError("[TIR] No TIR grids found.")

# Main analysis

all_master_rows = []
window_summary_rows = []

for analysis_name in ANALYSIS_ORDER:
    if analysis_name not in ANALYSIS_CONFIGS:
        print(f"[WARN] Unknown analysis family skipped: {analysis_name}")
        continue

    cfg = ANALYSIS_CONFIGS[analysis_name]
    analysis_ranges = cfg["ranges"]
    split_day_night = cfg["split_day_night"]

    counts_analysis_root = os.path.join(ZNORM_ROOT, analysis_name)
    out_analysis_root = os.path.join(OUT_ROOT_BASE, analysis_name)
    os.makedirs(out_analysis_root, exist_ok=True)

    print("\n" + "#" * 90)
    print(f"RUNNING ANALYSIS FAMILY: {analysis_name}")
    print(f"rolling_window_days = {ROLLING_WINDOW_DAYS}")
    print(f"split_day_night     = {split_day_night}")
    print("#" * 90)

    family_master_rows = []

    for range_idx, range_info in enumerate(analysis_ranges, start=1):
        base_label = range_info["label"]
        base_title = range_info["title"]
        start_dt = parse_date_flexible(range_info["start"]).replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt   = parse_date_flexible(range_info["end"]).replace(hour=23, minute=59, second=59, microsecond=0)

        print(f"\n{'='*78}")
        print(f"{analysis_name} | RANGE {range_idx}/{len(analysis_ranges)}: {base_title} [{start_dt.date()} -> {end_dt.date()}]")
        print(f"{'='*78}")

        target_days = build_daily_targets(start_dt, end_dt)
        earliest_window_start = start_dt - timedelta(days=ROLLING_WINDOW_DAYS - 1)

        df_range = tir_df_all[(tir_df_all["Time"] >= earliest_window_start) & (tir_df_all["Time"] <= end_dt)].copy()

        if len(df_range) == 0:
            print("[RANGE] No TIR files in date range including rolling pre-window days, skipping.")
            continue

        if split_day_night:
            df_range = attach_day_night_flag(
                df_range,
                df_daynight_ref,
                MATCH_TOLERANCE_MIN_DAYNIGHT,
                NIGHT_THRESHOLD
            )
            phase_items = [("day", "_day"), ("night", "_night")]
            print(f"[RANGE] Matched files after day/night merge: {len(df_range)}")
            if len(df_range) > 0:
                print(df_range["day_night"].value_counts(dropna=False).to_dict())
        else:
            phase_items = [(None, "")]

        for phase_name, phase_suffix in phase_items:
            if phase_name is None:
                df_phase = df_range.copy()
                effective_label = base_label
                effective_title = base_title
            else:
                df_phase = df_range[df_range["day_night"] == phase_name].copy()
                effective_label = f"{base_label}{phase_suffix}"
                effective_title = f"{base_title} {phase_name.capitalize()}"

            if len(df_phase) == 0:
                print(f"[PHASE] {effective_title}: 0 files, skipping.")
                continue

            counts_range_folder = os.path.join(counts_analysis_root, effective_label)

            print(f"[PHASE] {effective_title}: {len(df_phase)} files available for rolling windows")
            print(f"[COUNTS DIR] {counts_range_folder}")

            target_days_to_run = target_days
            print(f"[PHASE] Target days: {len(target_days_to_run)}")

            out_range_root = os.path.join(out_analysis_root, effective_label)

            if os.path.exists(out_range_root):
                shutil.rmtree(out_range_root)
                print(f"[CLEAN] Removed existing folder: {out_range_root}")

            os.makedirs(out_range_root, exist_ok=True)

            std_results = []
            std_timeseries = []
            quality_rows = []

            frames_dir = os.path.join(out_range_root, GIF_FRAMES_SUBDIR)
            os.makedirs(frames_dir, exist_ok=True)
            gif_frames = []

            for wi, target_day in enumerate(target_days_to_run, start=1):
                window_start, window_end, window_label, target_label, target_label_display, window_label_display = get_window_bounds(target_day, ROLLING_WINDOW_DAYS)

                window_imgs = df_phase[
                    (df_phase["Time"] >= window_start) &
                    (df_phase["Time"] <= window_end)
                ].sort_values("Time").reset_index(drop=True)

                print(f"\n=== TARGET DAY {target_label} ({wi}/{len(target_days_to_run)}) | {effective_title} ===")
                print(f"[WINDOW] {window_label}")
                print(f"[WINDOW] TIR images in rolling window: {len(window_imgs)}")

                summary_idx = len(window_summary_rows)
                window_summary_rows.append({
                    "analysis_name": analysis_name,
                    "range_label": effective_label,
                    "range_title": effective_title,
                    "target_day": target_label,
                    "window_label": window_label,
                    "window_number": wi,
                    "n_acquisitions": int(len(window_imgs)),
                    "n_points_after_merge": 0,
                })

                count_above_csv = find_count_above_target_csv(counts_range_folder, effective_label, target_label)

                if count_above_csv is None:
                    print(f"[WINDOW] Missing target-day-specific count_above file for target day: {target_label}")
                    continue

                print(f"[COUNTS] Using target-day file: {count_above_csv}")

                try:
                    groups, group_meta = load_group_masks_from_counts(count_above_csv, tir_mask)
                except Exception as e:
                    print(f"[COUNTS FAIL] {effective_label} | target_day={target_label}: {e}")
                    continue

                print(f"[GROUPS] q25={group_meta['q25']:.3f}, q75={group_meta['q75']:.3f}")
                print(f"[GROUPS] cold={group_meta['cold_pixels']} mid={group_meta['mid_pixels']} warm={group_meta['warm_pixels']}")

                try:
                    code_arr = build_group_code_array(groups, tir_mask)
                    safe_window = window_label.replace(":", "-").replace("/", "-").replace("\\", "-")
                    frame_png = os.path.join(
                        frames_dir,
                        f"{wi:04d}_target_{target_label}__window_{safe_window}.png"
                    )

                    render_group_frame_png(code_arr, target_label, window_label, frame_png,
                                           target_label_display, window_label_display)
                    gif_frames.append({
                        "target_day": target_label,
                        "window_label": window_label,
                        "png": frame_png,
                    })
                except Exception as e:
                    print(f"[GIF FRAME FAIL] {effective_label} | {target_label}: {e}")

                if len(window_imgs) < 2:
                    print("[WINDOW] Skipping: <2 images")
                    continue

                rows = []
                skip_read = 0
                skip_nanheavy = 0
                skip_empty_any = 0
                shown_skips = 0

                for ii, r in window_imgs.iterrows():
                    p = r["path"]
                    dt = r["Time"]

                    grid = read_grid_csv(p, H, W, sep=";")
                    if grid is None:
                        skip_read += 1
                        if shown_skips < PRINT_FIRST_N_SKIPS_PER_WINDOW:
                            print(f"[SKIP READ] {os.path.basename(p)}")
                            shown_skips += 1
                        continue

                    nan_frac = float(np.isnan(grid).mean())
                    if nan_frac > 0.80:
                        skip_nanheavy += 1
                        if shown_skips < PRINT_FIRST_N_SKIPS_PER_WINDOW:
                            print(f"[SKIP NaN>80%] {os.path.basename(p)} nan_frac={nan_frac:.2f}")
                            shown_skips += 1
                        continue

                    out = {
                        "Time": dt,
                        "target_day": target_label,
                        "window_label": window_label,
                        "group_source_target_day": target_label,
                        "group_source_window_label": window_label,
                        "tir_path": p,
                        "nan_frac": nan_frac
                    }
                    empty_flag = False

                    for gname, gmask in groups.items():
                        mval, nfin = safe_mean_over_mask(grid, gmask)
                        out[gname] = mval
                        out[f"n_{gname}"] = nfin
                        if nfin == 0:
                            empty_flag = True

                    if empty_flag:
                        skip_empty_any += 1
                        if shown_skips < PRINT_FIRST_N_SKIPS_PER_WINDOW:
                            print(
                                f"[SKIP EMPTY GROUP] {os.path.basename(p)} "
                                f"n_Tcold={out['n_Tcold']} n_Tmid={out['n_Tmid']} n_Twarm={out['n_Twarm']} nan_frac={nan_frac:.2f}"
                            )
                            shown_skips += 1
                        continue

                    rows.append(out)

                    if (ii + 1) % PRINT_EVERY_N_IMAGES == 0:
                        print(f"[PROGRESS] {ii+1}/{len(window_imgs)} images processed, kept={len(rows)}")

                print(f"[WINDOW] Kept images: {len(rows)} / {len(window_imgs)}")
                print(f"[WINDOW] Skips: read_fail={skip_read}, nan_heavy={skip_nanheavy}, empty_group={skip_empty_any}")

                quality_rows.append({
                    "analysis_name": analysis_name,
                    "range_label": effective_label,
                    "range_title": effective_title,
                    "base_label": base_label,
                    "phase": phase_name if phase_name is not None else "",
                    "rolling_window_days": ROLLING_WINDOW_DAYS,
                    "target_day": target_label,
                    "window_label": window_label,
                    "window_number": wi,
                    "count_above_csv": count_above_csv,
                    "n_window_imgs": int(len(window_imgs)),
                    "n_kept": int(len(rows)),
                    "skip_read": int(skip_read),
                    "skip_nanheavy": int(skip_nanheavy),
                    "skip_empty_group": int(skip_empty_any),
                    "cold_pixels": group_meta["cold_pixels"],
                    "mid_pixels": group_meta["mid_pixels"],
                    "warm_pixels": group_meta["warm_pixels"],
                    "q25": group_meta["q25"],
                    "q75": group_meta["q75"],
                })

                if len(rows) < MIN_POINTS_PER_WINDOW:
                    print(f"[WINDOW] Skipping regression: only {len(rows)} valid points (<{MIN_POINTS_PER_WINDOW})")
                    continue

                ts = pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)

                merged = pd.merge_asof(
                    ts,
                    met[["Time"] + METEO_VARS],
                    on="Time",
                    direction="nearest",
                    tolerance=pd.Timedelta(minutes=MERGE_TOLERANCE_MIN),
                )

                before = len(merged)
                merged = merged.dropna(subset=METEO_VARS).copy()
                after = len(merged)
                print(f"[MERGE] Rows before dropna: {before}, after dropna: {after}")

                window_summary_rows[summary_idx]["n_points_after_merge"] = int(after)

                if after < MIN_POINTS_PER_WINDOW:
                    print(f"[WINDOW] Skipping regression after merge: {after} points (<{MIN_POINTS_PER_WINDOW})")
                    continue

                keep_cols = [
                    "target_day", "window_label",
                    "group_source_target_day", "group_source_window_label",
                    "Time",
                    "Tcold", "Tmid", "Twarm",
                    "nan_frac", "n_Tcold", "n_Tmid", "n_Twarm"
                ] + METEO_VARS

                std_timeseries.append(merged[keep_cols])

                for yname in ["Tcold", "Tmid", "Twarm"]:
                    y_raw = merged[yname].astype(float)
                    X_raw = merged[METEO_VARS].astype(float)

                    if float(np.nanstd(y_raw)) == 0.0:
                        print(f"[REG] target_day={target_label} {yname}: zero variance in y, skipping standardized")
                        continue

                    Xz, x_means, x_stds, dropped_x = zscore_df(X_raw)
                    y_mean = float(y_raw.mean())
                    y_std  = float(y_raw.std(ddof=0))

                    if y_std <= 1e-12:
                        print(f"[REG STD] target_day={target_label} {yname}: y_std ~ 0, skipping standardized")
                        continue

                    yz = (y_raw - y_mean) / y_std

                    if Xz.shape[1] == 0:
                        print(f"[REG STD] target_day={target_label} {yname}: all predictors constant (dropped={dropped_x}), skipping standardized")
                        continue

                    Xs = sm.add_constant(Xz, has_constant="add")
                    model_std = sm.OLS(yz, Xs, missing="drop").fit()
                    print(f"[REG STD] target_day={target_label} {yname}: n={int(model_std.nobs)} R2={model_std.rsquared:.3f} dropped={dropped_x}")

                    vars_std = ["const"] + list(Xz.columns)
                    for var in vars_std:
                        std_results.append({
                            "analysis_name": analysis_name,
                            "range_label": effective_label,
                            "range_title": effective_title,
                            "base_label": base_label,
                            "phase": phase_name if phase_name is not None else "",
                            "rolling_window_days": ROLLING_WINDOW_DAYS,
                            "target_day": target_label,
                            "window_label": window_label,
                            "window_number": wi,
                            "group": yname,
                            "variable": var,
                            "coef": get_param_value(model_std.params, var),
                            "pvalue": get_param_value(model_std.pvalues, var),
                            "R2": float(model_std.rsquared),
                            "n": int(model_std.nobs),
                            "mode": "standardized",
                            "y_mean": y_mean,
                            "y_std": y_std,
                            "dropped_predictors": ";".join(dropped_x),
                            "x_mean": get_param_value(x_means, var),
                            "x_std":  get_param_value(x_stds, var),
                        })

            print("\n=== SAVING OUTPUTS ===")
            save_bundle(out_range_root, std_results, std_timeseries, quality_rows)

            gif_path = os.path.join(out_range_root, GIF_NAME)
            save_group_gif(gif_frames, gif_path)

            print(f"[SAVE] GIF frames folder kept: {frames_dir}")

            family_master_rows.append({
                "analysis_name": analysis_name,
                "range_label": effective_label,
                "range_title": effective_title,
                "base_label": base_label,
                "phase": phase_name if phase_name is not None else "",
                "rolling_window_days": ROLLING_WINDOW_DAYS,
                "n_std_rows": len(std_results),
                "n_timeseries_blocks": len(std_timeseries),
                "n_quality_rows": len(quality_rows),
                "gif_created": os.path.exists(gif_path),
                "gif_frames_dir": frames_dir,
                "n_gif_frames": len(gif_frames),
            })

    summary_path = os.path.join(out_analysis_root, "regression_master_summary.csv")
    pd.DataFrame(family_master_rows).to_csv(summary_path, index=False)
    print("[SAVE]", summary_path)

    all_master_rows.extend(family_master_rows)

# Global summary

global_summary_path = os.path.join(OUT_ROOT_BASE, "regression_master_summary_ALL_FAMILIES.csv")
pd.DataFrame(all_master_rows).to_csv(global_summary_path, index=False)
print("[SAVE]", global_summary_path)

# Daily window summary

if window_summary_rows:
    window_summary_df = pd.DataFrame(window_summary_rows)

    window_summary_df = window_summary_df.sort_values(
        by=["analysis_name", "range_label", "window_number", "target_day"]
    ).reset_index(drop=True)

    print("\n" + "=" * 130)
    print("DAILY WINDOW ACQUISITION + MERGE SUMMARY")
    print("=" * 130)
    print(window_summary_df.to_string(index=False))

    window_summary_csv = os.path.join(
        OUT_ROOT_BASE,
        "daily_window_acquisition_summary_ALL_FAMILIES.csv"
    )
    window_summary_df.to_csv(window_summary_csv, index=False)
    print("[SAVE]", window_summary_csv)
else:
    print("\n[WARN] No daily window summary rows were created.")

print("\nDONE.")
print(f"Outputs stored in:\n{OUT_ROOT_BASE}")