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
from matplotlib.patches import Patch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

ANALYSIS_NAME = "ALL"   # "ALL", "Snow_NoSnow", "Transition", "Rain_NoRain", "Day_Night"

CORR_FOLDER = os.path.join(DATA_DIR, "corrected")
METEO_CSV      = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")

CLUSTER_ROOT_BASE = os.path.join(DATA_DIR, "corrected", "analysis", "temperature_curve_clusters")
OUT_DIR_BASE      = os.path.join(DATA_DIR, "corrected", "analysis", "temperature_curve_regression_k2")

H, W = 252, 336

ROLLING_WINDOW_DAYS = 5
MERGE_TOLERANCE_MIN = 10
MIN_POINTS_PER_WINDOW = 3

PRINT_EVERY_N_IMAGES = 50
PRINT_FIRST_N_SKIPS_PER_WINDOW = 5

MATCH_TOLERANCE_MIN_DAYNIGHT = 10
NIGHT_THRESHOLD = 10.0   # SWRup < 10 => night, else day

BASE_VARS = ["TA", "RH", "VWND", "LWRup", "SWRup", "DWND"]

GIF_NAME = "daily_cluster_masks.gif"
GIF_FRAMES_SUBDIR = "gif_frames"
GIF_FPS = 1.0

CURVE_PLOTS_SUBDIR = "daily_mean_curves"

GROUP0_COLOR = "#B3CDE3"
GROUP1_COLOR = "#FBB4AE"

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

if ANALYSIS_NAME != "ALL" and ANALYSIS_NAME not in ANALYSIS_CONFIGS:
    raise RuntimeError(f"Unknown ANALYSIS_NAME: {ANALYSIS_NAME}")

# Helpers

TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

def name_to_dt(path_or_name: str):
    m = TS_RE.search(os.path.basename(path_or_name))
    if not m:
        return None
    yy, MM, DD, hh, mm, ss = map(int, m.groups())
    return datetime(2000 + yy, MM, DD, hh, mm, ss)

def parse_date_flexible(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {s}")

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

def safe_unique_counts(a: np.ndarray):
    u, c = np.unique(a, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

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

def build_daily_targets(start_dt: datetime, end_dt: datetime):
    days = []
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= end_day:
        days.append(cur)
        cur += timedelta(days=1)
    return days

def get_window_bounds(target_day: datetime, window_days: int):
    day_start = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = day_start - timedelta(days=window_days - 1)
    window_end = day_start.replace(hour=23, minute=59, second=59, microsecond=0)
    window_label = f"{window_start.strftime('%Y-%m-%d')}_to_{day_start.strftime('%Y-%m-%d')}"
    target_label = day_start.strftime("%Y-%m-%d")
    target_label_display = day_start.strftime("%d.%m.%y")
    window_label_display = f"{window_start.strftime('%d.%m.%y')} - {day_start.strftime('%d.%m.%y')}"
    return window_start, window_end, window_label, target_label, target_label_display, window_label_display

def load_day_night_table_from_meteo(met: pd.DataFrame) -> pd.DataFrame:
    df = met[["Time", "SWRup"]].dropna(subset=["Time", "SWRup"]).copy()
    df = df.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df

def attach_day_night_flag(df_files: pd.DataFrame, df_ref: pd.DataFrame, tol_min: int) -> pd.DataFrame:
    out = pd.merge_asof(
        df_files.sort_values("Time"),
        df_ref.rename(columns={"Time": "ref_time"}).sort_values("ref_time"),
        left_on="Time",
        right_on="ref_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tol_min),
    )
    out = out.dropna(subset=["SWRup"]).copy()
    out["day_night"] = np.where(out["SWRup"] < NIGHT_THRESHOLD, "night", "day")
    return out

def find_cluster_label_files(cluster_folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(cluster_folder, "cluster_labels_*_k2.csv")))

def extract_target_day_label_from_cluster_csv(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.search(r"_target_(\d{4}-\d{2}-\d{2})_window_.*_n\d+_k2\.csv$", base)
    if m:
        return m.group(1)
    return None

def save_bundle(out_dir: str, results: list, timeseries_list: list, quality: list):
    os.makedirs(out_dir, exist_ok=True)

    res_df = pd.DataFrame(results)
    res_path = os.path.join(out_dir, "daily_window_cluster_regression_k2.csv")
    res_df.to_csv(res_path, index=False)
    print("[SAVE]", res_path, f"({len(res_df)} rows)")

    if timeseries_list:
        ts_df = pd.concat(timeseries_list, ignore_index=True)
        ts_path = os.path.join(out_dir, "daily_window_cluster_timeseries_k2.csv")
        ts_df.to_csv(ts_path, index=False)
        print("[SAVE]", ts_path, f"({len(ts_df)} rows)")
    else:
        print("[WARN] No timeseries to save into:", out_dir)

    q_df = pd.DataFrame(quality)
    q_path = os.path.join(out_dir, "daily_window_quality_report_k2.csv")
    q_df.to_csv(q_path, index=False)
    print("[SAVE]", q_path, f"({len(q_df)} rows)")

def fit_ols_safe(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")

    valid = pd.concat([y, Xc], axis=1).dropna()
    if len(valid) < 2:
        return None

    yv = valid.iloc[:, 0]
    Xv = valid.iloc[:, 1:]

    if Xv.shape[1] == 0:
        return None

    try:
        model = sm.OLS(yv, Xv, missing="drop").fit()
        return model
    except Exception as e:
        print(f"[OLS FAIL] {e}")
        return None

def build_cluster_code_array(labels: np.ndarray) -> np.ndarray:
    arr = np.full((H, W), np.nan, dtype=np.float32)
    arr[labels == 0] = 0
    arr[labels == 1] = 1
    return arr

def render_cluster_frame_png(code_arr: np.ndarray, target_label: str, silhouette_val: float, out_png: str,
                             target_label_display: str = None):
    cmap = ListedColormap([
        GROUP0_COLOR,
        GROUP1_COLOR,
    ])

    sil_txt = "n/a" if pd.isna(silhouette_val) else f"{silhouette_val:.3f}"
    t_disp = target_label_display if target_label_display else target_label

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(code_arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"Temperature Curve Cluster: {t_disp} (5-Day Window | Silhouette Score: {sil_txt})")
    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        Patch(facecolor=GROUP0_COLOR, edgecolor="black", label="Cluster 1"),
        Patch(facecolor=GROUP1_COLOR, edgecolor="black", label="Cluster 2"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_cluster_gif(frames_info: list, out_gif: str):
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

def load_silhouette_summary(cluster_analysis_root: str, split_day_night: bool) -> pd.DataFrame:
    mode = "split" if split_day_night else "all"
    summary_path = os.path.join(
        cluster_analysis_root,
        f"silhouette_summary_{mode}_rolling_{ROLLING_WINDOW_DAYS}day.csv"
    )
    if not os.path.exists(summary_path):
        print(f"[WARN] Silhouette summary not found: {summary_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(summary_path, sep=";")
    except Exception:
        try:
            df = pd.read_csv(summary_path)
        except Exception as e:
            print(f"[WARN] Could not read silhouette summary: {summary_path} -> {e}")
            return pd.DataFrame()

    return df

def get_silhouette_value(summary_df: pd.DataFrame, range_label: str, target_day: str, k: int = 2) -> float:
    if summary_df.empty:
        return np.nan

    need_cols = {"range_label", "target_day", "k", "silhouette"}
    if not need_cols.issubset(summary_df.columns):
        return np.nan

    sub = summary_df[
        (summary_df["range_label"].astype(str) == str(range_label)) &
        (summary_df["target_day"].astype(str) == str(target_day)) &
        (pd.to_numeric(summary_df["k"], errors="coerce") == k)
    ].copy()

    if sub.empty:
        return np.nan

    return float(pd.to_numeric(sub["silhouette"], errors="coerce").iloc[0])

def save_daily_mean_curve_plot(curve_df: pd.DataFrame, out_png: str, target_day: str, window_label: str, silhouette_val: float,
                               target_day_display: str = None, window_label_display: str = None):
    if curve_df.empty:
        return

    curve_df = curve_df.copy()
    curve_df["day"] = curve_df["Time"].dt.strftime("%Y-%m-%d")
    curve_df["hour"] = (
        curve_df["Time"].dt.hour +
        curve_df["Time"].dt.minute / 60.0 +
        curve_df["Time"].dt.second / 3600.0
    )

    unique_days = sorted(curve_df["day"].unique().tolist())
    n_days = len(unique_days)
    if n_days == 0:
        return

    sil_txt = "n/a" if pd.isna(silhouette_val) else f"{silhouette_val:.3f}"

    fig, axes = plt.subplots(
        n_days, 1,
        figsize=(10, max(2.2 * n_days, 3.2)),
        sharex=True,
        sharey=True
    )

    if n_days == 1:
        axes = [axes]

    y_min = np.nanmin([curve_df["Tc0"].min(), curve_df["Tc1"].min()])
    y_max = np.nanmax([curve_df["Tc0"].max(), curve_df["Tc1"].max()])
    if np.isfinite(y_min) and np.isfinite(y_max):
        if np.isclose(y_min, y_max):
            pad = 1.0
        else:
            pad = 0.05 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    for ax, day_str in zip(axes, unique_days):
        dsub = curve_df[curve_df["day"] == day_str].sort_values("Time").copy()

        ax.plot(
            dsub["hour"], dsub["Tc0"],
            color=GROUP0_COLOR, linewidth=2.0, marker="o", markersize=3,
            label="Cluster 1"
        )
        ax.plot(
            dsub["hour"], dsub["Tc1"],
            color=GROUP1_COLOR, linewidth=2.0, marker="o", markersize=3,
            label="Cluster 2"
        )

        ax.set_title(day_str, fontsize=10, pad=4)
        ax.set_ylabel("Temperature")
        ax.grid(True, linestyle="--", alpha=0.5)

        if np.isfinite(y_min) and np.isfinite(y_max):
            ax.set_ylim(y_min, y_max)

    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xlim(0, 24)

    handles = [
        Patch(facecolor=GROUP0_COLOR, edgecolor="black", label="Cluster 1"),
        Patch(facecolor=GROUP1_COLOR, edgecolor="black", label="Cluster 2"),
    ]

    t_disp = target_day_display if target_day_display else target_day
    w_disp = window_label_display if window_label_display else window_label
    fig.suptitle(
        f"Target day: {t_disp}\nGrouping window: {w_disp} | Silhouette: {sil_txt}",
        fontsize=12, y=0.995
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Load meteo

print("=== LOADING METEO ===")
met, met_sep = robust_read_meteo_csv(METEO_CSV)
print(f"[METEO] Read {len(met)} rows with sep='{met_sep}'")
print(f"[METEO] Columns: {list(met.columns)}")

met["Time"] = pd.to_datetime(met["Time"], errors="coerce")
bad_time = int(met["Time"].isna().sum())
if bad_time > 0:
    print(f"[METEO] Dropping {bad_time} rows with unparseable Time")
met = met.dropna(subset=["Time"]).copy()

missing = [c for c in BASE_VARS if c not in met.columns]
if missing:
    raise RuntimeError(f"[METEO] Missing required columns: {missing}")

for c in BASE_VARS:
    met[c] = pd.to_numeric(met[c], errors="coerce")

nan_cols = {c: int(met[c].isna().sum()) for c in BASE_VARS}
print(f"[METEO] NaN counts: {nan_cols}")

rad = np.deg2rad(met["DWND"])
met["DWND_sin"] = np.sin(rad)
met["DWND_cos"] = np.cos(rad)

METEO_VARS = ["TA", "RH", "VWND", "LWRup", "SWRup", "DWND_sin", "DWND_cos"]
met = met.sort_values("Time").reset_index(drop=True)

print("[METEO] Time range:", met["Time"].min(), "->", met["Time"].max())
print()

# Load TIR file list

print("=== INDEXING TIR GRIDS ===")
tir_files = sorted(glob.glob(os.path.join(CORR_FOLDER, "*_corr.csv")))
print(f"[TIR] Found {len(tir_files)} *_corr.csv files in {CORR_FOLDER}")

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

print()

# Main analysis

all_master_rows = []
analysis_names_to_run = list(ANALYSIS_CONFIGS.keys()) if ANALYSIS_NAME == "ALL" else [ANALYSIS_NAME]

for current_analysis_name in analysis_names_to_run:
    print("\n" + "#" * 90)
    print(f"RUNNING ANALYSIS FAMILY: {current_analysis_name}")
    print("#" * 90)

    cfg = ANALYSIS_CONFIGS[current_analysis_name]
    analysis_ranges = cfg["ranges"]
    split_day_night = cfg["split_day_night"]

    cluster_analysis_root = os.path.join(CLUSTER_ROOT_BASE, current_analysis_name)
    out_analysis_root = os.path.join(OUT_DIR_BASE, current_analysis_name)
    os.makedirs(out_analysis_root, exist_ok=True)

    silhouette_summary_df = load_silhouette_summary(cluster_analysis_root, split_day_night)

    if split_day_night:
        df_daynight = load_day_night_table_from_meteo(met)
        print(f"[DAY/NIGHT] Loaded {len(df_daynight)} reference rows from meteo Time + SWRup.")
    else:
        df_daynight = None

    master_rows = []

    for ri, range_info in enumerate(analysis_ranges, start=1):
        base_label = range_info["label"]
        base_title = range_info["title"]

        start_dt = parse_date_flexible(range_info["start"]).replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt   = parse_date_flexible(range_info["end"]).replace(hour=23, minute=59, second=59, microsecond=0)

        print(f"\n{'='*78}")
        print(f"RANGE {ri}/{len(analysis_ranges)}: {base_title} [{start_dt.date()} -> {end_dt.date()}]")
        print(f"{'='*78}")

        target_days = build_daily_targets(start_dt, end_dt)
        earliest_window_start = start_dt - timedelta(days=ROLLING_WINDOW_DAYS - 1)

        df_range = tir_df_all[(tir_df_all["Time"] >= earliest_window_start) & (tir_df_all["Time"] <= end_dt)].copy()

        if len(df_range) == 0:
            print("[RANGE] No TIR files in date range, skipping.")
            continue

        if split_day_night:
            df_range = attach_day_night_flag(df_range, df_daynight, MATCH_TOLERANCE_MIN_DAYNIGHT)
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

            cluster_root = os.path.join(cluster_analysis_root, effective_label, "clusters_k2")
            label_files = find_cluster_label_files(cluster_root)

            if len(label_files) == 0:
                print(f"[PHASE] No cluster label grids found in: {cluster_root}")
                continue

            target_to_label = {}
            for lp in label_files:
                tlabel = extract_target_day_label_from_cluster_csv(lp)
                if tlabel is not None:
                    target_to_label[tlabel] = lp

            print(f"[PHASE] {effective_title}: {len(df_phase)} files")
            print(f"[CLUSTERS] Found {len(target_to_label)} usable cluster label grids")

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

            curve_dir = os.path.join(out_range_root, CURVE_PLOTS_SUBDIR)
            os.makedirs(curve_dir, exist_ok=True)

            for window_number, target_day in enumerate(target_days, start=1):
                window_start, window_end, window_label, target_label, target_label_display, window_label_display = get_window_bounds(target_day, ROLLING_WINDOW_DAYS)
                group = df_phase[
                    (df_phase["Time"] >= window_start) &
                    (df_phase["Time"] <= window_end)
                ].sort_values("Time").reset_index(drop=True)

                lab_path = target_to_label.get(target_label)

                print(f"\n=== TARGET DAY {target_label} | {effective_title} ===")
                print(f"[WINDOW] {window_label}")
                print(f"[WINDOW] TIR images: {len(group)}")

                if lab_path is None:
                    print(f"[WINDOW] No matching cluster_labels file for target day: {target_label}")
                    continue

                print(f"[WINDOW] Using label grid: {os.path.basename(lab_path)}")

                if len(group) < 2:
                    print("[WINDOW] Skipping: <2 images")
                    continue

                try:
                    labels = np.loadtxt(lab_path, delimiter=";").astype(int)
                except Exception as e:
                    print(f"[LABEL READ FAIL] {os.path.basename(lab_path)}: {e}")
                    continue

                if labels.shape != (H, W):
                    try:
                        labels = labels.reshape(H, W)
                        print(f"[LABELS] Reshaped labels to {(H, W)}")
                    except Exception:
                        print(f"[LABELS] Shape mismatch: got {labels.shape}, cannot reshape to {(H, W)}")
                        continue

                counts = safe_unique_counts(labels)
                mask0 = labels == 0
                mask1 = labels == 1
                print(f"[LABELS] Unique counts: {counts}")
                print(f"[LABELS] mask0 pixels: {int(mask0.sum())}, mask1 pixels: {int(mask1.sum())}")

                if mask0.sum() == 0 or mask1.sum() == 0:
                    print("[WINDOW] Skipping: empty cluster mask0 or mask1")
                    continue

                silhouette_val = get_silhouette_value(
                    silhouette_summary_df,
                    effective_label,
                    target_label,
                    k=2
                )
                sil_txt = "n/a" if pd.isna(silhouette_val) else f"{silhouette_val:.3f}"
                print(f"[SILHOUETTE] {effective_label} | {target_label}: {sil_txt}")

                try:
                    code_arr = build_cluster_code_array(labels)
                    frame_png = os.path.join(frames_dir, f"{window_number:04d}_{target_label}.png")
                    render_cluster_frame_png(code_arr, target_label, silhouette_val, frame_png,
                                             target_label_display)
                    gif_frames.append({
                        "target_day": target_label,
                        "png": frame_png,
                    })
                except Exception as e:
                    print(f"[GIF FRAME FAIL] {effective_label} | {target_label}: {e}")

                rows = []
                curve_rows = []
                skip_read = 0
                skip_empty = 0
                skip_nanheavy = 0
                shown_skips = 0

                for ii, r in group.iterrows():
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

                    vals0 = grid[mask0]
                    vals1 = grid[mask1]
                    n0 = int(np.isfinite(vals0).sum())
                    n1 = int(np.isfinite(vals1).sum())

                    if n0 == 0 or n1 == 0:
                        skip_empty += 1
                        if shown_skips < PRINT_FIRST_N_SKIPS_PER_WINDOW:
                            print(f"[SKIP EMPTY] {os.path.basename(p)} finite0={n0} finite1={n1} nan_frac={nan_frac:.2f}")
                            shown_skips += 1
                        continue

                    Tc0 = float(np.nanmean(vals0))
                    Tc1 = float(np.nanmean(vals1))

                    rows.append({
                        "Time": dt,
                        "target_day": target_label,
                        "window_label": window_label,
                        "tir_path": p,
                        "Tc0": Tc0,
                        "Tc1": Tc1,
                        "nan_frac": nan_frac,
                    })

                    curve_rows.append({
                        "Time": dt,
                        "Tc0": Tc0,
                        "Tc1": Tc1,
                    })

                    if (ii + 1) % PRINT_EVERY_N_IMAGES == 0:
                        print(f"[PROGRESS] {ii+1}/{len(group)} images processed, kept={len(rows)}")

                print(f"[WINDOW] Kept images: {len(rows)} / {len(group)}")
                print(f"[WINDOW] Skips: read_fail={skip_read}, nan_heavy={skip_nanheavy}, empty_cluster_vals={skip_empty}")

                quality_rows.append({
                    "analysis_name": current_analysis_name,
                    "range_label": effective_label,
                    "range_title": effective_title,
                    "base_label": base_label,
                    "phase": phase_name if phase_name is not None else "",
                    "rolling_window_days": ROLLING_WINDOW_DAYS,
                    "target_day": target_label,
                    "window_label": window_label,
                    "window_number": window_number,
                    "n_window_imgs": int(len(group)),
                    "n_kept": int(len(rows)),
                    "skip_read": int(skip_read),
                    "skip_nanheavy": int(skip_nanheavy),
                    "skip_empty": int(skip_empty),
                    "mask0_pixels": int(mask0.sum()),
                    "mask1_pixels": int(mask1.sum()),
                    "label_counts": str(counts),
                    "silhouette": silhouette_val,
                })

                if curve_rows:
                    curve_df = pd.DataFrame(curve_rows).sort_values("Time").reset_index(drop=True)
                    curve_png = os.path.join(curve_dir, f"mean_daily_curves_target_{target_label}.png")
                    try:
                        save_daily_mean_curve_plot(
                            curve_df=curve_df,
                            out_png=curve_png,
                            target_day=target_label,
                            window_label=window_label,
                            silhouette_val=silhouette_val,
                            target_day_display=target_label_display,
                            window_label_display=window_label_display,
                        )
                        print(f"[SAVE] {curve_png}")
                    except Exception as e:
                        print(f"[CURVE PLOT FAIL] {effective_label} | {target_label}: {e}")

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

                if after < MIN_POINTS_PER_WINDOW:
                    print(f"[WINDOW] Skipping regression after merge: {after} points (<{MIN_POINTS_PER_WINDOW})")
                    continue

                keep_cols = ["target_day", "window_label", "Time", "Tc0", "Tc1", "nan_frac"] + METEO_VARS
                std_timeseries.append(merged[keep_cols])

                for cname in ["Tc0", "Tc1"]:
                    y_raw = merged[cname].astype(float)
                    X_raw = merged[METEO_VARS].astype(float)

                    if float(np.nanstd(y_raw)) == 0.0:
                        print(f"[REG] {target_label} {cname}: zero variance in y, skipping standardized")
                        continue

                    Xz, x_means, x_stds, dropped_x = zscore_df(X_raw)
                    y_mean = float(y_raw.mean())
                    y_std  = float(y_raw.std(ddof=0))

                    if y_std <= 1e-12:
                        print(f"[REG STD] {target_label} {cname}: y_std ~ 0, skipping standardized")
                        continue

                    yz = (y_raw - y_mean) / y_std

                    if Xz.shape[1] == 0:
                        print(f"[REG STD] {target_label} {cname}: all predictors constant (dropped={dropped_x}), skipping standardized")
                        continue

                    model_std = fit_ols_safe(yz, Xz)
                    if model_std is None:
                        print(f"[REG STD] {target_label} {cname}: model fit failed, skipping")
                        continue

                    print(f"[REG STD] {target_label} {cname}: n={int(model_std.nobs)} R2={model_std.rsquared:.3f} dropped={dropped_x}")

                    vars_std = ["const"] + list(Xz.columns)
                    for var in vars_std:
                        coef_val = model_std.params.get(var, np.nan)
                        pval_val = model_std.pvalues.get(var, np.nan)

                        std_results.append({
                            "analysis_name": current_analysis_name,
                            "range_label": effective_label,
                            "range_title": effective_title,
                            "base_label": base_label,
                            "phase": phase_name if phase_name is not None else "",
                            "rolling_window_days": ROLLING_WINDOW_DAYS,
                            "target_day": target_label,
                            "window_label": window_label,
                            "window_number": window_number,
                            "cluster": cname,
                            "variable": var,
                            "coef": float(coef_val) if pd.notna(coef_val) else np.nan,
                            "pvalue": float(pval_val) if pd.notna(pval_val) else np.nan,
                            "R2": float(model_std.rsquared),
                            "n": int(model_std.nobs),
                            "mode": "standardized",
                            "y_mean": y_mean,
                            "y_std": y_std,
                            "dropped_predictors": ";".join(dropped_x),
                            "x_mean": float(x_means[var]) if (var in x_means.index) else np.nan,
                            "x_std":  float(x_stds[var])  if (var in x_stds.index)  else np.nan,
                            "silhouette": silhouette_val,
                        })

            print("\n=== SAVING OUTPUTS ===")
            save_bundle(out_range_root, std_results, std_timeseries, quality_rows)

            gif_path = os.path.join(out_range_root, GIF_NAME)
            save_cluster_gif(gif_frames, gif_path)

            print(f"[SAVE] GIF frames folder kept: {frames_dir}")
            print(f"[SAVE] Curve plots folder kept: {curve_dir}")

            master_rows.append({
                "analysis_name": current_analysis_name,
                "range_label": effective_label,
                "range_title": effective_title,
                "base_label": base_label,
                "phase": phase_name if phase_name is not None else "",
                "rolling_window_days": ROLLING_WINDOW_DAYS,
                "cluster_root": cluster_root,
                "n_std_rows": len(std_results),
                "n_timeseries_blocks": len(std_timeseries),
                "n_quality_rows": len(quality_rows),
                "gif_created": os.path.exists(gif_path),
                "gif_frames_dir": frames_dir,
                "n_gif_frames": len(gif_frames),
                "curve_plots_dir": curve_dir,
            })

    summary_path = os.path.join(out_analysis_root, "regression_master_summary.csv")
    pd.DataFrame(master_rows).to_csv(summary_path, index=False)
    print("[SAVE]", summary_path)

    all_master_rows.extend(master_rows)

# Global summary

summary_all_path = os.path.join(OUT_DIR_BASE, "regression_master_summary_all_families.csv")
pd.DataFrame(all_master_rows).to_csv(summary_all_path, index=False)
print("[SAVE]", summary_all_path)

print("\nDONE.")
print(f"Outputs stored in:\n{OUT_DIR_BASE}")