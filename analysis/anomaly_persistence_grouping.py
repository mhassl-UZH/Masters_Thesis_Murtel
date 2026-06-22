import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
CORR_FOLDER = os.path.join(DATA_DIR, "corrected")

OUTPUT_rain_norain = os.path.join(DATA_DIR, "analysis", "anomaly_persistence", "Rain_NoRain")
OUTPUT_snow_nosnow = os.path.join(DATA_DIR, "analysis", "anomaly_persistence", "Snow_NoSnow")
OUTPUT_transition  = os.path.join(DATA_DIR, "analysis", "anomaly_persistence", "Transition")
OUTPUT_day_night   = os.path.join(DATA_DIR, "analysis", "anomaly_persistence", "Day_Night")

MASK_PATH   = os.path.join(DATA_DIR, "TIR_mask.png")
METEO_PATH  = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")

MATCH_TOLERANCE_MIN = 10
NIGHT_THRESHOLD     = 10   # SWRup < 10 => night, else day
ROLLING_WINDOW_DAYS = 5

# Analysis ranges
ANALYSIS_RANGES_snow_nosnow = [
    {"label": "2021_NoSnow", "title": "NoSnow 2021", "start": "05.08.21", "end": "23.08.21"},
    {"label": "2022_NoSnow", "title": "NoSnow 2022", "start": "07.06.22", "end": "19.07.22"},
    {"label": "2022_Snow",   "title": "Snow 2022",   "start": "26.02.22", "end": "01.04.22"},
]

ANALYSIS_RANGES_day_night = [
    {"label": "2022_NoSnow", "title": "NoSnow 2022", "start": "07.06.22", "end": "19.07.22"},
    {"label": "2022_Snow",   "title": "Snow 2022",   "start": "26.02.22", "end": "14.03.22"},
]


ANALYSIS_RANGES_transition = [
    {"label": "2021_Spring", "title": "Spring 2021", "start": "14.06.21", "end": "25.07.21"},
    {"label": "2021_Autumn", "title": "Autumn 2021", "start": "13.09.21", "end": "14.11.21"},
    {"label": "2022_Spring", "title": "Spring 2022", "start": "09.05.22", "end": "26.06.22"},
    {"label": "2022_Autumn", "title": "Autumn 2022", "start": "31.10.22", "end": "18.12.22"},
    {"label": "2023_Spring", "title": "Spring 2023", "start": "15.05.23", "end": "02.07.23"},
]

ANALYSIS_RANGES_rain_norain = [
    {"label": "2022_NoSnow_NoRain", "title": "NoSnow NoRain 2022", "start": "05.07.22", "end": "22.07.22"},
    {"label": "2022_NoSnow_Rain",   "title": "NoSnow Rain 2022",   "start": "23.07.22", "end": "08.08.22"},
    {"label": "2022_Snow_NoRain",   "title": "Snow NoRain 2022",   "start": "24.02.22", "end": "28.03.22"},
    {"label": "2022_Snow_Rain",     "title": "Snow Rain 2022",     "start": "29.01.22", "end": "23.02.22"},
]

# Output toggles
SAVE_Z_FRAMES = True
SAVE_GLOBAL_MAPS = True
SAVE_PERIOD_MAPS = True
SAVE_PERIOD_CSVS = True

# Load TIR mask
mask_img = plt.imread(MASK_PATH)
if mask_img.ndim == 3:
    mask_img = mask_img[..., 0]

tir_mask = mask_img > 0.5
print("Loaded TIR mask:", tir_mask.sum(), "valid pixels.")

# Helpers
TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

def sanitize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def parse_date_flexible(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {s}")

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
    return window_start, window_end, window_label, target_label

def parse_time(fname: str):
    base = os.path.basename(fname).split("_")[0]
    m = TS_RE.match(base)
    if m is None:
        return None
    yy, MM, DD, hh, mm, ss = map(int, m.groups())
    return datetime(2000 + yy, MM, DD, hh, mm, ss)

def read_grid_csv(path: str) -> np.ndarray:
    return pd.read_csv(path, sep=";", header=None).iloc[1:, :].to_numpy(dtype=np.float32)

def z_normalize(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        std = 1e-6
    return (arr - mean) / std

def load_day_night_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "Time" not in df.columns or "SWRup" not in df.columns:
        raise ValueError("METEO_PATH must have columns 'Time' and 'SWRup'.")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["SWRup"] = pd.to_numeric(df["SWRup"], errors="coerce")
    df = df.dropna(subset=["Time", "SWRup"]).copy()

    df = df[["Time", "SWRup"]].drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df

def attach_day_night_flag(df_files: pd.DataFrame, df_ref: pd.DataFrame, tol_min: int, threshold: float) -> pd.DataFrame:
    out = pd.merge_asof(
        df_files.sort_values("datetime"),
        df_ref.rename(columns={"Time": "ref_time"}).sort_values("ref_time"),
        left_on="datetime",
        right_on="ref_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tol_min),
    )
    out = out.dropna(subset=["SWRup"]).copy()
    out["is_night"] = out["SWRup"] < threshold
    out["day_night"] = np.where(out["is_night"], "night", "day")
    return out

def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Load day/night reference
df_daynight = load_day_night_table(METEO_PATH)
print(f"Loaded {len(df_daynight)} day/night reference rows from Time + SWRup.")

# Load file list
all_files = []
for f in os.listdir(CORR_FOLDER):
    if f.endswith("_corr.csv") and f.startswith("m"):
        dt = parse_time(f)
        if dt is not None:
            all_files.append((dt, f))

all_files.sort()
print(f"Loaded {len(all_files)} files total.")

if len(all_files) == 0:
    raise RuntimeError("No *_corr.csv files found.")

# Image size
sample = read_grid_csv(os.path.join(CORR_FOLDER, all_files[0][1]))
H, W = sample.shape

# Analysis families
ANALYSIS_FAMILIES = [
    {
        "name": "Rain_NoRain",
        "out_root": OUTPUT_rain_norain,
        "day_night_mode": "all",
        "ranges": ANALYSIS_RANGES_rain_norain,
    },
    {
        "name": "Snow_NoSnow",
        "out_root": OUTPUT_snow_nosnow,
        "day_night_mode": "all",
        "ranges": ANALYSIS_RANGES_snow_nosnow,
    },
    {
        "name": "Transition",
        "out_root": OUTPUT_transition,
        "day_night_mode": "all",
        "ranges": ANALYSIS_RANGES_transition,
    },
    {
        "name": "Day_Night",
        "out_root": OUTPUT_day_night,
        "day_night_mode": "split",
        "ranges": ANALYSIS_RANGES_day_night,
    },
]

# Main analysis
for family in ANALYSIS_FAMILIES:
    ANALYSIS_NAME = family["name"]
    OUT_ROOT = family["out_root"]
    DAY_NIGHT_MODE = family["day_night_mode"]
    ANALYSIS_RANGES = family["ranges"]

    print("\n" + "#" * 90)
    print(f"START ANALYSIS FAMILY: {ANALYSIS_NAME}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"DAY_NIGHT_MODE: {DAY_NIGHT_MODE}")
    print(f"ROLLING_WINDOW_DAYS: {ROLLING_WINDOW_DAYS}")
    print("#" * 90)

    os.makedirs(OUT_ROOT, exist_ok=True)

    for range_idx, range_info in enumerate(ANALYSIS_RANGES, start=1):
        base_range_label = sanitize_text(range_info["label"])
        base_range_title = str(range_info.get("title", range_info["label"]))

        range_start = parse_date_flexible(range_info["start"]).replace(hour=0, minute=0, second=0, microsecond=0)
        range_end   = parse_date_flexible(range_info["end"]).replace(hour=23, minute=59, second=59, microsecond=0)

        print("\n" + "=" * 70)
        print(f"BASE RANGE {range_idx}: {base_range_title} [{range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')}]")
        print("=" * 70)

        selected_files_range_only = [(dt, f) for dt, f in all_files if range_start <= dt <= range_end]
        print(f"Images in selected base range itself: {len(selected_files_range_only)}")

        target_days = build_daily_targets(range_start, range_end)
        print(f"Number of target days: {len(target_days)}")

        if len(target_days) == 0:
            print("[WARN] No target days in this range. Skipping.")
            continue

        earliest_window_start = range_start - timedelta(days=ROLLING_WINDOW_DAYS - 1)
        selected_files_for_windows = [(dt, f) for dt, f in all_files if earliest_window_start <= dt <= range_end]

        if len(selected_files_for_windows) == 0:
            print("[WARN] No files found for this range including pre-range rolling days. Skipping.")
            continue

        df_range = pd.DataFrame(selected_files_for_windows, columns=["datetime", "fname"])

        if DAY_NIGHT_MODE == "split":
            df_range = attach_day_night_flag(df_range, df_daynight, MATCH_TOLERANCE_MIN, NIGHT_THRESHOLD)

            if df_range.empty:
                print("[WARN] No files could be matched to Time/SWRup. Skipping base range.")
                continue

            phase_infos = [
                ("day", "day", "Day"),
                ("night", "night", "Night"),
            ]
        else:
            df_range["day_night"] = "all"
            phase_infos = [
                ("all", "", ""),
            ]

        for phase_value, phase_suffix, phase_title in phase_infos:
            df_phase = df_range[df_range["day_night"] == phase_value].copy()

            if DAY_NIGHT_MODE == "split":
                range_label = f"{base_range_label}_{phase_suffix}"
                range_title = f"{base_range_title} {phase_title}"
            else:
                range_label = base_range_label
                range_title = base_range_title

            print(f"\n--- {range_title}: {len(df_phase)} matched images for rolling windows ---")

            if df_phase.empty:
                print(f"[WARN] No images for phase '{phase_value}' in this range. Skipping.")
                continue

            range_out_root = os.path.join(OUT_ROOT, range_label)
            os.makedirs(range_out_root, exist_ok=True)

            global_count_above = np.zeros((H, W), dtype=np.uint32)
            global_count_below = np.zeros((H, W), dtype=np.uint32)
            global_n_images = 0

            global_frames_dir = os.path.join(range_out_root, "z_frames")
            if SAVE_Z_FRAMES:
                os.makedirs(global_frames_dir, exist_ok=True)

            for target_day in target_days:
                window_start, window_end, window_label, target_label = get_window_bounds(target_day, ROLLING_WINDOW_DAYS)

                group = df_phase[(df_phase["datetime"] >= window_start) & (df_phase["datetime"] <= window_end)].copy()

                print(f"\n=== Processing {range_title} | target_day={target_label} | window={window_label} ({len(group)} images) ===")

                period_count_above = np.zeros((H, W), dtype=np.uint32)
                period_count_below = np.zeros((H, W), dtype=np.uint32)

                valid_items = []
                for dt, fname in zip(group["datetime"].tolist(), group["fname"].tolist()):
                    csv_path = os.path.join(CORR_FOLDER, fname)
                    try:
                        arr = read_grid_csv(csv_path)
                    except Exception as e:
                        print(f"[SKIP] {fname}: {e}")
                        continue

                    if arr.shape != (H, W):
                        print(f"[SKIP] {fname}: shape {arr.shape} != {(H, W)}")
                        continue

                    valid_items.append((dt, fname, arr))

                n_images_valid = len(valid_items)

                if n_images_valid == 0:
                    print(f"[WARN] target_day={target_label}: no valid images in rolling window. Skipping day.")
                    continue

                for dt, fname, arr in valid_items:
                    z = z_normalize(arr)

                    z_gt = ((z > 0) & tir_mask).astype(np.uint32)
                    z_lt = ((z < 0) & tir_mask).astype(np.uint32)

                    period_count_above += z_gt
                    period_count_below += z_lt

                    if range_start <= dt <= range_end:
                        global_count_above += z_gt
                        global_count_below += z_lt
                        global_n_images += 1

                    if SAVE_Z_FRAMES:
                        z_masked = np.where(tir_mask, z, np.nan)
                        timestamp_str = dt.strftime("%Y-%m-%d_%H-%M-%S")

                        out_frame = os.path.join(
                            global_frames_dir,
                            f"z_{target_label}_{timestamp_str}.png"
                        )

                        ensure_parent_dir(out_frame)

                        fig, ax = plt.subplots(figsize=(6, 4))
                        im = ax.imshow(z_masked, cmap="coolwarm", vmin=-3, vmax=3)
                        plt.colorbar(im, ax=ax, label="z-score")
                        ax.set_title(
                            f"{range_title} | target {target_label} | n={n_images_valid}\n"
                            f"window {window_label}\n"
                            f"{dt.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        ax.axis("off")
                        plt.tight_layout()
                        plt.savefig(out_frame, dpi=120)
                        plt.close(fig)

                period_count_above_masked = np.where(tir_mask, period_count_above.astype(np.float32), np.nan)
                period_count_below_masked = np.where(tir_mask, period_count_below.astype(np.float32), np.nan)

                base_stub = f"{range_label}_target_{target_label}_window_{window_label}_n{n_images_valid}"

                if SAVE_PERIOD_CSVS:
                    out_count_above_csv = os.path.join(range_out_root, f"count_above_0_{base_stub}.csv")

                    ensure_parent_dir(out_count_above_csv)
                    pd.DataFrame(period_count_above_masked).to_csv(
                        out_count_above_csv, sep=";", index=False, header=False, na_rep=""
                    )

                    print("Saved:", out_count_above_csv)

                if SAVE_PERIOD_MAPS:
                    out_count_above_png = os.path.join(range_out_root, f"count_above_0_{base_stub}.png")
                    out_count_below_png = os.path.join(range_out_root, f"count_below_0_{base_stub}.png")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(period_count_above_masked, cmap="inferno")
                    plt.colorbar(im, ax=ax, label="Frames with z > 0")
                    ax.set_title(
                        f"{range_title} | target {target_label} | n={n_images_valid}\n"
                        f"Rolling 5-day window {window_label}\n"
                        f"Count of frames where pixel was above image mean (z>0)"
                    )
                    plt.tight_layout()
                    ensure_parent_dir(out_count_above_png)
                    plt.savefig(out_count_above_png, dpi=300)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(period_count_below_masked, cmap="inferno")
                    plt.colorbar(im, ax=ax, label="Frames with z < 0")
                    ax.set_title(
                        f"{range_title} | target {target_label} | n={n_images_valid}\n"
                        f"Rolling 5-day window {window_label}\n"
                        f"Count of frames where pixel was below image mean (z<0)"
                    )
                    plt.tight_layout()
                    ensure_parent_dir(out_count_below_png)
                    plt.savefig(out_count_below_png, dpi=300)
                    plt.close(fig)

                    print("Saved:", out_count_above_png)
                    print("Saved:", out_count_below_png)

            global_count_above_masked = np.where(tir_mask, global_count_above.astype(np.float32), np.nan)
            global_count_below_masked = np.where(tir_mask, global_count_below.astype(np.float32), np.nan)

            n_images_range = int(global_n_images)

            if SAVE_GLOBAL_MAPS:
                out_count_above = os.path.join(range_out_root, f"count_above_0_{range_label}_ALL_n{n_images_range}.png")
                out_count_below = os.path.join(range_out_root, f"count_below_0_{range_label}_ALL_n{n_images_range}.png")

                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(global_count_above_masked, cmap="inferno")
                plt.colorbar(im, ax=ax, label="Frames with z > 0")
                ax.set_title(f"{range_title} | ALL nominal-range images | n={n_images_range}\nCount of frames where pixel was above image mean (z>0)")
                plt.tight_layout()
                ensure_parent_dir(out_count_above)
                plt.savefig(out_count_above, dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(global_count_below_masked, cmap="inferno")
                plt.colorbar(im, ax=ax, label="Frames with z < 0")
                ax.set_title(f"{range_title} | ALL nominal-range images | n={n_images_range}\nCount of frames where pixel was below image mean (z<0)")
                plt.tight_layout()
                ensure_parent_dir(out_count_below)
                plt.savefig(out_count_below, dpi=300)
                plt.close(fig)

                print("Saved:", out_count_above)
                print("Saved:", out_count_below)

    print(f"\nDONE ANALYSIS FAMILY: {ANALYSIS_NAME}")
    print(f"Outputs stored in:\n{OUT_ROOT}")

print("\nAll analysis families finished.")