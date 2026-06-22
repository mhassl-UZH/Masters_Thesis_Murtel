import os
import glob
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
CORR_FOLDER = os.path.join(DATA_DIR, "corrected")

OUTPUT_rain_norain = os.path.join(DATA_DIR, "analysis", "temperature_curve_clusters", "Rain_NoRain")
OUTPUT_snow_nosnow = os.path.join(DATA_DIR, "analysis", "temperature_curve_clusters", "Snow_NoSnow")
OUTPUT_transition  = os.path.join(DATA_DIR, "analysis", "temperature_curve_clusters", "Transition")
OUTPUT_day_night   = os.path.join(DATA_DIR, "analysis", "temperature_curve_clusters", "Day_Night")

MASK_PATH  = os.path.join(DATA_DIR, "glacier_mask.png")

METEO_PATH = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")
MATCH_TOLERANCE_MIN = 10
NIGHT_THRESHOLD = 10   # SWRup < 10 => night, else day

ROLLING_WINDOW_DAYS = 5

W, H = 336, 252

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

K_RANGE = range(2, 3)
CLUSTER_MODE = "TIR"   # "glacier", "TIR", or other
BAD_VALUE_MODE = "skip"

CUSTOM_COLORS = [
    "#013A63",
    "#468FAF",
    "#90BE6D",
    "#2D6A4F"
]

# Load masks

mask_img = plt.imread(MASK_PATH)
if mask_img.ndim == 3:
    mask_img = mask_img[..., 0]
glacier_mask = mask_img > 0.5

tir_mask_path = MASK_PATH.replace("glacier_mask.png", "TIR_mask.png")
tir_img = plt.imread(tir_mask_path)
if tir_img.ndim == 3:
    tir_img = tir_img[..., 0]
tir_mask = tir_img > 0.5

print(f"Loaded glacier mask: {glacier_mask.sum()} valid pixels.")
print(f"Loaded TIR mask: {tir_mask.sum()} valid pixels.")

# Helpers

TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

def name_to_dt(name: str):
    m = TS_RE.search(name)
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

def sanitize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

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

def read_grid_csv(path: str, H: int, W: int, sep: str = ";") -> np.ndarray | None:
    try:
        df0 = pd.read_csv(path, sep=sep, header=None, skiprows=1, engine="python")
    except Exception as e:
        print(f"[READ FAIL] {os.path.basename(path)}: {e}")
        return None

    arr = df0.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if arr.shape != (H, W):
        print(f"[SHAPE FAIL] {os.path.basename(path)}: got {arr.shape}, expected {(H, W)}")
        return None

    if np.isnan(arr).any():
        r, c = np.argwhere(np.isnan(arr))[0]
        print(f"[FIRST NaN] {os.path.basename(path)} at grid row={r}, col={c}")

    return arr

def fill_nans_in_grid(arr: np.ndarray) -> np.ndarray | None:
    if not np.isnan(arr).any():
        return arr
    med = np.nanmedian(arr)
    if np.isnan(med):
        return None
    out = arr.copy()
    out[np.isnan(out)] = med
    return out

def load_day_night_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "Time" not in df.columns or "SWRup" not in df.columns:
        raise ValueError("METEO_PATH must contain columns 'Time' and 'SWRup'.")
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

# Load file list

files = sorted(glob.glob(os.path.join(CORR_FOLDER, "*_corr.csv")))

records = []
for f in files:
    dt = name_to_dt(f)
    if dt is None:
        continue
    records.append((f, dt))

df_all = pd.DataFrame(records, columns=["path", "datetime"])
print(f"Loaded {len(df_all)} images total.")

df_daynight = load_day_night_table(METEO_PATH)
print(f"Loaded {len(df_daynight)} day/night reference rows.")

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

ALL_RESULTS_ROOT = os.path.join(DATA_DIR, "corrected", "analysis", "temperature_curve_clusters", "Rain_NoRain")
all_results = []

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
    results = []

    for range_idx, range_info in enumerate(ANALYSIS_RANGES, start=1):
        base_range_label = sanitize_text(range_info["label"])
        base_range_title = str(range_info.get("title", range_info["label"]))

        range_start = parse_date_flexible(range_info["start"]).replace(hour=0, minute=0, second=0, microsecond=0)
        range_end   = parse_date_flexible(range_info["end"]).replace(hour=23, minute=59, second=59, microsecond=0)

        print("\n" + "=" * 70)
        print(f"BASE RANGE {range_idx}: {base_range_title} [{range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')}]")
        print("=" * 70)

        target_days = build_daily_targets(range_start, range_end)

        earliest_window_start = range_start - timedelta(days=ROLLING_WINDOW_DAYS - 1)
        df_range = df_all[(df_all["datetime"] >= earliest_window_start) & (df_all["datetime"] <= range_end)].copy()

        if df_range.empty:
            print("[WARN] No images in this base range including pre-window days. Skipping.")
            continue

        if DAY_NIGHT_MODE == "split":
            df_range = attach_day_night_flag(df_range, df_daynight, MATCH_TOLERANCE_MIN, NIGHT_THRESHOLD)

            print(f"Images in selected base range + pre-window days: {len(df_range)}")
            print(f"Number of target days: {len(target_days)}")

            if df_range.empty:
                print("[WARN] No images could be matched to Time/SWRup. Skipping.")
                continue

            phase_infos = [
                ("day", "day", "Day"),
                ("night", "night", "Night"),
            ]
        else:
            df_range["day_night"] = "all"

            print(f"Images in selected base range + pre-window days: {len(df_range)}")
            print(f"Number of target days: {len(target_days)}")

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

            print(f"\n--- {range_title}: {len(df_phase)} matched images ---")

            if df_phase.empty:
                print(f"[WARN] No images for phase '{phase_value}' in this range. Skipping.")
                continue

            range_out_root = os.path.join(OUT_ROOT, range_label)
            os.makedirs(range_out_root, exist_ok=True)

            for target_day in target_days:
                window_start, window_end, window_label, target_label = get_window_bounds(target_day, ROLLING_WINDOW_DAYS)

                group = df_phase[
                    (df_phase["datetime"] >= window_start) &
                    (df_phase["datetime"] <= window_end)
                ].copy()

                print(f"\n=== Processing {range_title} | target_day={target_label} | window={window_label} ({len(group)} images) ===")

                grids = []
                dt_list = []
                path_list = []

                for path, dt in zip(group["path"].tolist(), group["datetime"].tolist()):
                    grid = read_grid_csv(path, H, W, sep=";")

                    if grid is None:
                        print(f"[SKIP] {os.path.basename(path)} (unreadable/shape)")
                        continue

                    if np.isnan(grid).any():
                        n_nan = int(np.isnan(grid).sum())
                        print(f"[NaN] {os.path.basename(path)} contains {n_nan} NaNs")

                        if BAD_VALUE_MODE == "skip":
                            print(f"[SKIP] {os.path.basename(path)} (NaNs present)")
                            continue

                        if BAD_VALUE_MODE == "nan_to_median":
                            grid2 = fill_nans_in_grid(grid)
                            if grid2 is None:
                                print(f"[SKIP] {os.path.basename(path)} (all NaN after coercion)")
                                continue
                            grid = grid2

                    grids.append(grid)
                    dt_list.append(dt)
                    path_list.append(path)

                n_images_valid = len(grids)

                if n_images_valid < 2:
                    print(f"[WARN] target_day={target_label}: not enough valid images after cleaning ({n_images_valid}). Skipping window.")
                    continue

                stack = np.stack(grids, axis=0).astype(np.float32)

                # Round time of day to nearest half hour
                times_raw = np.array(
                    [d.hour + d.minute / 60.0 + d.second / 3600.0 for d in dt_list],
                    dtype=np.float32
                )
                times_rounded = (np.round(times_raw * 2) / 2).astype(np.float32)

                # Sort by rounded time
                idx = np.argsort(times_rounded)
                stack = stack[idx]
                times_rounded = times_rounded[idx]

                # Average all images per half-hour slot
                unique_times = np.unique(times_rounded)
                mean_grids = []
                for t in unique_times:
                    slot_mask = times_rounded == t
                    mean_grids.append(stack[slot_mask].mean(axis=0))

                times = unique_times
                stack = np.stack(mean_grids, axis=0).astype(np.float32)

                n_images_unique_time = len(times)

                if n_images_unique_time < 2:
                    print(f"[WARN] target_day={target_label}: not enough half-hour slots after averaging ({n_images_unique_time}). Skipping window.")
                    continue

                t_grid = np.linspace(0, 24, 24).astype(np.float32)

                interp_stack = np.zeros((24, H, W), dtype=np.float32)
                for i in range(H):
                    for j in range(W):
                        interp_stack[:, i, j] = np.interp(t_grid, times, stack[:, i, j]).astype(np.float32)

                if CLUSTER_MODE == "glacier":
                    fit_mask = glacier_mask
                    append_rest = False
                elif CLUSTER_MODE == "TIR":
                    fit_mask = tir_mask
                    append_rest = False
                else:
                    fit_mask = glacier_mask
                    append_rest = True

                X_fit = interp_stack[:, fit_mask].T
                X_all = interp_stack.reshape(24, -1).T

                for k in K_RANGE:
                    print(f"\n  -- k = {k} --")

                    out_folder = os.path.join(range_out_root, f"clusters_k{k}")
                    os.makedirs(out_folder, exist_ok=True)

                    km = KMeans(n_clusters=k, n_init=10, random_state=0)
                    labels_fit = km.fit_predict(X_fit)
                    centroids = km.cluster_centers_

                    full_labels = np.full((H, W), -1, dtype=int)
                    full_labels[fit_mask] = labels_fit

                    if append_rest:
                        other_mask = tir_mask & (~glacier_mask)
                        X_other = interp_stack[:, other_mask].T
                        d2 = np.sum((X_other[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                        labels_other = np.argmin(d2, axis=1)
                        full_labels[other_mask] = labels_other

                    full_labels[~tir_mask] = -1

                    labels_all = full_labels.reshape(-1)
                    mask_valid = labels_all >= 0

                    if CLUSTER_MODE == "glacier":
                        X_val = X_all[glacier_mask.reshape(-1)]
                        labs_val = labels_all[glacier_mask.reshape(-1)]
                    elif CLUSTER_MODE == "TIR":
                        X_val = X_all[tir_mask.reshape(-1)]
                        labs_val = labels_all[tir_mask.reshape(-1)]
                    else:
                        X_val = X_all[mask_valid]
                        labs_val = labels_all[mask_valid]

                    if len(np.unique(labs_val)) > 1:
                        sil = silhouette_score(X_val, labs_val)
                    else:
                        sil = np.nan

                    row = {
                        "analysis_name": ANALYSIS_NAME,
                        "range_label": range_label,
                        "range_title": range_title,
                        "range_start": range_start.strftime("%Y-%m-%d"),
                        "range_end": range_end.strftime("%Y-%m-%d"),
                        "day_night": phase_value,
                        "day_night_mode": DAY_NIGHT_MODE,
                        "rolling_window_days": ROLLING_WINDOW_DAYS,
                        "target_day": target_label,
                        "window_label": window_label,
                        "n_images_valid": n_images_valid,
                        "n_images_unique_time": n_images_unique_time,
                        "k": k,
                        "silhouette": sil
                    }
                    results.append(row)
                    all_results.append(row)
                    print(f"     silhouette = {sil:.4f}")

                    valid_clusters = sorted(np.unique(labels_all[mask_valid]))

                    raw_curves = []
                    mean_vals = []

                    for c in valid_clusters:
                        mask_c = (full_labels == c)
                        X_c = interp_stack[:, mask_c].T
                        mean_curve = X_c.mean(axis=0)
                        raw_curves.append((c, mean_curve))
                        mean_vals.append((c, float(mean_curve.mean())))

                    sorted_means = sorted(mean_vals, key=lambda x: x[1])
                    ordered_old = [c for c, _ in sorted_means][:4]
                    remap = {old: new for new, old in enumerate(ordered_old)}

                    new_labels = np.full_like(full_labels, -1)
                    for old, new in remap.items():
                        new_labels[full_labels == old] = new

                    cluster_curves = []
                    for old, new in remap.items():
                        curve = [cc for cc in raw_curves if cc[0] == old][0][1]
                        cluster_curves.append((new, curve))

                    cluster_curves = sorted(cluster_curves, key=lambda x: x[0])

                    base_stub = f"{range_label}_target_{target_label}_window_{window_label}_n{n_images_valid}_k{k}"

                    label_grid_path = os.path.join(out_folder, f"cluster_labels_{base_stub}.csv")
                    np.savetxt(label_grid_path, new_labels.astype(np.int16), delimiter=";", fmt="%d")

                    csv_path = os.path.join(out_folder, f"cluster_curves_{base_stub}.csv")
                    df_curves = pd.DataFrame(
                        {f"Cluster_{i+1}": curve for i, (_, curve) in enumerate(cluster_curves)}
                    )
                    df_curves.insert(0, "Hour", np.arange(24))
                    df_curves.to_csv(csv_path, index=False, sep=";")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    for new_label, curve in cluster_curves:
                        ax.plot(
                            np.arange(24),
                            curve,
                            label=f"Cluster {new_label + 1}",
                            color=CUSTOM_COLORS[new_label]
                        )
                    ax.set_title(
                        f"{range_title} | Cluster Curves | target {target_label} | n={n_images_valid} (k={k})\n"
                        f"window {window_label}"
                    )
                    ax.set_xlabel("Hour of day")
                    ax.set_ylabel("Temperature")
                    ax.grid(True)
                    ax.legend()
                    fig.savefig(
                        os.path.join(out_folder, f"cluster_curves_{base_stub}.png"),
                        dpi=200,
                        bbox_inches="tight"
                    )
                    plt.close(fig)

                    cmap = colors.ListedColormap(CUSTOM_COLORS[:len(cluster_curves)])
                    norm = colors.Normalize(vmin=0, vmax=len(cluster_curves) - 1)

                    masked_labels = np.ma.masked_where(new_labels == -1, new_labels)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(masked_labels, cmap=cmap, norm=norm)
                    ax.set_title(
                        f"{range_title} | Clusters | target {target_label} | n={n_images_valid} (k={k})\n"
                        f"window {window_label}"
                    )
                    ax.axis("off")
                    fig.savefig(
                        os.path.join(out_folder, f"clusters_{base_stub}.png"),
                        dpi=200,
                        bbox_inches="tight"
                    )
                    plt.close(fig)

    summary_suffix = f"{DAY_NIGHT_MODE}_rolling_{ROLLING_WINDOW_DAYS}day"
    summary_path = os.path.join(
        OUT_ROOT,
        f"silhouette_summary_{summary_suffix}.csv"
    )
    pd.DataFrame(results).to_csv(summary_path, index=False, sep=";")

    print("\nDONE ANALYSIS FAMILY.")
    print(f"Outputs stored in:\n{OUT_ROOT}")

combined_path = os.path.join(ALL_RESULTS_ROOT, "silhouette_all_families.csv")
pd.DataFrame(all_results).to_csv(combined_path, index=False, sep=";")
print(f"\nCombined silhouette summary written to:\n{combined_path}")

print("\nALL 4 ANALYSIS FAMILIES FINISHED.")