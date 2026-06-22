# Generates 4 workflow visualization plots:
# (a) raw daily curve of one pixel, (b) mean curve of one pixel over rolling window,
# (c) all pixel mean curves, (d) all curves colored by k-means cluster with centroids.

import os
import glob
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

CSV_FOLDER = os.path.join(DATA_DIR, "corrected")
MASK_PATH  = os.path.join(DATA_DIR, "glacier_mask.png")
OUT_PATH   = os.path.join(DATA_DIR, "plots", "workflow_visualization")

PIX_ROW, PIX_COL = 150, 170

TARGET_DAY = datetime(2022, 5, 28)
ROLLING_WINDOW_DAYS = 5

W, H = 336, 252
K    = 2

CLUSTER_COLORS  = ["#6baed6", "#fb6a4a"]
CENTROID_COLORS = ["#2166ac", "#b2182b"]

# Derived dates

SINGLE_DAY_START = TARGET_DAY.replace(hour=0,  minute=0,  second=0)
SINGLE_DAY_END   = TARGET_DAY.replace(hour=23, minute=59, second=59)
WINDOW_START     = (TARGET_DAY - timedelta(days=ROLLING_WINDOW_DAYS - 1)).replace(
                       hour=0, minute=0, second=0)
WINDOW_END       = TARGET_DAY.replace(hour=23, minute=59, second=59)

window_label = (
    f"{WINDOW_START.strftime('%Y-%m-%d')} to {WINDOW_END.strftime('%Y-%m-%d')}"
)

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

print(f"Glacier mask : {glacier_mask.sum()} pixels")
print(f"TIR mask     : {tir_mask.sum()} pixels")

# Helpers

TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")


def name_to_dt(name: str):
    m = TS_RE.search(name)
    if not m:
        return None
    yy, MM, DD, hh, mm, ss = map(int, m.groups())
    return datetime(2000 + yy, MM, DD, hh, mm, ss)


def read_grid_csv(path: str) -> np.ndarray | None:
    try:
        df0 = pd.read_csv(path, sep=";", header=None, skiprows=1, engine="python")
    except Exception:
        return None
    arr = df0.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if arr.shape != (H, W) or np.isnan(arr).any():
        return None
    return arr


def load_images(dt_start: datetime, dt_end: datetime):
    files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*_corr.csv")))
    grids, dts = [], []
    for f in files:
        dt = name_to_dt(f)
        if dt is None or not (dt_start <= dt <= dt_end):
            continue
        grid = read_grid_csv(f)
        if grid is None:
            continue
        grids.append(grid)
        dts.append(dt)
    print(f"  Loaded {len(grids)} images  ({dt_start.date()} to {dt_end.date()})")
    return grids, dts


def to_daytime(dts):
    return np.array(
        [d.hour + d.minute / 60.0 + d.second / 3600.0 for d in dts],
        dtype=np.float32,
    )


def build_mean_interp_stack(grids, dts):
    stack         = np.stack(grids, axis=0).astype(np.float32)
    times_raw     = to_daytime(dts)
    times_rounded = (np.round(times_raw * 2) / 2).astype(np.float32)

    idx           = np.argsort(times_rounded)
    stack         = stack[idx]
    times_rounded = times_rounded[idx]

    unique_times = np.unique(times_rounded)
    slot_means   = np.stack(
        [stack[times_rounded == t].mean(axis=0) for t in unique_times],
        axis=0,
    ).astype(np.float32)

    t_grid        = np.arange(24, dtype=np.float32)
    interp_stack  = np.zeros((24, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            interp_stack[:, i, j] = np.interp(t_grid, unique_times, slot_means[:, i, j])

    return interp_stack, unique_times, slot_means


os.makedirs(OUT_PATH, exist_ok=True)
t_grid = np.arange(24, dtype=np.float32)

# Load data

print(f"\nLoading single-day images ({TARGET_DAY.date()})...")
grids_day, dts_day = load_images(SINGLE_DAY_START, SINGLE_DAY_END)

print(f"\nLoading window images ({window_label})...")
grids_win, dts_win = load_images(WINDOW_START, WINDOW_END)

print("\nBuilding mean curves for the rolling window...")
interp_win, slot_times_win, slot_means_win = build_mean_interp_stack(grids_win, dts_win)

pix_interp = interp_win[:, PIX_ROW, PIX_COL]

# Single-day pixel data
stack_day = np.stack(grids_day, axis=0).astype(np.float32)
times_day = to_daytime(dts_day)
idx_sort  = np.argsort(times_day)
times_day = times_day[idx_sort]
pix_raw   = stack_day[idx_sort, PIX_ROW, PIX_COL]

# All-pixel data (shared by plots c and d)
X_tir      = interp_win[:, tir_mask].T
sample_idx = np.arange(0, X_tir.shape[0], max(1, X_tir.shape[0] // 1000))
print(f"  {X_tir.shape[0]} valid pixels in TIR mask, plotting {len(sample_idx)} sampled")

# Shared y range across all 4 plots
y_min = float(min(pix_raw.min(), pix_interp.min(), X_tir[sample_idx].min()))
y_max = float(max(pix_raw.max(), pix_interp.max(), X_tir[sample_idx].max()))
pad   = (y_max - y_min) * 0.05
y_min -= pad
y_max += pad

# Plot (a) — raw daily curve of one pixel

print("\nPlot (a): raw daily curve of one pixel...")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times_day, pix_raw, "-o", color="black")
ax.set_xlabel("Time of day [h]")
ax.set_ylabel("Temperature [°C]")
ax.set_title(
    f"Raw Daily Temperature Curve\n"
    f"Pixel ({PIX_ROW}, {PIX_COL})  –  {TARGET_DAY.strftime('%Y-%m-%d')}"
)
ax.set_xlim(0, 24)
ax.set_ylim(y_min, y_max)
ax.set_xticks(range(0, 25, 2))
ax.grid(alpha=0.3)
fig.tight_layout()
p1 = os.path.join(OUT_PATH, "plot1_raw_daily_pixel.png")
fig.savefig(p1, dpi=200)
plt.close(fig)
print(f"  Saved: {p1}")

# Plot (b) — mean daily curve of one pixel over rolling window

print("Plot (b): mean daily curve of one pixel...")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_grid, pix_interp, "-", color="black")
ax.set_xlabel("Time of day [h]")
ax.set_ylabel("Temperature [°C]")
ax.set_title(
    f"Mean Daily Temperature Curve\n"
    f"Pixel ({PIX_ROW}, {PIX_COL}) – {window_label}"
)
ax.set_xlim(0, 24)
ax.set_ylim(y_min, y_max)
ax.set_xticks(range(0, 25, 2))
ax.grid(alpha=0.3)
fig.tight_layout()
p2 = os.path.join(OUT_PATH, "plot2_window_mean_pixel.png")
fig.savefig(p2, dpi=200)
plt.close(fig)
print(f"  Saved: {p2}")

# Plot (c) — all pixel mean curves

print("Plot (c): all pixel mean curves...")

segs = [np.column_stack([t_grid, X_tir[i]]) for i in sample_idx]
lc   = LineCollection(segs, colors="black", linewidths=0.5, alpha=0.15)

fig, ax = plt.subplots(figsize=(7, 5))
ax.add_collection(lc)
ax.set_xlabel("Time of day [h]")
ax.set_ylabel("Temperature [°C]")
ax.set_title(
    f"Mean Daily Temperature Curves\n"
    f"All Pixels – {window_label}"
)
ax.set_xlim(0, 24)
ax.set_ylim(y_min, y_max)
ax.set_xticks(range(0, 25, 2))
ax.grid(alpha=0.3)
fig.tight_layout()
p3 = os.path.join(OUT_PATH, "plot3_all_pixel_curves.png")
fig.savefig(p3, dpi=200)
plt.close(fig)
print(f"  Saved: {p3}")

# Plot (d) — all curves colored by cluster with centroids

print(f"Plot (d): k-means clustering (k={K}) + all curves colored by cluster...")

km         = KMeans(n_clusters=K, n_init=10, random_state=0)
labels_fit = km.fit_predict(X_tir)
centroids  = km.cluster_centers_

# Sort clusters cold → warm for consistent coloring
order           = np.argsort(centroids.mean(axis=1))
remap           = {old: new for new, old in enumerate(order)}
labels_remapped = np.array([remap[l] for l in labels_fit], dtype=int)
sorted_centroids = centroids[order]

fig, ax = plt.subplots(figsize=(7, 5))

for new_k in range(K):
    idx_k  = sample_idx[labels_remapped[sample_idx] == new_k]
    segs_k = [np.column_stack([t_grid, X_tir[i]]) for i in idx_k]
    lc_k   = LineCollection(
        segs_k,
        colors=CLUSTER_COLORS[new_k],
        linewidths=0.5,
        alpha=0.15,
    )
    ax.add_collection(lc_k)

# Centroid curves
for new_k in range(K):
    ax.plot(
        t_grid, sorted_centroids[new_k],
        color=CENTROID_COLORS[new_k], linewidth=3,
        label=f"Cluster {new_k + 1}",
        zorder=5,
    )

ax.set_xlabel("Time of day [h]")
ax.set_ylabel("Temperature [°C]")
ax.set_title(
    f"Mean Daily Temperature Curves Grouped by K-Means (k={K})\n"
    f"{window_label}"
)
ax.set_xlim(0, 24)
ax.set_ylim(y_min, y_max)
ax.set_xticks(range(0, 25, 2))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
p4 = os.path.join(OUT_PATH, "plot4_clustered_curves.png")
fig.savefig(p4, dpi=200)
plt.close(fig)
print(f"  Saved: {p4}")

print(f"\nAll 4 plots saved to:\n{OUT_PATH}")
