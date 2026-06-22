import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
RAW_FOLDER = os.path.join(DATA_DIR, "decoded_images_filtered")
CORR_FOLDER = os.path.join(DATA_DIR, "corrected")
DIST_PATH = os.path.join(DATA_DIR, "CSVs", "murtel_distance_filled.csv")

OUT_DIR = os.path.join(DATA_DIR, "plots", "difference_plots")
os.makedirs(OUT_DIR, exist_ok=True)

TS_RE = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

def extract_ts_key(fname):
    m = TS_RE.search(fname)
    if not m:
        return None
    return "m" + "".join(m.groups())

MAX_POINTS_KDE = 300_000   # lower = faster KDE
RANDOM_SEED = 42

YEAR_FILTER = 2022         # e.g. 2022, or None for all years
MONTH_FILTER = None        # e.g. [6, 7, 8] or None

MANUAL_YLIM = None         # e.g. (-3, 3)

np.random.seed(RANDOM_SEED)

# Load distance
dist_df = pd.read_csv(DIST_PATH)
dist_df = dist_df.apply(pd.to_numeric, errors="coerce")
dist_df = dist_df.dropna(axis=1, how="all")

dist = dist_df.values.astype(float)
dist_flat = dist.flatten()

print("Distance shape:", dist.shape)

# Helpers
def parse_timestamp(fname):
    base = os.path.basename(fname)
    ts = base.split("_")[0][1:]  # remove "m"
    return pd.to_datetime(ts[:12], format="%y%m%d%H%M%S", errors="coerce")

def get_season(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

def keep_timestamp(dt):
    if pd.isna(dt):
        return False
    if YEAR_FILTER is not None and dt.year != YEAR_FILTER:
        return False
    if MONTH_FILTER is not None and dt.month not in MONTH_FILTER:
        return False
    return True

def padded_limits(values, frac=0.03):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = 1.0
    else:
        pad = (vmax - vmin) * frac
    return (vmin - pad, vmax + pad)

# Storage
season_data = {
    "winter": [],
    "spring": [],
    "summer": [],
    "autumn": []
}

all_data = []

# Index raw files by timestamp key
raw_paths = glob.glob(os.path.join(RAW_FOLDER, "*.csv"))
raw_by_key = {}
for p in raw_paths:
    key = extract_ts_key(os.path.basename(p))
    if key is not None and key not in raw_by_key:
        raw_by_key[key] = p

print(f"Indexed {len(raw_by_key)} raw CSVs by timestamp key")

# Load corrected files and compute the difference image (corrected - raw) on the fly
files = sorted(glob.glob(os.path.join(CORR_FOLDER, "*_corr.csv")))
print(f"Found {len(files)} corrected files")

# Process files
for i, f_corr in enumerate(files):

    fname_corr = os.path.basename(f_corr)
    dt = parse_timestamp(fname_corr)
    if not keep_timestamp(dt):
        continue

    key = extract_ts_key(fname_corr)
    f_raw = raw_by_key.get(key)
    if f_raw is None:
        print(f"[SKIP] no matching raw file for {fname_corr}")
        continue

    season = get_season(dt)

    try:
        corr = pd.read_csv(f_corr, sep=";", header=None, skiprows=1).values.astype(float)
        raw  = pd.read_csv(f_raw, sep=";", header=None, skiprows=8).values.astype(float)
    except Exception as e:
        print(f"[ERROR] {fname_corr}: {e}")
        continue

    if corr.shape != raw.shape:
        print(f"[SKIP] shape mismatch between corrected and raw for {fname_corr}: corr={corr.shape}, raw={raw.shape}")
        continue

    diff_flat = (corr - raw).flatten()

    if diff_flat.shape[0] != dist_flat.shape[0]:
        print(f"[SKIP] shape mismatch in flattened arrays for {fname_corr}")
        continue

    valid = np.isfinite(diff_flat) & np.isfinite(dist_flat)

    if valid.sum() == 0:
        continue

    chunk = np.column_stack((dist_flat[valid], diff_flat[valid]))
    season_data[season].append(chunk)
    all_data.append(chunk)

    if (i + 1) % 500 == 0:
        print(f"[PROGRESS] {i+1}/{len(files)} processed")

if len(all_data) == 0:
    raise RuntimeError("No valid data found after filtering.")

# Global axis limits
all_arr = np.vstack(all_data)
global_xlim = padded_limits(all_arr[:, 0], frac=0.0)  # no side padding
global_ylim = MANUAL_YLIM if MANUAL_YLIM is not None else padded_limits(all_arr[:, 1], frac=0.03)

print("Global xlim:", global_xlim)
print("Global ylim:", global_ylim)
print("Total valid points before any subsampling:", all_arr.shape[0])

# Plotting
def plot_density(data, title, out_path, xlim, ylim):

    if len(data) == 0:
        print(f"[WARN] no data for {title}")
        return

    arr = np.vstack(data)
    print(f"{title}: {arr.shape[0]} points before subsampling")

    if arr.shape[0] > MAX_POINTS_KDE:
        idx = np.random.choice(arr.shape[0], MAX_POINTS_KDE, replace=False)
        arr = arr[idx]
        print(f"{title}: reduced to {arr.shape[0]} points")

    x = arr[:, 0]
    y = arr[:, 1]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sparse points first so dense points render on top
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(
        x, y,
        c=z,
        s=3,
        cmap="viridis",
        edgecolors="none",
        alpha=1.0
    )

    ax.set_xlabel("Distance to Camera [m]")
    ax.set_ylabel("ΔT (Corrected – Raw) [°C]")
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.margins(x=0, y=0)

    # Y-axis ticks every 2 °C
    if ylim is not None:
        ymin, ymax = ylim
        yticks = np.arange(np.floor(ymin / 2) * 2, np.ceil(ymax / 2) * 2 + 1, 2)
        ax.set_yticks(yticks)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.minorticks_on()

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Point Density")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# Create plots
year_txt = f"{YEAR_FILTER}" if YEAR_FILTER is not None else "All years"

plot_density(
    all_data,
    f"ΔT vs Distance ({year_txt})",
    os.path.join(OUT_DIR, f"density_all_{year_txt.replace(' ', '_')}.png"),
    global_xlim,
    global_ylim
)

for season in ["winter", "spring", "summer", "autumn"]:
    plot_density(
        season_data[season],
        f"ΔT vs Distance ({season.capitalize()}, {year_txt})",
        os.path.join(OUT_DIR, f"density_{season}_{year_txt.replace(' ', '_')}.png"),
        global_xlim,
        global_ylim
    )

print("DONE: density plots created.")