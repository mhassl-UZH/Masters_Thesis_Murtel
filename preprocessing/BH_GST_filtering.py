import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Input files
BH_FILES = [
    os.path.join(DATA_DIR, "validation_data", "BH_1.csv"),
    os.path.join(DATA_DIR, "validation_data", "BH_2.csv"),
    os.path.join(DATA_DIR, "validation_data", "BH_3.csv"),
]
GST_FILE = os.path.join(DATA_DIR, "validation_data", "GST.csv")

# Settings
SHOW_PLOTS = False
TIME_COL = "time"
FROM_DATE = "2021-01-01"
TO_DATE   = "2023-12-31"

# Output folder
OUT_DIR = os.path.join(DATA_DIR, "plots", "qc_plots_reference")
os.makedirs(OUT_DIR, exist_ok=True)

# Helpers
def read_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", low_memory=False)

    if df.shape[1] == 1:
        col0 = df.columns[0]
        sample = df.iloc[:5, 0].astype(str).str.contains(",").any()
        if sample or ("," in str(col0)):
            df = pd.read_csv(path, sep=",", engine="python", low_memory=False)

    if df.shape[1] == 1:
        df2 = pd.read_csv(path, sep=";", engine="python", low_memory=False)
        if df2.shape[1] > 1:
            df = df2

    return df


def prep_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TIME_COL not in df.columns:
        raise KeyError(f"Missing time column '{TIME_COL}'")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df = df[
        (df[TIME_COL] >= pd.to_datetime(FROM_DATE)) &
        (df[TIME_COL] <= pd.to_datetime(TO_DATE))
    ]
    df = df.sort_values(TIME_COL)
    return df


def depth_sort_key(x) -> float:
    try:
        return float(str(x))
    except Exception:
        return float("inf")


def round_to_half(x: float) -> float:
    return round(x * 2) / 2


def borehole_title(name: str) -> str:
    # "BH_1" -> "Borehole 1"
    return f"Borehole {name.split('_')[-1]}"


def gst_title(col: str) -> str:
    # "GST_1" -> "GST 1"
    return str(col).replace("_", " ")


# NaN statistics (plotted period only)
print(f"\nNaN percentages (plotted period only: {FROM_DATE} to {TO_DATE})")
print("==============================================================")

print("\nBoreholes (mean NaN% across depths):")
print("Borehole   | Mean NaN % across depths")
print("-----------+--------------------------")

for bh_path in BH_FILES:
    name = os.path.splitext(os.path.basename(bh_path))[0]  # BH_1, BH_2, BH_3
    df_stat = read_csv_robust(bh_path)
    df_stat = prep_time(df_stat)

    depth_cols_stat = [c for c in df_stat.columns if c != TIME_COL]
    for c in depth_cols_stat:
        df_stat[c] = pd.to_numeric(df_stat[c], errors="coerce")

    # BH_1: remove specific depths
    if name == "BH_1":
        depth_cols_stat = [c for c in depth_cols_stat if c not in ("3.01", "5.01")]
        df_stat = df_stat[[TIME_COL] + depth_cols_stat]

    # BH_2: round depths to nearest 0.5 (merge collisions by mean)
    if name == "BH_2":
        depth_map = {}
        for c in depth_cols_stat:
            try:
                d = float(c)
                d_round = round_to_half(d)
                depth_map.setdefault(f"{d_round:.1f}", []).append(c)
            except Exception:
                depth_map.setdefault(c, []).append(c)

        new_df = df_stat[[TIME_COL]].copy()
        for new_col, old_cols in depth_map.items():
            if len(old_cols) == 1:
                new_df[new_col] = df_stat[old_cols[0]]
            else:
                new_df[new_col] = df_stat[old_cols].mean(axis=1)

        df_stat = new_df
        depth_cols_stat = [c for c in df_stat.columns if c != TIME_COL]

    # Mean NaN% across depths
    if len(depth_cols_stat) == 0:
        mean_nan_pct = float("nan")
    else:
        per_depth_nan_pct = df_stat[depth_cols_stat].isna().mean(axis=0) * 100.0
        mean_nan_pct = float(per_depth_nan_pct.mean())

    print(f"{borehole_title(name):<11} | {mean_nan_pct:24.2f}")

print("\nGST variables (NaN% per variable):")
print("GST        | NaN %")
print("-----------+-------")

gst_stat = read_csv_robust(GST_FILE)
gst_stat = prep_time(gst_stat)

gst_cols_stat = [c for c in gst_stat.columns if c != TIME_COL]
for c in gst_cols_stat:
    gst_stat[c] = pd.to_numeric(gst_stat[c], errors="coerce")

n_time = len(gst_stat)
for c in gst_cols_stat:
    nan_pct = 100.0 * gst_stat[c].isna().sum() / n_time if n_time > 0 else float("nan")
    print(f"{gst_title(c):<11} | {nan_pct:6.2f}")

# Borehole plots
for bh_path in BH_FILES:
    name = os.path.splitext(os.path.basename(bh_path))[0]  # BH_1, BH_2, BH_3
    df = read_csv_robust(bh_path)
    df = prep_time(df)

    depth_cols = [c for c in df.columns if c != TIME_COL]

    for c in depth_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # BH_1: remove specific depths
    if name == "BH_1":
        depth_cols = [c for c in depth_cols if c not in ("3.01", "5.01")]
        df = df[[TIME_COL] + depth_cols]

    # BH_2: round depths to nearest 0.5 (merge collisions by mean)
    if name == "BH_2":
        depth_map = {}
        for c in depth_cols:
            try:
                d = float(c)
                d_round = round_to_half(d)
                depth_map.setdefault(f"{d_round:.1f}", []).append(c)
            except Exception:
                depth_map.setdefault(c, []).append(c)

        new_df = df[[TIME_COL]].copy()
        for new_col, old_cols in depth_map.items():
            if len(old_cols) == 1:
                new_df[new_col] = df[old_cols[0]]
            else:
                new_df[new_col] = df[old_cols].mean(axis=1)

        df = new_df
        depth_cols = [c for c in df.columns if c != TIME_COL]

    depth_cols = sorted(depth_cols, key=depth_sort_key)

    n = len(depth_cols)
    cmap = plt.cm.viridis_r
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig = plt.figure(figsize=(12, 5))
    for col, color in zip(depth_cols, colors):
        plt.plot(df[TIME_COL], df[col], linewidth=0.8, color=color, label=str(col))

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title(borehole_title(name))
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(title="Depth (m)", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{name}_all_depths_{FROM_DATE}_to_{TO_DATE}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    print(f"Saved: {out_png}")

# GST plots
gst = read_csv_robust(GST_FILE)
gst = prep_time(gst)

gst_cols = [c for c in gst.columns if c != TIME_COL]
for c in gst_cols:
    gst[c] = pd.to_numeric(gst[c], errors="coerce")

for c in gst_cols:
    fig = plt.figure(figsize=(12, 4))
    plt.plot(gst[TIME_COL], gst[c], linewidth=0.8)
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title(gst_title(c))
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"GST_{c}_{FROM_DATE}_to_{TO_DATE}.png")
    fig.savefig(out_png, dpi=200)

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    print(f"Saved: {out_png}")

print(f"\nAll plots saved to: {OUT_DIR}")
