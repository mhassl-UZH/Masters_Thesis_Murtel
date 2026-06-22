import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
MODE = "emissivity"   # "emissivity" or "transmissivity"

MODE_CONFIG = {
    "emissivity": {
        "out_dir": os.path.join(DATA_DIR, "plots", "validation_scatterplots_emissivity_compare"),
        "suffix":  "_emissivity_comparison.png",
        "datasets": [
            ("E = 0.90", "insert e=0.90 path", "#d95f02"),
            ("E = 0.92", "insert e=0.92 path", "#7570b3"),
            ("E = 0.94", "insert e=0.94 path", "#1b9e77"),
        ],
    },
    "transmissivity": {
        "out_dir": r"F:\UZH\masterarbeit\images_260225\plots\validation_scatterplots_transmissivity_compare",
        "suffix":  "_transmissivity_comparison.png",
        "datasets": [
            ("τ = 0.86", "insert tau=0.86 path", "#d95f02"),
            ("τ = 0.90", "insert tau=0.90 path", "#7570b3"),
            ("τ = 0.96", "insert tau=0.96 path", "#1b9e77"),
        ],
    },
}

cfg      = MODE_CONFIG[MODE]
OUT_DIR  = cfg["out_dir"]
SUFFIX   = cfg["suffix"]
DATASETS = cfg["datasets"]

FIGSIZE     = (9, 7)
DPI         = 300
ALPHA       = 0.50
MARKER_SIZE = 30

os.makedirs(OUT_DIR, exist_ok=True)

# Column groups
GST_GROUP = [
    ("validation_GST_GST_1", "GST_1_corr", "o", "GST 1"),
    ("validation_GST_GST_2", "GST_2_corr", "^", "GST 2"),
    ("validation_GST_GST_3", "GST_3_corr", "s", "GST 3"),
    ("validation_GST_GST_4", "GST_4_corr", "*", "GST 4"),
    ("validation_GST_GST_5", "GST_5_corr", "h", "GST 5"),
]

BH_GROUP = [
    ("validation_BH_1_0.25", "BH_1_corr", "o", "BH 1 (0.25 m)"),
    ("validation_BH_2_0.55", "BH_2_corr", "^", "BH 2 (0.55 m)"),
    ("validation_BH_3_0.5",  "BH_3_corr", "s", "BH 3 (0.5 m)"),
]

METEO_GROUP = [
    ("LW_temp", "Meteo_Station_corr", "o", "Longwave Temperature"),
    ("validation_ridge_Corrected_Target_Ambient_LW_Lab", "radio_ridge_corr",  "^", "Radiometer Ridge"),
    ("validation_furrow_Corrected_Target_Ambient_LW_Lab", "radio_furrow_corr", "s", "Radiometer Furrow"),
]

ALL_GROUPS = GST_GROUP + BH_GROUP + METEO_GROUP

# Helpers
def read_csv_robust(path):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise RuntimeError(f"Could not read CSV properly: {path}")


def find_time_column(df):
    for col in ["time", "Time", "timestamp", "Timestamp", "datetime", "Datetime", "date", "Date"]:
        if col in df.columns:
            return col
    raise RuntimeError("No timestamp column found in CSV")


def apply_time_filter(df):
    time_col = find_time_column(df)
    dt  = pd.to_datetime(df[time_col], errors="coerce")
    iso = dt.dt.isocalendar()
    years = iso.year.astype("Int64")
    weeks = iso.week.astype("Int64")
    mask = dt.notna() & (
        ((years == 2021) & (weeks >= 29) & (weeks <= 37)) |
        ((years == 2022) & (weeks >= 25) & (weeks <= 43))
    )
    return df.loc[mask].copy()


def load_dataset(csv_path):
    df = read_csv_robust(csv_path)
    df = apply_time_filter(df)
    print(f"Loaded {os.path.basename(csv_path)}: {len(df)} rows after filtering")
    return df


def compute_global_limits(dfs, triples):
    vals = []
    for df in dfs:
        for xcol, ycol, _, _ in triples:
            if all(c in df.columns for c in [xcol, ycol]):
                vals.extend(pd.to_numeric(df[xcol], errors="coerce").dropna())
                vals.extend(pd.to_numeric(df[ycol], errors="coerce").dropna())
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise RuntimeError("No valid numeric values found to compute plot limits.")
    pad = 0.05 * (vals.max() - vals.min())
    return float(vals.min()) - pad, float(vals.max()) + pad


def compute_regression_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.allclose(np.nanstd(x), 0):
        return None
    slope, intercept = np.polyfit(x, y, 1)
    yhat   = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if np.isclose(ss_tot, 0) else 1 - ss_res / ss_tot
    return slope, intercept, r2


def add_group_legend(ax, triples):
    color_handles = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color, markeredgecolor=color,
               markersize=8, alpha=ALPHA)
        for label, _, color in DATASETS
    ] + [Line2D([0], [0], linestyle='--', color='k', label='1:1 Line')]

    seen = {}
    for _, _, marker, label in triples:
        if marker not in seen:
            seen[marker] = label

    shape_handles = [
        Line2D([0], [0], marker=m, color='k', linestyle='None', label=lbl, markersize=8)
        for m, lbl in seen.items()
    ]
    ax.legend(handles=color_handles + shape_handles, loc="upper left")


# Plots
def make_group_plot(dfs_with_meta, triples, title, out_name, lim_min, lim_max):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for _, df, color in dfs_with_meta:
        for xcol, ycol, marker, _ in triples:
            if not all(c in df.columns for c in [xcol, ycol]):
                continue
            x = pd.to_numeric(df[xcol], errors="coerce")
            y = pd.to_numeric(df[ycol], errors="coerce")
            mask = x.notna() & y.notna()
            ax.scatter(x[mask], y[mask], s=MARKER_SIZE, c=color,
                       marker=marker, alpha=ALPHA, linewidths=0)

    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.0)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title(title)
    ax.set_xlabel("Validation Temperature [°C]")
    ax.set_ylabel("TIR Temperature [°C]")
    ax.grid(True, alpha=0.3)
    add_group_legend(ax, triples)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=DPI)
    plt.close()


def make_single_sensor_plot(dfs_with_meta, triple, title, out_name, lim_min, lim_max):
    xcol, ycol, _, _ = triple
    fig, ax = plt.subplots(figsize=FIGSIZE)
    legend_handles = [Line2D([0], [0], linestyle='--', color='k', label='1:1 Line')]
    plotted_any = False
    xline = np.array([lim_min, lim_max], dtype=float)

    for dataset_label, df, color in dfs_with_meta:
        if not all(c in df.columns for c in [xcol, ycol]):
            continue
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        mask = x.notna() & y.notna()
        x_vals = x[mask].to_numpy(dtype=float)
        y_vals = y[mask].to_numpy(dtype=float)
        if len(x_vals) == 0:
            continue
        plotted_any = True
        ax.scatter(x_vals, y_vals, s=MARKER_SIZE, c=color, alpha=ALPHA, linewidths=0)
        stats = compute_regression_stats(x_vals, y_vals)
        if stats is not None:
            slope, intercept, r2 = stats
            ax.plot(xline, slope * xline + intercept, color=color, lw=1.5)
            legend_handles.append(
                Line2D([0], [0], linestyle='-', color=color,
                       label=f"{dataset_label} fit (R² = {r2:.2f})")
            )
        else:
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markeredgecolor=color,
                       markersize=8, alpha=ALPHA, label=dataset_label)
            )

    if not plotted_any:
        plt.close()
        return

    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.0)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title(title)
    ax.set_xlabel("Validation Temperature [°C]")
    ax.set_ylabel("TIR Temperature [°C]")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=DPI)
    plt.close()


# Main
def main():
    print(f"Mode: {MODE}")

    dfs_with_meta = [(label, load_dataset(path), color) for label, path, color in DATASETS]
    dfs_only = [df for _, df, _ in dfs_with_meta]
    lim_min, lim_max = compute_global_limits(dfs_only, ALL_GROUPS)

    make_group_plot(dfs_with_meta, GST_GROUP,
                    "TIR Temperatures vs Validation Temperatures: GST",
                    "GST" + SUFFIX, lim_min, lim_max)

    make_group_plot(dfs_with_meta, BH_GROUP,
                    "TIR Temperatures vs Validation Temperatures: Boreholes",
                    "BH" + SUFFIX, lim_min, lim_max)

    make_group_plot(dfs_with_meta, METEO_GROUP,
                    "TIR Temperatures vs Validation Temperatures: Longwave Temperature & Radiometers",
                    "validation_radiometer_combined" + SUFFIX, lim_min, lim_max)

    for i, triple in enumerate(GST_GROUP, 1):
        _, _, _, label = triple
        make_single_sensor_plot(dfs_with_meta, triple,
                                f"TIR Temperatures vs Validation Temperatures: {label}",
                                f"GST_{i}" + SUFFIX, lim_min, lim_max)

    for triple in BH_GROUP:
        _, _, _, label = triple
        slug = label.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        make_single_sensor_plot(dfs_with_meta, triple,
                                f"TIR Temperatures vs Validation Temperatures: {label}",
                                slug + SUFFIX, lim_min, lim_max)

    for triple, name in zip(METEO_GROUP, ["validation_longwave_temperature",
                                          "validation_radiometer_ridge",
                                          "validation_radiometer_furrow"]):
        _, _, _, label = triple
        make_single_sensor_plot(dfs_with_meta, triple,
                                f"TIR Temperatures vs Validation Temperatures: {label}",
                                name + SUFFIX, lim_min, lim_max)

    print("Done.")


if __name__ == "__main__":
    main()
