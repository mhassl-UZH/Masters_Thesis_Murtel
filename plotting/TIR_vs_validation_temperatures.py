import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

CSV_PATH = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")
OUT_DIR  = os.path.join(DATA_DIR, "plots", "validation_scatterplots")

RAW_COLOR  = "#d95f02"
CORR_COLOR = "#7570b3"

FIGSIZE = (9, 7)
DPI = 300
ALPHA = 0.50
MARKER_SIZE = 30

os.makedirs(OUT_DIR, exist_ok=True)

# Column groups

GST_GROUP = [
    ("validation_GST_GST_1", "GST_1_uncorr", "GST_1_corr", "o", "GST 1"),
    ("validation_GST_GST_2", "GST_2_uncorr", "GST_2_corr", "^", "GST 2"),
    ("validation_GST_GST_3", "GST_3_uncorr", "GST_3_corr", "s", "GST 3"),
    ("validation_GST_GST_4", "GST_4_uncorr", "GST_4_corr", "*", "GST 4"),
    ("validation_GST_GST_5", "GST_5_uncorr", "GST_5_corr", "h", "GST 5"),
]

BH_GROUP = [
    ("validation_BH_1_0.25", "BH_1_uncorr", "BH_1_corr", "o", "BH 1 (0.25 m)"),
    ("validation_BH_2_0.55", "BH_2_uncorr", "BH_2_corr", "^", "BH 2 (0.55 m)"),
    ("validation_BH_3_0.5",  "BH_3_uncorr", "BH_3_corr", "s", "BH 3 (0.5 m)"),
]

METEO_GROUP = [
    ("LW_temp", "Meteo_Station_uncorr", "Meteo_Station_corr", "o", "Longwave Temperature"),
    ("validation_ridge_Corrected_Target_Ambient_LW_Lab", "radio_ridge_uncorr", "radio_ridge_corr", "^", "Radiometer Ridge"),
    ("validation_furrow_Corrected_Target_Ambient_LW_Lab", "radio_furrow_uncorr", "radio_furrow_corr", "s", "Radiometer Furrow"),
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

    dt = pd.to_datetime(df[time_col], errors="coerce")
    iso = dt.dt.isocalendar()

    years = iso.year.astype("Int64")
    weeks = iso.week.astype("Int64")

    mask_2021 = (years == 2021) & (weeks >= 29) & (weeks <= 37)
    mask_2022 = (years == 2022) & (weeks >= 25) & (weeks <= 43)

    mask = dt.notna() & (mask_2021 | mask_2022)

    df_out = df.loc[mask].copy()
    print(f"Rows after filtering: {len(df_out)}")

    return df_out


def compute_global_limits(df, triples):
    vals = []

    for xcol, yraw, ycorr, _, _ in triples:
        if all(c in df.columns for c in [xcol, yraw, ycorr]):
            vals.extend(pd.to_numeric(df[xcol], errors="coerce").dropna())
            vals.extend(pd.to_numeric(df[yraw], errors="coerce").dropna())
            vals.extend(pd.to_numeric(df[ycorr], errors="coerce").dropna())

    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        raise RuntimeError("No valid numeric values found to compute plot limits.")

    vmin = np.min(vals)
    vmax = np.max(vals)
    pad = 0.05 * (vmax - vmin)

    return vmin - pad, vmax + pad


def compute_regression_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    if np.allclose(np.nanstd(x), 0):
        return None

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = np.nan if np.isclose(ss_tot, 0) else 1 - (ss_res / ss_tot)

    return slope, intercept, r2


def add_standard_legend(ax, triples):
    color_legend = [
        Line2D([0], [0], marker='o', color='w', label='Raw',
               markerfacecolor=RAW_COLOR, markeredgecolor=RAW_COLOR,
               markersize=8, alpha=ALPHA),
        Line2D([0], [0], marker='o', color='w', label='Corrected',
               markerfacecolor=CORR_COLOR, markeredgecolor=CORR_COLOR,
               markersize=8, alpha=ALPHA),
        Line2D([0], [0], linestyle='--', color='k', label='1:1 Line')
    ]

    seen = {}
    for _, _, _, marker, label in triples:
        if marker not in seen:
            seen[marker] = label

    shape_legend = [
        Line2D([0], [0], marker=m, color='k', linestyle='None',
               label=lbl, markersize=8)
        for m, lbl in seen.items()
    ]

    ax.legend(handles=color_legend + shape_legend, loc="upper left")


# Plots

def make_group_plot(df, triples, title, out_name, lim_min, lim_max):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for xcol, yraw, ycorr, marker, label in triples:
        if not all(c in df.columns for c in [xcol, yraw, ycorr]):
            continue

        x = pd.to_numeric(df[xcol], errors="coerce")
        y_raw = pd.to_numeric(df[yraw], errors="coerce")
        y_corr = pd.to_numeric(df[ycorr], errors="coerce")

        mask_raw = x.notna() & y_raw.notna()
        mask_corr = x.notna() & y_corr.notna()

        ax.scatter(x[mask_raw], y_raw[mask_raw],
                   s=MARKER_SIZE, c=RAW_COLOR, marker=marker,
                   alpha=ALPHA, linewidths=0)

        ax.scatter(x[mask_corr], y_corr[mask_corr],
                   s=MARKER_SIZE, c=CORR_COLOR, marker=marker,
                   alpha=ALPHA, linewidths=0)

    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.0)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_title(title)
    ax.set_xlabel("Validation Temperature [°C]")
    ax.set_ylabel("TIR Temperature [°C]")
    ax.grid(True, alpha=0.3)

    add_standard_legend(ax, triples)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=DPI)
    plt.close()


def make_single_sensor_plot(df, triple, title, out_name, lim_min, lim_max):
    xcol, yraw, ycorr, marker, label = triple

    if not all(c in df.columns for c in [xcol, yraw, ycorr]):
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = pd.to_numeric(df[xcol], errors="coerce")
    y_raw = pd.to_numeric(df[yraw], errors="coerce")
    y_corr = pd.to_numeric(df[ycorr], errors="coerce")

    mask_raw = x.notna() & y_raw.notna()
    mask_corr = x.notna() & y_corr.notna()

    x_raw = x[mask_raw].to_numpy(dtype=float)
    y_raw_vals = y_raw[mask_raw].to_numpy(dtype=float)

    x_corr = x[mask_corr].to_numpy(dtype=float)
    y_corr_vals = y_corr[mask_corr].to_numpy(dtype=float)

    ax.scatter(x_raw, y_raw_vals,
               s=MARKER_SIZE, c=RAW_COLOR, alpha=ALPHA, linewidths=0)

    ax.scatter(x_corr, y_corr_vals,
               s=MARKER_SIZE, c=CORR_COLOR, alpha=ALPHA, linewidths=0)

    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.0)

    raw_stats = compute_regression_stats(x_raw, y_raw_vals)
    corr_stats = compute_regression_stats(x_corr, y_corr_vals)

    xline = np.array([lim_min, lim_max], dtype=float)

    if raw_stats is not None:
        ax.plot(xline, raw_stats[0] * xline + raw_stats[1],
                color=RAW_COLOR, lw=1.5,
                label=f"Raw fit (R² = {raw_stats[2]:.2f})")

    if corr_stats is not None:
        ax.plot(xline, corr_stats[0] * xline + corr_stats[1],
                color=CORR_COLOR, lw=1.5,
                label=f"Corrected fit (R² = {corr_stats[2]:.2f})")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_title(title)
    ax.set_xlabel("Validation Temperature [°C]")
    ax.set_ylabel("TIR Temperature [°C]")
    ax.grid(True, alpha=0.3)

    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=DPI)
    plt.close()


# Main

def main():
    df = read_csv_robust(CSV_PATH)
    df = apply_time_filter(df)

    lim_min, lim_max = compute_global_limits(df, ALL_GROUPS)

    make_group_plot(
        df, GST_GROUP,
        "TIR Temperatures vs Validation Temperatures: GST",
        "GST.png", lim_min, lim_max
    )

    make_group_plot(
        df, BH_GROUP,
        "TIR Temperatures vs Validation Temperatures: Boreholes",
        "BH.png", lim_min, lim_max
    )

    make_group_plot(
        df, METEO_GROUP,
        "TIR Temperatures vs Validation Temperatures: Longwave Temperature & Radiometers",
        "validation_radiometer_combined.png", lim_min, lim_max
    )

    for i, triple in enumerate(GST_GROUP, 1):
        _, _, _, _, label = triple
        make_single_sensor_plot(
            df, triple,
            f"TIR Temperatures vs Validation Temperatures: {label}",
            f"GST_{i}.png", lim_min, lim_max
        )

    for triple in BH_GROUP:
        _, _, _, _, label = triple
        out_name = (
            label.replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace(".", "")
            + ".png"
        )

        make_single_sensor_plot(
            df, triple,
            f"TIR Temperatures vs Validation Temperatures: {label}",
            out_name, lim_min, lim_max
        )

    meteo_names = [
        "validation_longwave_temperature.png",
        "validation_radiometer_ridge.png",
        "validation_radiometer_furrow.png",
    ]

    for triple, out_name in zip(METEO_GROUP, meteo_names):
        _, _, _, _, label = triple
        make_single_sensor_plot(
            df, triple,
            f"TIR Temperatures vs Validation Temperatures: {label}",
            out_name, lim_min, lim_max
        )

    print("Done.")


if __name__ == "__main__":
    main()