#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# ---------------------------------------------------------------------
# Output folder
OUT_DIR = os.path.join(DATA_DIR, "plots", "validation_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# File paths
PIXEL_CSV = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")
BH_CSV    = os.path.join(DATA_DIR, "validation_data", "BH_1.csv")
GST_CSV   = os.path.join(DATA_DIR, "validation_data", "GST.csv")
METEO_CSV = os.path.join(DATA_DIR, "CSVs", "murtel_met_qc.csv")

# Time range
FULL_START = pd.Timestamp("2022-01-01")
FULL_END   = pd.Timestamp("2022-12-31")

lw = 1

# ---------------------------------------------------------------------
# Load pixel temperature data
df_pix = pd.read_csv(PIXEL_CSV, sep=";", decimal=".")
df_pix["Time"] = pd.to_datetime(df_pix["Time"], errors="coerce")
df_pix = df_pix.dropna(subset=["Time"]).sort_values("Time")

df_pix_roll = (
    df_pix.set_index("Time")
          .select_dtypes(include=[np.number])
          .rolling("24H")
          .mean()
)

# ---------------------------------------------------------------------
# Load BH
df_bh = pd.read_csv(BH_CSV, sep=",")
df_bh["time"] = pd.to_datetime(df_bh["time"], errors="coerce")
df_bh = df_bh.dropna(subset=["time"]).sort_values("time")

df_bh_roll = (
    df_bh.set_index("time")
         .select_dtypes(include=[np.number])
         .rolling("12H")
         .mean()
)

# ---------------------------------------------------------------------
# Load GST
df_gst = pd.read_csv(GST_CSV, sep=",")
df_gst["time"] = pd.to_datetime(df_gst["time"], errors="coerce")
df_gst = df_gst.dropna(subset=["time"]).sort_values("time")

df_gst_roll = (
    df_gst.set_index("time")
          .select_dtypes(include=[np.number])
          .rolling("3H")
          .mean()
)

# ---------------------------------------------------------------------
# Load meteo
df_met = pd.read_csv(METEO_CSV, sep=",", engine="python")
df_met["TimeStamp"] = pd.to_datetime(df_met["TimeStamp"], errors="coerce")
df_met = df_met.dropna(subset=["TimeStamp"]).sort_values("TimeStamp")

for col in ["TA", "HS1", "RH", "VWND1"]:
    if col in df_met.columns:
        df_met[col] = pd.to_numeric(df_met[col], errors="coerce")

df_met_roll = (
    df_met.set_index("TimeStamp")
          .select_dtypes(include=[np.number])
          .rolling("12H")
          .mean()
)

# ---------------------------------------------------------------------
# SEASON SHADING
def add_season_shading(ax):
    seasons = [
        ("Winter",  pd.Timestamp("2022-01-01"), pd.Timestamp("2022-02-28"), "#d0e7ff"),
        ("Spring",  pd.Timestamp("2022-03-01"), pd.Timestamp("2022-05-31"), "#dfffd6"),
        ("Summer",  pd.Timestamp("2022-06-01"), pd.Timestamp("2022-08-31"), "#fff7cc"),
        ("Autumn",  pd.Timestamp("2022-09-01"), pd.Timestamp("2022-11-30"), "#ffe0cc"),
        ("Winter2", pd.Timestamp("2022-12-01"), pd.Timestamp("2022-12-31"), "#d0e7ff"),
    ]
    for _, start, end, col in seasons:
        ax.axvspan(start, end, color=col, alpha=0.30, zorder=0)

# ---------------------------------------------------------------------
# Function to create one seasonal plot
def plot_quarter(start_date, end_date, out_name):

    df_pix_q = df_pix_roll[(df_pix_roll.index >= start_date) & (df_pix_roll.index <= end_date)]
    df_bh_q  = df_bh_roll[(df_bh_roll.index >= start_date) & (df_bh_roll.index <= end_date)]
    df_gst_q = df_gst_roll[(df_gst_roll.index >= start_date) & (df_gst_roll.index <= end_date)]
    df_met_q = df_met_roll[(df_met_roll.index >= start_date) & (df_met_roll.index <= end_date)]

    # width x2 instead of x3
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(24, 16))

    for ax in axes:
        add_season_shading(ax)

    # -------------------------------------------------------------
    # Helper: enable major + minor vertical gridlines
    def apply_vertical_grid(ax):
        #ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.5)
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.35)

    # -------------------------------------------------------------
    # 1) TIR corrected / LW_temp / TSS
    ax = axes[0]
    if "Meteo_Station_corr" in df_pix_q.columns:
        ax.plot(df_pix_q.index, df_pix_q["Meteo_Station_corr"], label="Corrected Temp", color="red", lw=1.5)
    if "LW_temp" in df_pix_q.columns:
        ax.plot(df_pix_q.index, df_pix_q["LW_temp"], label="LW Temp", color="orange", lw=lw)
    if "TSS" in df_pix_q.columns:
        ax.plot(df_pix_q.index, df_pix_q["TSS"], label="Surface Temp", color="blue", lw=lw)

    ax.set_ylabel("Temp (°C)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.7)
    apply_vertical_grid(ax)
    leg = ax.legend(loc="lower left", ncol=3, fontsize=8, frameon=True)
    leg.get_frame().set_alpha(1)      # 0.0 = fully transparent, 1.0 = solid
    # -------------------------------------------------------------
    # 2) Borehole BH_1
    ax = axes[1]
    bh_cols = ["0.25", "1", "3", "5", "10"]
    bh_colors = ["purple", "magenta", "red", "orange", "yellow"]
    for col, col_color in zip(bh_cols, bh_colors):
        if col in df_bh_q.columns:
            ax.plot(df_bh_q.index, df_bh_q[col], label=f"{col} m", lw=lw, color=col_color)

    ax.set_ylabel("Borehole 1 Temp (°C)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.7)
    apply_vertical_grid(ax)
    leg = ax.legend(loc="lower left", ncol=5, fontsize=8, frameon=True)
    leg.get_frame().set_alpha(1)      # 0.0 = fully transparent, 1.0 = solid
    # -------------------------------------------------------------
    # 3) GST_4
    ax = axes[2]
    if "GST_4" in df_gst_q.columns:
        ax.plot(df_gst_q.index, df_gst_q["GST_4"], lw=lw, color="green")

    ax.set_ylabel("GST 4 Temp (°C)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.7)
    apply_vertical_grid(ax)

    # -------------------------------------------------------------
    # 4) Relative humidity
    ax = axes[3]
    if "RH" in df_met_q.columns:
        ax.plot(df_met_q.index, df_met_q["RH"], lw=lw)

    ax.set_ylabel("Relative Humidity (%)")
    apply_vertical_grid(ax)

    # -------------------------------------------------------------
    # 5) Air temperature
    ax = axes[4]
    if "TA" in df_met_q.columns:
        ax.plot(df_met_q.index, df_met_q["TA"], lw=lw)

    ax.set_ylabel("Air Temp (°C)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.7)
    apply_vertical_grid(ax)

    # -------------------------------------------------------------
    # 6) Snow height
    ax = axes[5]
    if "HS1" in df_met_q.columns:
        ax.plot(df_met_q.index, df_met_q["HS1"], lw=lw)

    ax.set_ylabel("Snow Height (cm)")
    apply_vertical_grid(ax)

    # -------------------------------------------------------------
    # 7) Wind
    ax = axes[6]
    if "VWND1" in df_met_q.columns:
        ax.plot(df_met_q.index, df_met_q["VWND1"], lw=lw)

    ax.set_ylabel("Wind Speed (m/s)")
    apply_vertical_grid(ax)

    # -------------------------------------------------------------
    # X-axis tick settings
    axes[-1].set_xlim(start_date, end_date)

    # Monthly major ticks
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())

    # Minor ticks every ~7 days → gives ~4 minor gridlines between months
    axes[-1].xaxis.set_minor_locator(mdates.DayLocator(interval=7))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.autofmt_xdate(rotation=0, ha="center")
    plt.tight_layout()

    out_file = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_file, dpi=200)
    plt.close()

    print("Saved:", out_file)


# ---------------------------------------------------------------------
# Generate the 4 seasonal plots
plot_quarter(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-03-31"), "Q1_Jan_Mar.png")
plot_quarter(pd.Timestamp("2022-04-01"), pd.Timestamp("2022-06-30"), "Q2_Apr_Jun.png")
plot_quarter(pd.Timestamp("2022-07-01"), pd.Timestamp("2022-09-30"), "Q3_Jul_Sep.png")
plot_quarter(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-12-31"), "Q4_Oct_Dec.png")
