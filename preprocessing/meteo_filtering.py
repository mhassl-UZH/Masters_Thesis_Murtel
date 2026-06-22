#Filters the raw Murtèl meteorological station data to the study period, 
# removes unrealistic values and writes a cleaned CSV used by all downsteam analyses. 
# Output: murtel_met_qc.csv
# (optional: raw and filtered comparison plots of all important variables)

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Raw input file
CSV_PATH = os.path.join(DATA_DIR, "CSVs", "murtel_met.csv")

# Time settings
TIME_COL = "TimeStamp"
FROM_DATE = "2021-01-01"

# Toggle interactive plot display (True = show in Python, False = only save)
SHOW_PLOTS = False

# Toggle which plot types to save
SAVE_RAW_PLOTS = False
SAVE_FILTERED_PLOTS = False
SAVE_SIDE_BY_SIDE = True

VARS = [
    "TA", "RH", "VWND1", "DWND1",
    "LWRup", "LWRdown", 
    "SWRup", "SWRdown",
    "HS1",
]

OUT_DIR = os.path.join(DATA_DIR, "plots", "Meteo_quality_control_plots")

# Output folder
# Stores the filtered version of the CSV (same folder as input)
OUT_QC_CSV = os.path.join(
    os.path.dirname(CSV_PATH),
    os.path.splitext(os.path.basename(CSV_PATH))[0] + "_qc.csv"
)

# Labels
LABELS = {
    "TA":        {"title": "Air Temperature",                 "ylabel": "Temperature (°C)",        "units": "°C"},
    "RH":        {"title": "Relative Humidity",               "ylabel": "Relative Humidity (%)",   "units": "%"},
    "VWND1":     {"title": "Wind Speed",                      "ylabel": "Wind Speed (m s$^{-1}$)", "units": "m s-1"},
    "DWND1":     {"title": "Wind Direction",                  "ylabel": "Direction (°)",           "units": "°"},
    "LWRup":     {"title": "Downward Longwave Radiation",     "ylabel": "Downward Longwave Radiation (W m$^{-2}$)", "units": "W m-2"},
    "LWRdown":   {"title": "Upward Longwave Radiation",       "ylabel": "Upward Longwave Radiation (W m$^{-2}$)",   "units": "W m-2"},
    "SWRup":     {"title": "Downward Shortwave Radiation",    "ylabel": "Downward Shortwave Radiation (W m$^{-2}$)","units": "W m-2"},
    "SWRdown":   {"title": "Upward Shortwave Radiation",      "ylabel": "Upward Shortwave Radiation (W m$^{-2}$)",  "units": "W m-2"},
    "HS1":       {"title": "Snow Depth",                      "ylabel": "Snow Depth (cm)",         "units": "cm"}
}
DEFAULT_LABEL = {"title": None, "ylabel": None, "units": ""}

# Plot Colors
VAR_COLORS = {
    "TA":        "#57b5e9",
    "RH":        "#0172b2",
    "VWND1":     "#cc79bd",
    "DWND1":     "#949495",
    "LWRup":     "#d55f01",
    "LWRdown":   "#d55f01",
    "SWRup":     "#ece033",
    "SWRdown":   "#ece033",
    "HS1":       "black"
}

# Filtering Thresholds
THRESHOLDS = {
    "TA":        (-25, 30),        # °C
    "RH":        (0, 100),          # %
    "VWND1":     (0, 60),           # m s-1
    "DWND1":     (0, 360),          # degrees
    "LWRup":     (100, 600),        # W m-2
    "LWRdown":   (100, 600),        # W m-2
    "SWRup":     (0, 1400),         # W m-2
    "SWRdown":   (0, 1400),         # W m-2
    "HS1":       (0.01, 500),       # cm
}

# Load raw File
raw_df = pd.read_csv(CSV_PATH, sep=",", low_memory=False)

raw_df[TIME_COL] = pd.to_datetime(raw_df[TIME_COL], errors="coerce")
raw_df = raw_df.dropna(subset=[TIME_COL])
raw_df = raw_df[raw_df[TIME_COL] >= pd.to_datetime(FROM_DATE)].copy()
raw_df = raw_df.sort_values(TIME_COL)

for c in VARS:
    if c in raw_df.columns:
        raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

print(f"File: {CSV_PATH}")
print(f"Rows (>= {FROM_DATE}): {len(raw_df):,}")

# Make a filtered copy
df = raw_df.copy()

# Apply thresholds based on visual inspectation
qc_report = []
for var, (vmin, vmax) in THRESHOLDS.items():
    if var not in df.columns:
        qc_report.append((var, vmin, vmax, "MISSING", 0))
        continue

    before_nan = int(df[var].isna().sum())
    df.loc[(df[var] < vmin) | (df[var] > vmax), var] = pd.NA
    after_nan = int(df[var].isna().sum())

    qc_report.append((var, vmin, vmax, "OK", after_nan - before_nan))


print("\nQC thresholding summary (values set to NaN):")
print("Variable      | Min      | Max      | Status  | Newly masked")
print("------------------------------------------------------------")
for var, vmin, vmax, status, added in qc_report:
    print(f"{var:<13} {str(vmin):>8}   {str(vmax):>8}   {status:<7} {added:>12}")

# Save filtered file
df_out = df.copy()
df_out[TIME_COL] = df_out[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
df_out.to_csv(OUT_QC_CSV, index=False)
print(f"\nSaved QC CSV to: {OUT_QC_CSV}")

# Print stats after QC
rows = []
for c in VARS:
    if c not in df.columns:
        rows.append((c, "MISSING", "", "", "", "", "", ""))
        continue

    s = df[c]
    nan_pct = s.isna().mean() * 100.0
    s_valid = s.dropna()

    if s_valid.empty:
        rows.append((c, "OK", f"{nan_pct:.2f}", "NaN", "NaN", "NaN", "NaN", "NaN"))
        continue

    rows.append((
        c,
        "OK",
        f"{nan_pct:.2f}",
        f"{s_valid.min():.6g}",
        f"{s_valid.max():.6g}",
        f"{s_valid.mean():.6g}",
        f"{s_valid.median():.6g}",
        f"{s_valid.std(ddof=1):.6g}",
    ))

colnames = ["Variable", "Status", "NaN_%", "Min", "Max", "Mean", "Median", "StdDev"]
widths = [max(len(colnames[i]), max(len(str(r[i])) for r in rows)) for i in range(len(colnames))]

def fmt_row(r):
    return " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(r)))

print("\nDescriptive statistics (after QC):")
print(fmt_row(colnames))
print("-+-".join("-" * w for w in widths))
for r in rows:
    print(fmt_row(r))

# Plots
os.makedirs(OUT_DIR, exist_ok=True)

for c in VARS:
    if c not in raw_df.columns or c not in df.columns:
        continue

    lab = LABELS.get(c, DEFAULT_LABEL)
    title = lab["title"] if lab["title"] else c
    ylabel = lab["ylabel"] if lab["ylabel"] else c
    color = VAR_COLORS.get(c, None)

    # 1) Raw only
    if SAVE_RAW_PLOTS:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(raw_df[TIME_COL], raw_df[c], linewidth=0.8, color=color)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(f"{title} (raw)")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(OUT_DIR, f"{c}_raw_from_2021.png")
        fig.savefig(out_png, dpi=200)

        if SHOW_PLOTS:
            plt.show()

        plt.close(fig)

    # 2) filtered only
    if SAVE_FILTERED_PLOTS:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(df[TIME_COL], df[c], linewidth=0.8, color=color)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(f"{title} (filtered)")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(OUT_DIR, f"{c}_filtered_from_2021.png")
        fig.savefig(out_png, dpi=200)

        if SHOW_PLOTS:
            plt.show()

        plt.close(fig)

    # 3) Raw vs Filtered
    if SAVE_SIDE_BY_SIDE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=False)

        # RAW
        axes[0].plot(raw_df[TIME_COL], raw_df[c], linewidth=0.8, color=color)
        axes[0].set_title(f"{title} (raw)")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel(f"{ylabel} (raw)")
        axes[0].grid(True, which="both", alpha=0.3)

        # FILTERED
        axes[1].plot(df[TIME_COL], df[c], linewidth=0.8, color=color)
        axes[1].set_title(f"{title} (filtered)")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel(f"{ylabel} (filtered)")
        axes[1].grid(True, which="both", alpha=0.3)

        fig.tight_layout()

        out_png = os.path.join(OUT_DIR, f"{c}_raw_vs_filtered_from_2021.png")
        fig.savefig(out_png, dpi=200)

        if SHOW_PLOTS:
            plt.show()

        plt.close(fig)

print(f"\nSaved plots to: {OUT_DIR}")