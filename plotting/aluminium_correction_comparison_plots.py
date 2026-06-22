#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
STANDARD_DIR = os.path.join(DATA_DIR, "corrected")
ALU_DIR      = "path" #insert aluminium-corrected folder path

DIST_PATH    = os.path.join(DATA_DIR, "CSVs", "murtel_distance_filled.csv")
CAMTEMP_CSV  = os.path.join(DATA_DIR, "CSVs", "internal_camera_temps.csv")

PIXEL_STD_CSV = os.path.join(DATA_DIR, "corrected")
PIXEL_ALU_CSV = "path" #insert aluminium-corrected csv path

OUT_DIR = os.path.join(DATA_DIR, "plots", "alu_correction_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 336, 252
MAX_POINTS_KDE = 300_000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_REASONABLE_TEMP = -40.0
MAX_REASONABLE_TEMP = 80.0
MAX_REASONABLE_DIFF = 40.0

MIN_PLOT_TEMP = -23.0

FILTER_MODE = "daily"   # "none", "rolling", or "daily"
ROLLING_HOURS = 12
ROLLING_MIN_PERIODS = 3
DAILY_AGG_FUNC = "mean"   # "mean" or "median"

START_DATE = "2021-01-01"
END_DATE   = "2023-08-31"
CAMTEMP_TOL_MIN = 60

BAND_C = 10.0   # uncertainty band width [°C]

YLIM_PAD = 0.05
YLIM_MIN_PAD = 1.0

SHOW_TRANSITION_ZONES = True
TRANSITION_ZONE_COLOR = "grey"
TRANSITION_ZONE_ALPHA = 0.18
Z_TRANSITION = 0

TRANSITION_CW_RANGES = {
    2021: [(24, 29), (37, 45)],
    2022: [(19, 25), (44, 50)],
    2023: [(20, 26)],
}

# Colors
COL_VAL  = "#1b9e77"
COL_RAW  = "#d95f02"
COL_STD  = "#7570b3"
COL_ALU  = "#b41f87"
COL_CAM  = "#2F2F2F"

# Z-order
Z_RAW_BAND  = 1
Z_STD_BAND  = 2
Z_ALU_BAND  = 3
Z_RAW_LINE  = 4
Z_STD_LINE  = 5
Z_ALU_LINE  = 6
Z_VAL       = 7
Z_CAM       = 8

# Line widths
LW_RAW = 0.7
LW_STD = 1.2
LW_ALU = 1.2
LW_VAL = 1.3
LW_CAM = 1.2

# Aluminium plate measurement periods
ALU_PERIODS = [
    (
        pd.Timestamp("2022-07-28 08:03"),
        pd.Timestamp("2022-08-08 19:05"),
        "2022-07-28 to 2022-08-08"
    ),
    (
        pd.Timestamp("2023-05-31 13:06"),
        pd.Timestamp("2023-08-27 01:05"),
        "2023-05-31 to 2023-08-27"
    ),
]

RADIO_RIDGE_COL = "validation_ridge_Corrected_Target_Ambient_LW_Lab"
RADIO_FURROW_COL = "validation_furrow_Corrected_Target_Ambient_LW_Lab"
RADIO_RIDGE_COL_DAILY = RADIO_RIDGE_COL + "_daily"
RADIO_FURROW_COL_DAILY = RADIO_FURROW_COL + "_daily"

RADIO_RIDGE_TIME_COL = "validation_ridge_Time"
RADIO_FURROW_TIME_COL = "validation_furrow_Time"

# Helpers
def to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def parse_timestamp_from_filename(fname):
    base = os.path.basename(fname)
    m = re.search(r"m(\d{12,15})", base)
    if not m:
        return pd.NaT
    token = m.group(1)
    return pd.to_datetime(token[:12], format="%y%m%d%H%M%S", errors="coerce")


def in_any_alu_period(dt):
    if pd.isna(dt):
        return False
    for start, end, _ in ALU_PERIODS:
        if start <= dt <= end:
            return True
    return False


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


def padded_ylim(values, frac=YLIM_PAD, min_pad=YLIM_MIN_PAD):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        pad = max(min_pad, abs(vmin) * frac, 1.0)
    else:
        pad = max(min_pad, (vmax - vmin) * frac)
    return (vmin - pad, vmax + pad)


def series_to_numeric_robust(s):
    if s is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype("string")
    s2 = s2.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})
    s2 = s2.str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^0-9eE\.\-\+]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")


def finite_values(*arrays):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        s = np.asarray(series_to_numeric_robust(pd.Series(arr)), dtype=float)
        s = s[np.isfinite(s)]
        if s.size:
            vals.append(s)
    if not vals:
        return np.array([], dtype=float)
    return np.concatenate(vals)


def print_array_stats(name, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    print("\n" + "=" * 60)
    print(f"STATISTICS: {name}")
    print("=" * 60)

    if arr.size == 0:
        print("No finite values available.")
        return

    print(f"n              : {arr.size}")
    print(f"mean           : {np.mean(arr):.4f} °C")
    print(f"median         : {np.median(arr):.4f} °C")
    if arr.size > 1:
        print(f"std            : {np.std(arr, ddof=1):.4f} °C")
    else:
        print("std            : 0.0000 °C")
    print(f"min            : {np.min(arr):.4f} °C")
    print(f"25th percentile: {np.percentile(arr, 25):.4f} °C")
    print(f"75th percentile: {np.percentile(arr, 75):.4f} °C")
    print(f"max            : {np.max(arr):.4f} °C")

# Grid file helpers
def load_grid_raw(path):
    df = pd.read_csv(path, sep=";", header=None)
    df = df.iloc[1:]
    arr = df.values.astype(float)
    if arr.shape != (H, W):
        raise ValueError(f"Unexpected shape {arr.shape} in {os.path.basename(path)}")
    return arr


def clean_grid(arr):
    arr = arr.copy()
    arr[(arr <= MIN_REASONABLE_TEMP) | (arr > MAX_REASONABLE_TEMP)] = np.nan
    return arr


def unreasonable_mask(arr):
    return np.isfinite(arr) & ((arr <= MIN_REASONABLE_TEMP) | (arr > MAX_REASONABLE_TEMP))

# Distance and camera temperature
def load_distance_grid(path, h=252, w=336):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None, engine="python")
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
            arr = df.values.astype(float)

            if arr.shape == (h, w):
                print(f"Distance grid loaded directly with sep='{sep}' -> {arr.shape}")
                return arr

            if arr.size == h * w:
                arr = arr.reshape(h, w)
                print(f"Distance grid reshaped with sep='{sep}' -> {arr.shape}")
                return arr
        except Exception:
            pass

    raise ValueError(
        f"Could not load distance grid correctly from {path}. "
        f"Expected shape {(h, w)} or total size {h*w}."
    )


dist = load_distance_grid(DIST_PATH, h=H, w=W)
dist_flat = dist.flatten()

cam_df = pd.read_csv(CAMTEMP_CSV, sep=",", low_memory=False)
cam_df["filename"] = cam_df["filename"].astype(str).str.strip()
cam_df["cam_temp"] = pd.to_numeric(cam_df["cam_temp"], errors="coerce")
cam_map = dict(zip(cam_df["filename"], cam_df["cam_temp"]))


def camtemp_for_corr_filename(corr_fname):
    m = re.search(r"(m\d{15})", corr_fname)
    if not m:
        return np.nan
    jpg_key = m.group(1) + ".jpg"
    return float(cam_map.get(jpg_key, np.nan))


def attach_camtemp(df):
    if not os.path.exists(CAMTEMP_CSV):
        df["cam_temp"] = np.nan
        return df

    cam = pd.read_csv(CAMTEMP_CSV)

    if "timestamp" not in cam.columns:
        for c in cam.columns:
            if "time" in str(c).lower() or "date" in str(c).lower():
                cam = cam.rename(columns={c: "timestamp"})
                break

    if "cam_temp" not in cam.columns:
        for c in cam.columns:
            if str(c).lower() in ["camtemp", "cam_temperature", "camera_temp", "camera_temperature", "temp_cam"]:
                cam = cam.rename(columns={c: "cam_temp"})
                break

    cam["timestamp_dt"] = to_dt(cam["timestamp"])
    cam["cam_temp"] = pd.to_numeric(cam["cam_temp"], errors="coerce")
    cam = cam.dropna(subset=["timestamp_dt"]).sort_values("timestamp_dt")

    out = df.sort_values("Time_dt").copy()
    tol = pd.Timedelta(minutes=CAMTEMP_TOL_MIN)

    out = pd.merge_asof(
        out,
        cam[["timestamp_dt", "cam_temp"]].sort_values("timestamp_dt"),
        left_on="Time_dt",
        right_on="timestamp_dt",
        direction="nearest",
        tolerance=tol,
    )
    out.drop(columns=["timestamp_dt"], inplace=True, errors="ignore")
    return out

# KDE scatter
def density_scatter(ax, x, y, cmap="viridis", s=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return None

    arr = np.column_stack([x, y])

    if arr.shape[0] > MAX_POINTS_KDE:
        idx = np.random.choice(arr.shape[0], MAX_POINTS_KDE, replace=False)
        arr = arr[idx]

    x = arr[:, 0]
    y = arr[:, 1]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    sc = ax.scatter(
        x, y,
        c=z,
        s=s,
        cmap=cmap,
        edgecolors="none",
        alpha=1.0
    )
    return sc

# Validation plot helpers
def apply_time_filter(df):
    if START_DATE:
        df = df[df["Time_dt"] >= pd.Timestamp(START_DATE)]
    if END_DATE:
        df = df[df["Time_dt"] <= pd.Timestamp(END_DATE)]
    return df


def rolling_median(series, time_index):
    s = series_to_numeric_robust(series)
    t = pd.DatetimeIndex(time_index)
    s.index = t
    out = s.rolling(f"{ROLLING_HOURS}h", center=True, min_periods=ROLLING_MIN_PERIODS).median()
    return out.reset_index(drop=True)


def ensure_filtered_columns(df, cols, suffix="_filt"):
    if FILTER_MODE == "daily":
        for c in cols:
            if c in df.columns and (c + suffix) not in df.columns:
                df[c + suffix] = series_to_numeric_robust(df[c])
        return df

    for c in cols:
        if c not in df.columns:
            continue
        if FILTER_MODE == "none":
            df[c + suffix] = series_to_numeric_robust(df[c])
        elif FILTER_MODE == "rolling":
            df[c + suffix] = rolling_median(df[c], df["Time_dt"])
        else:
            raise ValueError(f"Unknown FILTER_MODE: {FILTER_MODE}")
    return df


def daily_agg_df(df_y, cols_filt, suffix="_filt"):
    d = df_y.copy().set_index("Time_dt")
    if d.index.size == 0:
        return pd.DataFrame()

    t0 = pd.Timestamp(d.index.min()).floor("D")
    t1 = pd.Timestamp(d.index.max()).floor("D")
    daily_index = pd.date_range(t0, t1, freq="D")
    out = pd.DataFrame(index=daily_index)

    for c in cols_filt:
        if c not in d.columns:
            continue
        s = series_to_numeric_robust(d[c])
        if DAILY_AGG_FUNC == "median":
            out[c + suffix] = s.resample("D").median()
        else:
            out[c + suffix] = s.resample("D").mean()

    return out


def daily_agg_series(time_index, values, how="mean"):
    t = pd.to_datetime(pd.Series(time_index), errors="coerce")
    y = series_to_numeric_robust(pd.Series(values))

    tmp = pd.DataFrame({"t": t, "y": y}).dropna(subset=["t"])
    if tmp.empty:
        return np.array([]), np.array([])

    tmp = tmp.set_index("t").sort_index()

    t0 = pd.Timestamp(tmp.index.min()).floor("D")
    t1 = pd.Timestamp(tmp.index.max()).floor("D")
    daily_index = pd.date_range(t0, t1, freq="D")

    if how == "median":
        g = tmp["y"].resample("D").median()
    else:
        g = tmp["y"].resample("D").mean()

    g = g.reindex(daily_index)
    return g.index.to_numpy(), g.to_numpy()


def pick_best_numeric_col(df, cols):
    best = None
    best_n = -1
    for c in cols:
        if c not in df.columns:
            continue
        x = series_to_numeric_robust(df[c])
        n = int(np.isfinite(x.to_numpy()).sum())
        if n > best_n:
            best_n = n
            best = c
    return best


def validation_cols_for_prefix(df, prefix):
    return [c for c in df.columns if c.startswith(prefix + "_") and c != f"{prefix}_Time"]


def gst_validation_column(df, gst_idx):
    cols = validation_cols_for_prefix(df, "validation_GST")
    if not cols:
        return None

    pat = re.compile(rf"(?:^|_)(?:GST)?[_\s-]*0*{gst_idx}(?:$|_)", re.IGNORECASE)
    candidates = []
    for c in cols:
        suffix = c[len("validation_GST_"):]
        if pat.search("_" + suffix + "_"):
            candidates.append(c)

    if candidates:
        return pick_best_numeric_col(df, candidates)

    return pick_best_numeric_col(df, cols)


def get_validation_time_series(df_y, time_col_fallback="Time_dt", validation_time_col=None):
    if validation_time_col and validation_time_col in df_y.columns:
        t = pd.to_datetime(df_y[validation_time_col], errors="coerce")
        if t.notna().any():
            return t
    return df_y[time_col_fallback]


def _iter_finite_segments(t, y):
    if t.size == 0:
        return
    mask = np.isfinite(y)
    if not mask.any():
        return
    idx = np.where(mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, ends):
        sl = slice(s, e + 1)
        yield t[sl], y[sl]


def plot_line_no_nan_bridge(ax, t, y, **plot_kwargs):
    t = np.asarray(t)
    y = np.asarray(series_to_numeric_robust(pd.Series(y)), dtype=float)
    for t_seg, y_seg in _iter_finite_segments(t, y):
        ax.plot(t_seg, y_seg, **plot_kwargs)


def fill_between_no_nan_bridge(ax, t, y1, y2, **fill_kwargs):
    t = np.asarray(t)
    y1 = np.asarray(series_to_numeric_robust(pd.Series(y1)), dtype=float)
    y2 = np.asarray(series_to_numeric_robust(pd.Series(y2)), dtype=float)
    mask = np.isfinite(y1) & np.isfinite(y2)
    if not mask.any():
        return
    idx = np.where(mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, ends):
        sl = slice(s, e + 1)
        ax.fill_between(t[sl], y1[sl], y2[sl], **fill_kwargs)


def year_xlim(year):
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year+1}-01-01")


def iso_week_start(year, week):
    return pd.Timestamp.fromisocalendar(year, week, 1)


def iso_week_end_exclusive(year, week):
    return pd.Timestamp.fromisocalendar(year, week, 7) + pd.Timedelta(days=1)


def add_transition_zones(ax, year):
    if not SHOW_TRANSITION_ZONES:
        return

    x0, x1 = year_xlim(year)
    ranges = TRANSITION_CW_RANGES.get(year, [])

    for wk_start, wk_end in ranges:
        s = iso_week_start(year, wk_start)
        e = iso_week_end_exclusive(year, wk_end)

        s_clip = max(s, x0)
        e_clip = min(e, x1)

        if s_clip < e_clip:
            ax.axvspan(
                s_clip,
                e_clip,
                color=TRANSITION_ZONE_COLOR,
                alpha=TRANSITION_ZONE_ALPHA,
                zorder=Z_TRANSITION,
                linewidth=0
            )


def apply_month_grid_and_formatter(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="x", labelbottom=True)
    ax.margins(x=0)


def add_common_axis_format(ax, title, ylabel="Temperature [°C]"):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    apply_month_grid_and_formatter(ax)


def set_family_ylim(axes, ylim):
    if ylim is None:
        return
    for ax in axes:
        ax.set_ylim(*ylim)


def set_shared_ylim_pair(ax_left, ax_right, arr_left, arr_right, band=0.0):
    vals = finite_values(arr_left, arr_right)
    if vals.size == 0:
        return

    vmin = float(np.nanmin(vals)) - band
    vmax = float(np.nanmax(vals)) + band

    ylim = padded_ylim(np.array([vmin, vmax]), frac=YLIM_PAD, min_pad=YLIM_MIN_PAD)
    if ylim is not None:
        ax_left.set_ylim(*ylim)
        ax_right.set_ylim(*ylim)


def plot_raw_std_alu_with_band(ax, t, y_alu, y_std, y_raw):
    y_alu = series_to_numeric_robust(pd.Series(y_alu))
    y_std = series_to_numeric_robust(pd.Series(y_std))
    y_raw = series_to_numeric_robust(pd.Series(y_raw))

    fill_between_no_nan_bridge(
        ax, t, y_raw - BAND_C, y_raw + BAND_C,
        color=COL_RAW, alpha=0.14, linewidth=0, zorder=Z_RAW_BAND
    )
    fill_between_no_nan_bridge(
        ax, t, y_std - BAND_C, y_std + BAND_C,
        color=COL_STD, alpha=0.16, linewidth=0, zorder=Z_STD_BAND
    )
    fill_between_no_nan_bridge(
        ax, t, y_alu - BAND_C, y_alu + BAND_C,
        color=COL_ALU, alpha=0.14, linewidth=0, zorder=Z_ALU_BAND
    )

    plot_line_no_nan_bridge(ax, t, y_raw, color=COL_RAW, linewidth=LW_RAW, zorder=Z_RAW_LINE)
    plot_line_no_nan_bridge(ax, t, y_std, color=COL_STD, linewidth=LW_STD, zorder=Z_STD_LINE)
    plot_line_no_nan_bridge(ax, t, y_alu, color=COL_ALU, linewidth=LW_ALU, zorder=Z_ALU_LINE)


def plot_validation_line(ax, t, y_val_raw):
    plot_line_no_nan_bridge(ax, t, y_val_raw, color=COL_VAL, linewidth=LW_VAL, linestyle="-", zorder=Z_VAL)


def plot_validation_dotted_with_markers(ax, t, y_val_raw):
    y = np.asarray(series_to_numeric_robust(pd.Series(y_val_raw)), dtype=float)
    t = np.asarray(t)
    plot_line_no_nan_bridge(ax, t, y, color=COL_VAL, linewidth=1.1, linestyle=":", zorder=Z_VAL)
    m = np.isfinite(y)
    if m.any():
        ax.plot(
            t[m], y[m],
            linestyle="None",
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
            color=COL_VAL,
            zorder=Z_VAL
        )


def plot_radiometer_validation(ax, t_src, y_src):
    if FILTER_MODE == "daily":
        t_d, y_d = daily_agg_series(t_src, y_src, how=DAILY_AGG_FUNC)
        if t_d.size > 0:
            plot_validation_line(ax, t_d, y_d)
            return
        plot_validation_line(ax, np.asarray(t_src), y_src)
        return

    plot_validation_line(ax, np.asarray(t_src), y_src)


def plot_camtemp(ax, t, y_cam, title):
    plot_line_no_nan_bridge(ax, t, y_cam, color=COL_CAM, linewidth=LW_CAM, zorder=Z_CAM)
    add_common_axis_format(ax, title, ylabel="Temperature [°C]")


def add_global_legend(fig):
    h_val = Line2D([0], [0], color=COL_VAL, linewidth=2, label="Validation")
    h_corr = Line2D([0], [0], color=COL_STD, linewidth=2, label="Corrected")
    h_corr_band = Patch(facecolor=COL_STD, alpha=0.16, label="Corrected ±10°C Band")
    h_raw = Line2D([0], [0], color=COL_RAW, linewidth=1, label="Raw")
    h_raw_band = Patch(facecolor=COL_RAW, alpha=0.14, label="Raw ±10°C Band")
    h_alu = Line2D([0], [0], color=COL_ALU, linewidth=2, label="Aluminum Corrected")
    h_alu_band = Patch(facecolor=COL_ALU, alpha=0.14, label="Aluminum Corrected ±10°C Band")
    h_cam = Line2D([0], [0], color=COL_CAM, linewidth=2, label="Camera Temperature")
    h_trans = Patch(facecolor=TRANSITION_ZONE_COLOR, alpha=TRANSITION_ZONE_ALPHA, label="Transition Period")

    handles = [
        h_val,
        h_alu,
        h_corr,
        h_alu_band,
        h_corr_band,
        h_cam,
        h_raw,
        h_trans,
        h_raw_band,
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9
    )


def get_tir_time_and_df(df_y, filter_cols):
    if FILTER_MODE == "daily":
        df_day = daily_agg_df(df_y, filter_cols, suffix="_filt")
        t_tir = df_day.index.to_numpy()
        return t_tir, df_day
    else:
        t_tir = df_y["Time_dt"].to_numpy()
        return t_tir, df_y

# Two-period validation plots
def period_width_ratios():
    durations_days = []
    for start, end, _ in ALU_PERIODS:
        durations_days.append(max((end - start).total_seconds() / 86400.0, 1.0))
    return durations_days


def make_two_period_boreholes_plot(df, filter_cols):
    width_ratios = period_width_ratios()
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(18, 12), sharey="row", sharex=False,
        gridspec_kw={"width_ratios": width_ratios}
    )

    boreholes = [
        ("BH_1", "Borehole 1 (0.25 m)"),
        ("BH_2", "Borehole 2 (0.55 m)"),
        ("BH_3", "Borehole 3 (0.5 m)"),
    ]

    row_values = {0: [[], []], 1: [[], []], 2: [[], []], 3: [[], []]}

    for col_idx, (start, end, _) in enumerate(ALU_PERIODS):
        year = start.year
        df_y = df[(df["Time_dt"] >= start) & (df["Time_dt"] <= end)].copy()
        t_tir, df_tir = get_tir_time_and_df(df_y, filter_cols)
        t_val = df_y["Time_dt"].to_numpy()

        for row_idx, (bh_key, bh_label) in enumerate(boreholes):
            ax = axes[row_idx, col_idx]
            add_transition_zones(ax, year)

            vprefix = f"validation_{bh_key}"
            vcol = pick_best_numeric_col(df_y, validation_cols_for_prefix(df_y, vprefix))

            plot_raw_std_alu_with_band(
                ax, t_tir,
                df_tir.get(f"{bh_key}_corr_alu_filt", np.nan),
                df_tir.get(f"{bh_key}_corr_filt", np.nan),
                df_tir.get(f"{bh_key}_uncorr_filt", np.nan)
            )

            if vcol:
                plot_validation_dotted_with_markers(ax, t_val, df_y[vcol])

            title_txt = "2022" if col_idx == 0 else f"2023 - {bh_label}"
            ax.set_title(title_txt)
            ax.set_ylabel("Temperature [°C]")
            if col_idx == 1:
                ax.set_ylabel("")
            apply_month_grid_and_formatter(ax)
            ax.set_xlim(start, end)

            row_values[row_idx][col_idx] = finite_values(
                df_tir.get(f"{bh_key}_corr_alu_filt", np.nan),
                df_tir.get(f"{bh_key}_corr_filt", np.nan),
                df_tir.get(f"{bh_key}_uncorr_filt", np.nan),
                df_y[vcol] if vcol else np.nan
            )

        ax = axes[3, col_idx]
        add_transition_zones(ax, year)
        title_txt = "2022" if col_idx == 0 else "2023 - Internal Camera Temperature"
        plot_camtemp(ax, t_tir, df_tir.get("cam_temp_filt", np.nan), title_txt)
        if col_idx == 1:
            ax.set_ylabel("")
        ax.set_xlim(start, end)
        row_values[3][col_idx] = finite_values(df_tir.get("cam_temp_filt", np.nan))

    for r in range(3):
        set_shared_ylim_pair(axes[r, 0], axes[r, 1], row_values[r][0], row_values[r][1], band=BAND_C)
    set_shared_ylim_pair(axes[3, 0], axes[3, 1], row_values[3][0], row_values[3][1], band=0.0)

    add_global_legend(fig)
    fig.subplots_adjust(hspace=0.35, wspace=0.04, bottom=0.1)
    out = os.path.join(OUT_DIR, "01_validation_boreholes_two_periods.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_two_period_meteo_radiometers_plot(df, filter_cols):
    width_ratios = period_width_ratios()
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(18, 12), sharey="row", sharex=False,
        gridspec_kw={"width_ratios": width_ratios}
    )

    row_values = {0: [[], []], 1: [[], []], 2: [[], []], 3: [[], []]}

    for col_idx, (start, end, _) in enumerate(ALU_PERIODS):
        year = start.year
        df_y = df[(df["Time_dt"] >= start) & (df["Time_dt"] <= end)].copy()
        t_tir, df_tir = get_tir_time_and_df(df_y, filter_cols)

        ax = axes[0, col_idx]
        add_transition_zones(ax, year)
        plot_raw_std_alu_with_band(
            ax, t_tir,
            df_tir.get("Meteo_Station_corr_alu_filt", np.nan),
            df_tir.get("Meteo_Station_corr_filt", np.nan),
            df_tir.get("Meteo_Station_uncorr_filt", np.nan)
        )
        plot_validation_line(ax, t_tir, df_tir.get("LW_temp_filt", np.nan))
        title_txt = "2022" if col_idx == 0 else "2023 - Surface Temperature Derived from Upwelling Longwave Radiation (Meteorological Station)"
        ax.set_title(title_txt)
        ax.set_ylabel("Temperature [°C]")
        if col_idx == 1:
            ax.set_ylabel("")
        apply_month_grid_and_formatter(ax)
        ax.set_xlim(start, end)
        row_values[0][col_idx] = finite_values(
            df_tir.get("Meteo_Station_corr_alu_filt", np.nan),
            df_tir.get("Meteo_Station_corr_filt", np.nan),
            df_tir.get("Meteo_Station_uncorr_filt", np.nan),
            df_tir.get("LW_temp_filt", np.nan)
        )

        ax = axes[1, col_idx]
        add_transition_zones(ax, year)
        plot_raw_std_alu_with_band(
            ax, t_tir,
            df_tir.get("radio_furrow_corr_alu_filt", np.nan),
            df_tir.get("radio_furrow_corr_filt", np.nan),
            df_tir.get("radio_furrow_uncorr_filt", np.nan)
        )
        t_src = get_validation_time_series(df_y, validation_time_col=RADIO_FURROW_TIME_COL)
        if FILTER_MODE == "daily" and (RADIO_FURROW_COL_DAILY in df_y.columns):
            plot_radiometer_validation(ax, t_src, df_y[RADIO_FURROW_COL_DAILY])
            row_values[1][col_idx] = finite_values(
                df_tir.get("radio_furrow_corr_alu_filt", np.nan),
                df_tir.get("radio_furrow_corr_filt", np.nan),
                df_tir.get("radio_furrow_uncorr_filt", np.nan),
                df_y[RADIO_FURROW_COL_DAILY]
            )
        elif RADIO_FURROW_COL in df_y.columns:
            plot_radiometer_validation(ax, t_src, df_y[RADIO_FURROW_COL])
            row_values[1][col_idx] = finite_values(
                df_tir.get("radio_furrow_corr_alu_filt", np.nan),
                df_tir.get("radio_furrow_corr_filt", np.nan),
                df_tir.get("radio_furrow_uncorr_filt", np.nan),
                df_y[RADIO_FURROW_COL]
            )
        else:
            row_values[1][col_idx] = finite_values(
                df_tir.get("radio_furrow_corr_alu_filt", np.nan),
                df_tir.get("radio_furrow_corr_filt", np.nan),
                df_tir.get("radio_furrow_uncorr_filt", np.nan)
            )
        title_txt = "2022" if col_idx == 0 else "2023 - Radiometer Furrow"
        ax.set_title(title_txt)
        ax.set_ylabel("Temperature [°C]")
        if col_idx == 1:
            ax.set_ylabel("")
        apply_month_grid_and_formatter(ax)
        ax.set_xlim(start, end)

        ax = axes[2, col_idx]
        add_transition_zones(ax, year)
        plot_raw_std_alu_with_band(
            ax, t_tir,
            df_tir.get("radio_ridge_corr_alu_filt", np.nan),
            df_tir.get("radio_ridge_corr_filt", np.nan),
            df_tir.get("radio_ridge_uncorr_filt", np.nan)
        )
        t_src = get_validation_time_series(df_y, validation_time_col=RADIO_RIDGE_TIME_COL)
        if FILTER_MODE == "daily" and (RADIO_RIDGE_COL_DAILY in df_y.columns):
            plot_radiometer_validation(ax, t_src, df_y[RADIO_RIDGE_COL_DAILY])
            row_values[2][col_idx] = finite_values(
                df_tir.get("radio_ridge_corr_alu_filt", np.nan),
                df_tir.get("radio_ridge_corr_filt", np.nan),
                df_tir.get("radio_ridge_uncorr_filt", np.nan),
                df_y[RADIO_RIDGE_COL_DAILY]
            )
        elif RADIO_RIDGE_COL in df_y.columns:
            plot_radiometer_validation(ax, t_src, df_y[RADIO_RIDGE_COL])
            row_values[2][col_idx] = finite_values(
                df_tir.get("radio_ridge_corr_alu_filt", np.nan),
                df_tir.get("radio_ridge_corr_filt", np.nan),
                df_tir.get("radio_ridge_uncorr_filt", np.nan),
                df_y[RADIO_RIDGE_COL]
            )
        else:
            row_values[2][col_idx] = finite_values(
                df_tir.get("radio_ridge_corr_alu_filt", np.nan),
                df_tir.get("radio_ridge_corr_filt", np.nan),
                df_tir.get("radio_ridge_uncorr_filt", np.nan)
            )
        title_txt = "2022" if col_idx == 0 else "2023 - Radiometer Ridge"
        ax.set_title(title_txt)
        ax.set_ylabel("Temperature [°C]")
        if col_idx == 1:
            ax.set_ylabel("")
        apply_month_grid_and_formatter(ax)
        ax.set_xlim(start, end)

        ax = axes[3, col_idx]
        add_transition_zones(ax, year)
        title_txt = "2022" if col_idx == 0 else "2023 - Internal Camera Temperature"
        plot_camtemp(ax, t_tir, df_tir.get("cam_temp_filt", np.nan), title_txt)
        if col_idx == 1:
            ax.set_ylabel("")
        ax.set_xlim(start, end)
        row_values[3][col_idx] = finite_values(df_tir.get("cam_temp_filt", np.nan))

    for r in range(3):
        set_shared_ylim_pair(axes[r, 0], axes[r, 1], row_values[r][0], row_values[r][1], band=BAND_C)
    set_shared_ylim_pair(axes[3, 0], axes[3, 1], row_values[3][0], row_values[3][1], band=0.0)

    add_global_legend(fig)
    fig.subplots_adjust(hspace=0.35, wspace=0.04, bottom=0.1)
    out = os.path.join(OUT_DIR, "01_validation_meteo_radiometers_two_periods.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def make_two_period_gst_plot(df, filter_cols):
    width_ratios = period_width_ratios()
    fig, axes = plt.subplots(
        nrows=6, ncols=2, figsize=(18, 15), sharey="row", sharex=False,
        gridspec_kw={"width_ratios": width_ratios}
    )

    row_values = {0: [[], []], 1: [[], []], 2: [[], []], 3: [[], []], 4: [[], []], 5: [[], []]}

    for col_idx, (start, end, _) in enumerate(ALU_PERIODS):
        year = start.year
        df_y = df[(df["Time_dt"] >= start) & (df["Time_dt"] <= end)].copy()
        t_tir, df_tir = get_tir_time_and_df(df_y, filter_cols)
        t_val = df_y["Time_dt"].to_numpy()

        for i in range(1, 6):
            ax = axes[i - 1, col_idx]
            gst = f"GST_{i}"
            add_transition_zones(ax, year)

            plot_raw_std_alu_with_band(
                ax, t_tir,
                df_tir.get(f"{gst}_corr_alu_filt", np.nan),
                df_tir.get(f"{gst}_corr_filt", np.nan),
                df_tir.get(f"{gst}_uncorr_filt", np.nan)
            )

            vcol = gst_validation_column(df_y, i)
            if vcol:
                plot_validation_dotted_with_markers(ax, t_val, df_y[vcol])

            title_txt = "2022" if col_idx == 0 else f"2023 - {gst.replace('_', ' ')}"
            ax.set_title(title_txt)
            ax.set_ylabel("Temperature [°C]")
            if col_idx == 1:
                ax.set_ylabel("")
            apply_month_grid_and_formatter(ax)
            ax.set_xlim(start, end)

            row_values[i - 1][col_idx] = finite_values(
                df_tir.get(f"{gst}_corr_alu_filt", np.nan),
                df_tir.get(f"{gst}_corr_filt", np.nan),
                df_tir.get(f"{gst}_uncorr_filt", np.nan),
                df_y[vcol] if vcol else np.nan
            )

        ax = axes[5, col_idx]
        add_transition_zones(ax, year)
        title_txt = "2022" if col_idx == 0 else "2023 - Internal Camera Temperature"
        plot_camtemp(ax, t_tir, df_tir.get("cam_temp_filt", np.nan), title_txt)
        if col_idx == 1:
            ax.set_ylabel("")
        ax.set_xlim(start, end)
        row_values[5][col_idx] = finite_values(df_tir.get("cam_temp_filt", np.nan))

    for r in range(5):
        set_shared_ylim_pair(axes[r, 0], axes[r, 1], row_values[r][0], row_values[r][1], band=BAND_C)
    set_shared_ylim_pair(axes[5, 0], axes[5, 1], row_values[5][0], row_values[5][1], band=0.0)

    add_global_legend(fig)
    fig.subplots_adjust(hspace=0.35, wspace=0.04, bottom=0.1)
    out = os.path.join(OUT_DIR, "01_validation_gst_two_periods.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

# Part 1: validation plots
print("\n=== PART 1: validation plots ===")

df_std = pd.read_csv(PIXEL_STD_CSV, sep=";", low_memory=False)
df_std["Time_dt"] = to_dt(df_std.get("Time", np.nan))
df_std = df_std.dropna(subset=["Time_dt"]).sort_values("Time_dt").reset_index(drop=True)

df_alu = pd.read_csv(PIXEL_ALU_CSV, sep=";", low_memory=False)
df_alu["Time_dt"] = to_dt(df_alu.get("Time", np.nan))
df_alu = df_alu.dropna(subset=["Time_dt"]).sort_values("Time_dt").reset_index(drop=True)

alu_corr_cols = [c for c in df_alu.columns if c.endswith("_corr")]
keep_alu_cols = ["Time_dt"] + alu_corr_cols
df_alu = df_alu[keep_alu_cols].copy()

rename_alu = {c: c + "_alu" for c in alu_corr_cols}
df_alu = df_alu.rename(columns=rename_alu)

df = pd.merge_asof(
    df_std.sort_values("Time_dt"),
    df_alu.sort_values("Time_dt"),
    on="Time_dt",
    direction="nearest",
    tolerance=pd.Timedelta("60min")
)

df = apply_time_filter(df)
df = attach_camtemp(df)

tir_filter_cols = [
    "cam_temp",
    "LW_temp",
    "Meteo_Station_corr", "Meteo_Station_uncorr", "Meteo_Station_corr_alu",
    "radio_furrow_corr", "radio_furrow_uncorr", "radio_furrow_corr_alu",
    "radio_ridge_corr", "radio_ridge_uncorr", "radio_ridge_corr_alu",
    "BH_1_corr", "BH_1_uncorr", "BH_1_corr_alu",
    "BH_2_corr", "BH_2_uncorr", "BH_2_corr_alu",
    "BH_3_corr", "BH_3_uncorr", "BH_3_corr_alu",
    "GST_1_corr", "GST_1_uncorr", "GST_1_corr_alu",
    "GST_2_corr", "GST_2_uncorr", "GST_2_corr_alu",
    "GST_3_corr", "GST_3_uncorr", "GST_3_corr_alu",
    "GST_4_corr", "GST_4_uncorr", "GST_4_corr_alu",
    "GST_5_corr", "GST_5_uncorr", "GST_5_corr_alu",
]

for c in tir_filter_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c != "cam_temp":
            df.loc[df[c] <= MIN_PLOT_TEMP, c] = np.nan

df = ensure_filtered_columns(df, tir_filter_cols, suffix="_filt")

saved_validation = []
saved_validation.append(make_two_period_boreholes_plot(df, tir_filter_cols))
saved_validation.append(make_two_period_meteo_radiometers_plot(df, tir_filter_cols))
saved_validation.append(make_two_period_gst_plot(df, tir_filter_cols))

print("Saved validation files:")
for p in saved_validation:
    print(" ", p)

# Match grid files for pixel scatter plots
std_files = sorted(glob.glob(os.path.join(STANDARD_DIR, "*_corr.csv")))
alu_files = sorted(glob.glob(os.path.join(ALU_DIR, "*_corr.csv")))

std_by_name = {os.path.basename(f): f for f in std_files}
alu_by_name = {os.path.basename(f): f for f in alu_files}

common_names = sorted(set(std_by_name.keys()).intersection(set(alu_by_name.keys())))
print(f"\nFound {len(common_names)} common corrected files")

matched = []
for name in common_names:
    dt = parse_timestamp_from_filename(name)
    if not in_any_alu_period(dt):
        continue
    matched.append((dt, std_by_name[name], alu_by_name[name], name))

matched = sorted(matched, key=lambda x: x[0])
print(f"{len(matched)} common files fall within aluminium periods")

if len(matched) == 0:
    raise RuntimeError("No matching standard/alu corrected files found in aluminium periods.")

# Part 2: diagnostics and pixel arrays
print("\n=== PART 2: diagnostics + scatter inputs ===")

records_pixel = []
bad_records = []

for i, (dt, std_path, alu_path, name) in enumerate(matched, start=1):
    try:
        std_grid_raw = load_grid_raw(std_path)
        alu_grid_raw = load_grid_raw(alu_path)
    except Exception as e:
        print(f"[SKIP] {name}: {e}")
        continue

    bad_std_mask = unreasonable_mask(std_grid_raw)
    bad_alu_mask = unreasonable_mask(alu_grid_raw)

    n_bad_std = int(np.sum(bad_std_mask))
    n_bad_alu = int(np.sum(bad_alu_mask))
    n_bad_total = n_bad_std + n_bad_alu

    if n_bad_total > 0:
        bad_records.append({
            "datetime": dt,
            "filename": name,
            "n_bad_standard": n_bad_std,
            "n_bad_alu": n_bad_alu,
            "n_bad_total": n_bad_total,
            "standard_min": float(np.nanmin(std_grid_raw)),
            "standard_max": float(np.nanmax(std_grid_raw)),
            "alu_min": float(np.nanmin(alu_grid_raw)),
            "alu_max": float(np.nanmax(alu_grid_raw)),
        })

    std_grid = clean_grid(std_grid_raw)
    alu_grid = clean_grid(alu_grid_raw)

    std_flat = std_grid.flatten()
    alu_flat = alu_grid.flatten()
    diff_flat = alu_flat - std_flat
    diff_flat[np.abs(diff_flat) > MAX_REASONABLE_DIFF] = np.nan

    cam_temp = camtemp_for_corr_filename(name)
    if np.isfinite(cam_temp) and ((cam_temp <= MIN_REASONABLE_TEMP) or (cam_temp > MAX_REASONABLE_TEMP)):
        cam_temp = np.nan
    cam_flat = np.full_like(std_flat, cam_temp, dtype=float)

    valid = (
        np.isfinite(std_flat) &
        np.isfinite(alu_flat) &
        np.isfinite(diff_flat) &
        np.isfinite(dist_flat) &
        (std_flat > MIN_PLOT_TEMP) &
        (alu_flat > MIN_PLOT_TEMP)
    )

    if np.any(valid):
        records_pixel.append(
            np.column_stack([
                alu_flat[valid],   # 0
                std_flat[valid],   # 1
                dist_flat[valid],  # 2
                diff_flat[valid],  # 3
                cam_flat[valid],   # 4
            ])
        )

    if i % 200 == 0:
        print(f"[PROGRESS] {i}/{len(matched)} files processed")

if len(records_pixel) == 0:
    raise RuntimeError("No valid pixel data collected after threshold filtering.")

arr_all = np.vstack(records_pixel)

# Summary statistics
# arr_all columns: 0=alu, 1=std, 2=dist, 3=alu-std, 4=cam_temp

diff_corr_minus_alu = arr_all[:, 1] - arr_all[:, 0]
diff_alu_minus_corr = arr_all[:, 0] - arr_all[:, 1]

print_array_stats("Corrected - Aluminium-Corrected", diff_corr_minus_alu)
print_array_stats("Aluminium-Corrected - Corrected", diff_alu_minus_corr)

if bad_records:
    df_bad = pd.DataFrame(bad_records).sort_values(
        ["n_bad_total", "n_bad_standard", "n_bad_alu", "datetime"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    bad_csv = os.path.join(OUT_DIR, "02_ranked_unreasonable_acquisitions.csv")
    df_bad.to_csv(bad_csv, sep=";", index=False)

    print("\n============================================================")
    print("ACQUISITIONS WITH UNREASONABLE VALUES (ranked by total count)")
    print(f"Unreasonable if value <= {MIN_REASONABLE_TEMP:.1f} °C or > {MAX_REASONABLE_TEMP:.1f} °C")
    print("============================================================")
    for _, row in df_bad.iterrows():
        print(
            f"{row['datetime']} | total={int(row['n_bad_total'])} | "
            f"standard={int(row['n_bad_standard'])} | alu={int(row['n_bad_alu'])} | "
            f"{row['filename']}"
        )
    print(f"\nSaved ranked unreasonable acquisitions: {bad_csv}")
else:
    print("\nNo unreasonable acquisition values found.")

# Part 3: corrected vs aluminium-corrected scatter
print("\n=== PART 3: T_standard vs T_alu ===")

x1 = arr_all[:, 0]
y1 = arr_all[:, 1]

lims1 = padded_limits(np.concatenate([x1, y1]), frac=0.03)

fig, ax = plt.subplots(figsize=(9, 7))
sc = density_scatter(ax, x1, y1, cmap="viridis", s=3)

if lims1 is not None:
    ax.set_xlim(*lims1)
    ax.set_ylim(*lims1)
    ax.plot(lims1, lims1, "k--", lw=1, alpha=0.75, label="1:1 line")

ax.set_xlabel("Aluminium-Corrected Temperature [°C]")
ax.set_ylabel("Corrected Temperature [°C]")
ax.set_title("Corrected vs Aluminium-Corrected Temperatures")
ax.set_axisbelow(True)
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
ax.minorticks_on()
ax.set_aspect("equal", adjustable="box")
ax.legend()

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Point Density")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_density_Tstandard_vs_Talu.png"), dpi=200)
plt.close(fig)

print("Saved part 3 plot")

# Part 4: difference vs distance
print("\n=== PART 4: difference vs distance ===")

x2 = arr_all[:, 2]
y2 = arr_all[:, 3]

xlim2 = padded_limits(x2, frac=0.0)
ylim2 = padded_limits(y2, frac=0.03)

fig, ax = plt.subplots(figsize=(10, 6))
sc = density_scatter(ax, x2, y2, cmap="viridis", s=3)

if xlim2 is not None:
    ax.set_xlim(*xlim2)
if ylim2 is not None:
    ax.set_ylim(*ylim2)

ax.set_xlabel("Distance to camera [m]")
ax.set_ylabel("Aluminium-Corrected - Corrected Temperature [°C]")
ax.set_title("Temperature Difference vs Distance to Camera")
ax.set_axisbelow(True)
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
ax.minorticks_on()

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Point Density")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_density_Diff_vs_Distance.png"), dpi=200)
plt.close(fig)

print("Saved part 4 plot")

# Part 5: difference vs camera temperature
print("\n=== PART 5: difference vs camera temperature ===")

x3 = arr_all[:, 4]
y3 = arr_all[:, 3]

xlim3 = padded_limits(x3, frac=0.03)
ylim3 = padded_limits(y3, frac=0.03)

fig, ax = plt.subplots(figsize=(10, 6))
sc = density_scatter(ax, x3, y3, cmap="viridis", s=3)

if xlim3 is not None:
    ax.set_xlim(*xlim3)
if ylim3 is not None:
    ax.set_ylim(*ylim3)

ax.set_xlabel("Internal Camera Temperature [°C]")
ax.set_ylabel("Aluminium-Corrected - Corrected Temperature [°C]")
ax.set_title("Temperature Difference vs Internal Camera Temperature")
ax.set_axisbelow(True)
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
ax.minorticks_on()

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Point Density")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_density_Diff_vs_CamTemp.png"), dpi=200)
plt.close(fig)

print("Saved part 5 plot")
print("\nDONE")