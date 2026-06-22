import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygam import LinearGAM, s

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

IN_ROOT = os.path.join(DATA_DIR, "analysis", "temperature_curve_regression_k2")
OUT_ROOT = os.path.join(DATA_DIR, "analysis", "temperature_curve_gam_k2")

os.makedirs(OUT_ROOT, exist_ok=True)

ANALYSIS_NAME = "ALL"
MIN_POINTS_PER_WINDOW = 10

FEATURES = ["TA", "RH", "SWRup", "LWRup", "VWND", "DWND_cos", "DWND_sin"]

FEATURE_LABELS = {
    "TA": "Air Temperature [°C]",
    "RH": "Relative Humidity [%]",
    "VWND": "Wind Speed [m/s]",
    "LWRup": "Downwelling Longwave Radiation [W m$^{-2}$]",
    "SWRup": "Downwelling Shortwave Radiation [W m$^{-2}$]",
    "DWND_sin": "Wind Direction (E - W)",
    "DWND_cos": "Wind Direction (N - S)",
}

RESPONSES = ["Tc0", "Tc1"]

CLUSTER_INFO = {
    "Tc0": {
        "cluster_code": 0,
        "cluster_label": "cluster 1",
        "cluster_title": "Cluster 1",
        "cluster_long": "cluster 1",
        "color": "#6baed6",
    },
    "Tc1": {
        "cluster_code": 1,
        "cluster_label": "cluster 2",
        "cluster_title": "Cluster 2",
        "cluster_long": "cluster 2",
        "color": "#fb6a4a",
    },
}

DROPNA_STRICT = True
MAKE_PDP_PLOTS = True
PDP_YLIM_PAD_FRAC = 0.08

# Helpers

def sanitize_for_filename(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "_")
    s = s.replace(":", "-")
    s = s.replace("/", "_")
    s = s.replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s

def pretty_window_label(window_label: str) -> str:
    try:
        start_s, end_s = str(window_label).split("_to_")
        start_dt = pd.to_datetime(start_s)
        end_dt = pd.to_datetime(end_s)
        return f"{start_dt.strftime('%d.%m.%Y')} to {end_dt.strftime('%d.%m.%Y')}"
    except Exception:
        return str(window_label)

def pretty_target_day(target_day: str) -> str:
    try:
        dt = pd.to_datetime(target_day)
        return dt.strftime("%d.%m.%Y")
    except Exception:
        return str(target_day)

def fit_gam(X: np.ndarray, y: np.ndarray) -> LinearGAM:
    gam = LinearGAM(
        s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6)
    ).gridsearch(X, y, progress=False)
    return gam

def safe_get_pseudo_r2_and_explained_dev(stats_dict):
    pseudo_r2_val = stats_dict.get("pseudo_r2", np.nan)
    if isinstance(pseudo_r2_val, dict):
        explained = pseudo_r2_val.get("explained_deviance", np.nan)
        return pseudo_r2_val, explained
    return pseudo_r2_val, np.nan

def build_shared_pdp_ylim(gam_records, feature_names, pad_frac=0.08):
    ymin = np.inf
    ymax = -np.inf

    for rec in gam_records:
        gam = rec["gam"]

        for i, _ in enumerate(feature_names):
            try:
                XX = gam.generate_X_grid(term=i)
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

                local_min = np.nanmin([
                    np.nanmin(pdep),
                    np.nanmin(confi[:, 0]),
                    np.nanmin(confi[:, 1]),
                    0.0
                ])
                local_max = np.nanmax([
                    np.nanmax(pdep),
                    np.nanmax(confi[:, 0]),
                    np.nanmax(confi[:, 1]),
                    0.0
                ])

                ymin = min(ymin, local_min)
                ymax = max(ymax, local_max)
            except Exception:
                continue

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None

    if np.isclose(ymin, ymax):
        span = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * 0.2
        ymin -= span
        ymax += span
    else:
        span = ymax - ymin
        pad = span * pad_frac
        ymin -= pad
        ymax += pad

    return (ymin, ymax)

def plot_pdps_overlay_two_clusters(
    gam_records: list,
    feature_names: list[str],
    out_png: str,
    period_title: str,
    period_date_range: str,
    target_day: str,
    day_num: int,
    shared_ylim=None
):
    order = ["Tc0", "Tc1"]
    rec_map = {rec["cluster"]: rec for rec in gam_records if rec["cluster"] in order}

    if len(rec_map) == 0:
        return

    n_features = len(feature_names)

    fig, axes = plt.subplots(
        n_features, 1,
        figsize=(10, 3.0 * n_features + 2.0),
        squeeze=False
    )
    axes = axes.flatten()

    r2_parts = []
    for c in ["Tc0", "Tc1"]:
        if c in rec_map:
            r2_val = rec_map[c].get("r2", np.nan)
            r2_str = f"{r2_val:.2f}" if np.isfinite(r2_val) else "n/a"
            r2_parts.append(f"{CLUSTER_INFO[c]['cluster_title']}: R²={r2_str}")
    r2_line = " | ".join(r2_parts)
    n_val   = max(r["n"] for r in rec_map.values())

    fig.suptitle(
        f"{period_title}\n"
        f"{period_date_range}\n"
        f"Day {day_num}: {pretty_target_day(target_day)}\n"
        f"{r2_line}\n"
        f"n={n_val}",
        y=0.995,
        fontsize=14
    )

    for idx, feat in enumerate(feature_names):
        ax = axes[idx]

        # Confidence bands (drawn first so lines stay on top)
        for cluster_key in order:
            if cluster_key not in rec_map:
                continue
            rec = rec_map[cluster_key]
            gam = rec["gam"]
            color = CLUSTER_INFO[cluster_key]["color"]
            try:
                XX = gam.generate_X_grid(term=idx)
                _, confi = gam.partial_dependence(term=idx, X=XX, width=0.95)
                ax.fill_between(XX[:, idx], confi[:, 0], confi[:, 1],
                                color=color, alpha=0.12, zorder=1)
            except Exception:
                pass

        for cluster_key in order:
            if cluster_key not in rec_map:
                continue

            rec = rec_map[cluster_key]
            gam = rec["gam"]
            color = CLUSTER_INFO[cluster_key]["color"]

            try:
                XX = gam.generate_X_grid(term=idx)
                pdep, _ = gam.partial_dependence(term=idx, X=XX, width=0.95)

                ax.plot(XX[:, idx], pdep, linewidth=2.0, color=color,
                        label=CLUSTER_INFO[cluster_key]["cluster_title"], zorder=2)
            except Exception as e:
                print(f"  [WARN] PDP failed | {cluster_key} | {feat}: {e}")

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.axhline(0, linewidth=1.2, color="black", linestyle="--", alpha=0.8)

        if shared_ylim is not None:
            ax.set_ylim(shared_ylim)

        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=13)
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=12)
        ax.set_ylabel("Effect", fontsize=12)
        ax.tick_params(labelsize=10)

    handles = [
        plt.Line2D([0], [0], color=CLUSTER_INFO[c]["color"], lw=2,
                   label=CLUSTER_INFO[c]["cluster_title"])
        for c in order if c in rec_map
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                   frameon=True, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def discover_timeseries_files(root: str) -> list[dict]:
    pattern = os.path.join(root, "*", "*", "daily_window_cluster_timeseries_k2.csv")
    ts_files = sorted(glob.glob(pattern))

    out = []
    for ts_path in ts_files:
        range_dir = os.path.dirname(ts_path)
        analysis_dir = os.path.dirname(range_dir)

        range_label = os.path.basename(range_dir)
        analysis_name = os.path.basename(analysis_dir)

        quality_path = os.path.join(range_dir, "daily_window_quality_report_k2.csv")
        reg_path     = os.path.join(range_dir, "daily_window_cluster_regression_k2.csv")

        out.append({
            "analysis_name": analysis_name,
            "range_label": range_label,
            "ts_path": ts_path,
            "quality_path": quality_path,
            "reg_path": reg_path,
        })

    return out

def load_quality_lookup(quality_path: str) -> pd.DataFrame:
    if not os.path.exists(quality_path):
        return pd.DataFrame()

    try:
        q = pd.read_csv(quality_path)
    except Exception as e:
        print(f"[WARN] Could not read quality file: {quality_path} | {e}")
        return pd.DataFrame()

    keep_cols = [
        "analysis_name",
        "range_label",
        "range_title",
        "base_label",
        "phase",
        "rolling_window_days",
        "target_day",
        "window_label",
        "window_number",
        "n_window_imgs",
        "n_kept",
        "skip_read",
        "skip_nanheavy",
        "skip_empty",
        "mask0_pixels",
        "mask1_pixels",
        "label_counts",
    ]
    keep_cols = [c for c in keep_cols if c in q.columns]

    if len(keep_cols) == 0:
        return pd.DataFrame()

    q = q[keep_cols].copy()

    if "target_day" in q.columns:
        q = q.drop_duplicates(subset=["target_day"]).copy()

    return q

# Days to process per range label (1-indexed). Leave list empty to process all days.

DAYS_TO_PROCESS = {
    "2022_NoSnow_day": [1,17,18],
    "2022_NoSnow_night": [],
    "2022_Snow_day": [1,17,18],
    "2022_Snow_night": [3,4,5,6,11,12,13,14,15],
    "2022_NoSnow_NoRain": [],
    "2022_NoSnow_Rain": [6],
    "2022_Snow_NoRain": [14],
    "2022_Snow_Rain": [18],
    "2021_NoSnow": [],
    "2022_NoSnow": [],
    "2022_Snow": [12],
    "2021_Autumn": [3,32,33,34,37,38,39],
    "2021_Spring": [7,8,9,17,18,19,26,27,31,32,33,34,35],
    "2022_Autumn": [7,10,13,21,22,23,24,25,26],
    "2022_Spring": [14,15,16,17,18,19,20,29],
    "2023_Spring": [18,19,20,21,22,23,29,30,39,40,41,42,43],
}

RANGE_TITLES = {
    "2022_NoSnow_day":   "Day: Snow-Free",
    "2022_NoSnow_night": "Night: Snow-Free",
    "2022_Snow_day":     "Day: Snow-Covered",
    "2022_Snow_night":   "Night: Snow-Covered",
    "2022_NoSnow_NoRain": "No Precipitation: Snow-Free",
    "2022_NoSnow_Rain":   "Precipitation: Snow-Free",
    "2022_Snow_NoRain":   "No Precipitation: Snow-Covered",
    "2022_Snow_Rain":     "Precipitation: Snow-Covered",
    "2021_NoSnow":  "Snow-Free",
    "2022_NoSnow":  "Snow-Free",
    "2022_Snow":    "Snow-Covered",
    "2021_Autumn":  "Transition: Snow-Free to Snow-Covered",
    "2021_Spring":  "Transition: Snow-Covered to Snow-Free",
    "2022_Autumn":  "Transition: Snow-Free to Snow-Covered",
    "2022_Spring":  "Transition: Snow-Covered to Snow-Free",
    "2023_Spring":  "Transition: Snow-Covered to Snow-Free",
}

# Discover inputs

datasets = discover_timeseries_files(IN_ROOT)

if ANALYSIS_NAME != "ALL":
    datasets = [d for d in datasets if d["analysis_name"] == ANALYSIS_NAME]

if len(datasets) == 0:
    raise RuntimeError(f"No matching daily_window_cluster_timeseries_k2.csv found in {IN_ROOT}")

print(f"[INFO] Found {len(datasets)} range datasets")
for d in datasets:
    print(f"  - {d['analysis_name']} | {d['range_label']}")

if not DAYS_TO_PROCESS:
    print("\n[TEMPLATE] Copy this into DAYS_TO_PROCESS and fill in day numbers:")
    for d in datasets:
        print(f'    "{d["range_label"]}": [],')
    print()

# Main

summary_log = []

for ds in datasets:
    analysis_name = ds["analysis_name"]
    range_label   = ds["range_label"]
    ts_path       = ds["ts_path"]
    quality_path  = ds["quality_path"]
    reg_path      = ds["reg_path"]

    # Load regression R² lookup
    reg_r2       = {}
    reg_day_rank = {}
    if os.path.exists(reg_path):
        try:
            reg_df = pd.read_csv(reg_path)
            if "group" not in reg_df.columns and "cluster" in reg_df.columns:
                reg_df = reg_df.rename(columns={"cluster": "group"})
            tcol = next((c for c in ["target_day", "window_label", "period", "week"]
                         if c in reg_df.columns), None)
            if tcol:
                reg_tdays = sorted(reg_df[tcol].dropna().astype(str).unique().tolist())
                reg_day_rank = {td: i + 1 for i, td in enumerate(reg_tdays)}
            if tcol and "R2" in reg_df.columns and "group" in reg_df.columns and "variable" in reg_df.columns:
                for ycol in RESPONSES:
                    d = reg_df[(reg_df["group"] == ycol) & (reg_df["variable"] == "const")].copy()
                    if d.empty:
                        continue
                    d["R2"] = pd.to_numeric(d["R2"], errors="coerce")
                    d = d[[tcol, "R2"]].dropna(subset=[tcol, "R2"]).drop_duplicates(subset=[tcol])
                    for _, row in d.iterrows():
                        reg_r2.setdefault(str(row[tcol]), {})[ycol] = float(row["R2"])
        except Exception as e:
            print(f"[WARN] Could not load regression R²: {e}")

    out_range_dir = os.path.join(OUT_ROOT, analysis_name, range_label)
    os.makedirs(out_range_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"[INFO] PROCESSING: {analysis_name} | {range_label}")
    print("=" * 90)

    try:
        df = pd.read_csv(ts_path)
    except Exception as e:
        print(f"[WARN] Failed reading timeseries file: {ts_path} | {e}")
        continue

    if "Time" not in df.columns:
        print(f"[WARN] Missing 'Time' column in {ts_path}")
        continue

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).copy()

    required_cols = ["target_day", "window_label"] + RESPONSES + FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Missing required columns in {ts_path}: {missing_cols}")
        continue

    df["target_day"]   = df["target_day"].astype(str)
    df["window_label"] = df["window_label"].astype(str)

    quality_lookup = load_quality_lookup(quality_path)

    if not quality_lookup.empty and "target_day" in quality_lookup.columns:
        window_meta = quality_lookup.copy()
    else:
        unique_windows = (
            df[["target_day", "window_label"]]
            .drop_duplicates()
            .sort_values(by=["target_day", "window_label"])
            .reset_index(drop=True)
        )
        unique_windows["window_number"] = np.arange(1, len(unique_windows) + 1)
        window_meta = unique_windows

    target_days = sorted(df["target_day"].dropna().unique().tolist())
    period_start_dt   = pd.Timestamp(min(target_days))
    period_date_range = (
        f"{period_start_dt.strftime('%d.%m.%Y')} – "
        f"{pd.Timestamp(max(target_days)).strftime('%d.%m.%Y')}"
    )

    if DAYS_TO_PROCESS and range_label in DAYS_TO_PROCESS:
        wanted_ranks = set(DAYS_TO_PROCESS[range_label])
        keep_days    = {td for td, rank in reg_day_rank.items() if rank in wanted_ranks}
        target_days  = [d for d in target_days if d in keep_days]

    print(f"[INFO] Target days to process: {len(target_days)}")

    period_title = RANGE_TITLES.get(range_label, range_label)

    for target_day in target_days:
        sub = df[df["target_day"] == target_day].copy()
        if len(sub) == 0:
            continue

        meta_row = {}
        qsub = window_meta[window_meta["target_day"] == target_day].copy()
        if len(qsub) > 0:
            meta_row = qsub.iloc[0].to_dict()

        window_label = meta_row.get("window_label", sub["window_label"].iloc[0])
        day_num      = reg_day_rank.get(target_day, (pd.Timestamp(target_day) - period_start_dt).days + 1)

        plot_records = []
        day_stats    = []

        for ycol in RESPONSES:
            dfm = sub.dropna(subset=[ycol] + FEATURES).copy() if DROPNA_STRICT else sub.copy()

            n = len(dfm)
            if n < MIN_POINTS_PER_WINDOW:
                continue

            y = dfm[ycol].astype(float).to_numpy()
            X = dfm[FEATURES].astype(float).to_numpy()

            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
                continue

            if np.nanstd(y) <= 1e-12:
                continue

            try:
                gam = fit_gam(X, y)
            except Exception as e:
                print(f"  [WARN] GAM failed | {range_label} | {target_day} | {ycol}: {e}")
                continue

            yhat   = gam.predict(X)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

            day_stats.append({"cluster": ycol, "n": n, "r2": r2})
            if MAKE_PDP_PLOTS:
                plot_records.append({"gam": gam, "cluster": ycol, "n": n, "r2": r2})

        if day_stats:
            td_reg = reg_r2.get(target_day, {})
            parts = []
            for stat in day_stats:
                title      = CLUSTER_INFO[stat["cluster"]]["cluster_title"]
                gam_r2_str = f"{stat['r2']:.3f}" if np.isfinite(stat["r2"]) else "n/a"
                reg_val    = td_reg.get(stat["cluster"], np.nan)
                reg_str    = f"{reg_val:.3f}" if np.isfinite(reg_val) else "n/a"
                parts.append(f"{title}: GAM R²={gam_r2_str} / Reg R²={reg_str}")
            ns = " / ".join(str(stat["n"]) for stat in day_stats)
            summary_log.append(f"  {period_title} ({period_date_range}): Day {day_num}: {', '.join(parts)}, n={ns}")

        if MAKE_PDP_PLOTS and plot_records:
            shared_ylim = build_shared_pdp_ylim(
                gam_records=plot_records,
                feature_names=FEATURES,
                pad_frac=PDP_YLIM_PAD_FRAC
            )
            fname   = f"PDP__{sanitize_for_filename(range_label)}__Day{day_num:03d}__{sanitize_for_filename(target_day)}.png"
            out_png = os.path.join(out_range_dir, fname)
            try:
                plot_pdps_overlay_two_clusters(
                    gam_records=plot_records,
                    feature_names=FEATURES,
                    out_png=out_png,
                    period_title=period_title,
                    period_date_range=period_date_range,
                    target_day=target_day,
                    day_num=day_num,
                    shared_ylim=shared_ylim
                )
            except Exception as e:
                print(f"  [WARN] PDP plot failed | {range_label} | {target_day}: {e}")

for dirpath, _, _ in os.walk(OUT_ROOT, topdown=False):
    if dirpath != OUT_ROOT and not os.listdir(dirpath):
        os.rmdir(dirpath)
        print(f"[CLEANUP] Removed empty dir: {dirpath}")

print("\n" + "=" * 90)
print("R² SUMMARY")
print("=" * 90)
for line in summary_log:
    print(line)

print("\n" + "=" * 90)
print(f"[OK] Finished. Outputs stored in: {OUT_ROOT}")
print("=" * 90)