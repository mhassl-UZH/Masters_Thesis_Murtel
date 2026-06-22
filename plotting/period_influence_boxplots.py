import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
MODE = "both"   # "temperature_curve", "anomaly_persistence", or "both"

MODE_CONFIG = {
    "temperature_curve": {
        "root":        os.path.join(DATA_DIR, "analysis", "temperature_curve_regression_k2"),
        "csv_pattern": "daily_window_cluster_regression_k2.csv",
        "group_col":   "cluster",
        "out_name":    "period_influence_boxplots_temperature_curve.png",
        "group_keys":  ["Tc0", "Tc1"],
        "group_labels": {"Tc0": "Cluster 1", "Tc1": "Cluster 2"},
        "group_shade": {
            "Tc0": (None,      0.00),
            "Tc1": ("#bbbbbb", 0.22),
        },
    },
    "anomaly_persistence": {
        "root":        os.path.join(DATA_DIR, "analysis", "anomaly_persistence_regression"),
        "csv_pattern": "daily_window_group_regression_qtiles.csv",
        "group_col":   "group",
        "out_name":    "period_influence_boxplots_anomaly_persistence.png",
        "group_keys":  ["Tcold", "Tmid", "Twarm"],
        "group_labels": {"Tcold": "Persistently Cold", "Tmid": "Intermediate", "Twarm": "Persistently Warm"},
        "group_shade": {
            "Tcold": (None,      0.00),
            "Tmid":  ("#bbbbbb", 0.18),
            "Twarm": ("#888888", 0.28),
        },
    },
}

CANON_VARS  = ["TA", "RH", "SWRup", "LWRup", "VWND", "DWND_cos", "DWND_sin"]
LEGEND_VARS = ["TA", "VWND", "RH", "DWND_cos", "SWRup", "DWND_sin", "LWRup"]

VAR_COLORS = {
    "TA":       "#57b5e9",
    "RH":       "#0172b2",
    "VWND":     "#cc79bd",
    "LWRup":    "#d55f01",
    "SWRup":    "#ece033",
    "DWND_sin": "#949495",
    "DWND_cos": "#039e73",
}

VAR_LABELS = {
    "TA":       "Air Temperature",
    "RH":       "Relative Humidity",
    "VWND":     "Wind Speed",
    "LWRup":    "Downwelling Longwave Radiation",
    "SWRup":    "Downwelling Shortwave Radiation",
    "DWND_sin": "Wind Direction (E - W)",
    "DWND_cos": "Wind Direction (N - S)",
}

GROUP_GAP = 0.8

FS_TITLE    = 14
FS_COND_LBL = 11
FS_GRP_LBL  = 10
FS_YTICK    = 10
FS_YLABEL   = 11
FS_LEGEND   = 11

# Analysis periods
PERIODS = [
    {
        "title":       "Transition",
        "condA_label": "Snow-Covered to Snow-Free",
        "condA":       ("Transition", lambda rl: "Spring" in rl),
        "condB_label": "Snow-Free to Snow-Covered",
        "condB":       ("Transition", lambda rl: "Autumn" in rl),
    },
    {
        "title":       "Snow-Free vs. Snow-Covered",
        "condA_label": "Snow-Free",
        "condA":       ("Snow_NoSnow", lambda rl: "NoSnow" in rl),
        "condB_label": "Snow-Covered",
        "condB":       ("Snow_NoSnow", lambda rl: "Snow" in rl and "NoSnow" not in rl),
    },
    {
        "title":       "Precipitation Periods (Snow-Free)",
        "condA_label": "No Precipitation",
        "condA":       ("Rain_NoRain", lambda rl: "NoSnow" in rl and "NoRain" in rl),
        "condB_label": "Precipitation",
        "condB":       ("Rain_NoRain", lambda rl: "NoSnow" in rl and "Rain" in rl and "NoRain" not in rl),
    },
    {
        "title":       "Precipitation Periods (Snow-Covered)",
        "condA_label": "No Precipitation",
        "condA":       ("Rain_NoRain", lambda rl: "Snow" in rl and "NoSnow" not in rl and "NoRain" in rl),
        "condB_label": "Precipitation",
        "condB":       ("Rain_NoRain", lambda rl: "Snow" in rl and "NoSnow" not in rl and "Rain" in rl and "NoRain" not in rl),
    },
    {
        "title":       "Day vs. Night (Snow-Free)",
        "condA_label": "Day",
        "condA":       ("Day_Night", lambda rl: "NoSnow" in rl and rl.lower().endswith("_day")),
        "condB_label": "Night",
        "condB":       ("Day_Night", lambda rl: "NoSnow" in rl and rl.lower().endswith("_night")),
    },
    {
        "title":       "Day vs. Night (Snow-Covered)",
        "condA_label": "Day",
        "condA":       ("Day_Night", lambda rl: "Snow" in rl and "NoSnow" not in rl and rl.lower().endswith("_day")),
        "condB_label": "Night",
        "condB":       ("Day_Night", lambda rl: "Snow" in rl and "NoSnow" not in rl and rl.lower().endswith("_night")),
    },
]

# Data loading
def load_all_regression_data() -> pd.DataFrame:
    pattern = os.path.join(ROOT, "*", "*", cfg["csv_pattern"])
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f"No regression CSVs found under: {ROOT}")
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  [WARN] Cannot read {p}: {e}")
    master = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(master):,} rows from {len(dfs)} files.")
    return master


def compute_influence_pct(master: pd.DataFrame) -> pd.DataFrame:
    gcol = cfg["group_col"]
    drivers = master[master["variable"] != "const"].copy()
    drivers["coef_abs"] = pd.to_numeric(drivers["coef"], errors="coerce").abs()
    xcol = "window_label" if "window_label" in drivers.columns else "target_day"
    group_keys = ["analysis_name", "range_label", gcol, xcol]
    rows = []
    for keys, grp in drivers.groupby(group_keys, sort=False):
        total = grp["coef_abs"].sum()
        if pd.isna(total) or total <= 0:
            continue
        for _, row in grp.iterrows():
            var = str(row["variable"])
            if var not in CANON_VARS:
                continue
            rows.append({
                "analysis_name": keys[0],
                "range_label":   keys[1],
                "group":         keys[2],
                "time_key":      keys[3],
                "variable":      var,
                "influence_pct": 100.0 * row["coef_abs"] / total,
            })
    result = pd.DataFrame(rows)
    print(f"  Computed {len(result):,} influence rows.")
    return result


def compute_r2_data(master: pd.DataFrame) -> pd.DataFrame:
    gcol = cfg["group_col"]
    const_rows = master[master["variable"] == "const"].copy()
    const_rows["R2_num"] = pd.to_numeric(const_rows["R2"], errors="coerce")
    xcol = "window_label" if "window_label" in const_rows.columns else "target_day"
    r2_df = (
        const_rows
        .rename(columns={xcol: "time_key", gcol: "group"})
        [["analysis_name", "range_label", "group", "time_key", "R2_num"]]
        .dropna(subset=["R2_num"])
        .drop_duplicates(subset=["analysis_name", "range_label", "group", "time_key"])
        .reset_index(drop=True)
    )
    print(f"  Extracted {len(r2_df):,} R² values.")
    return r2_df


def get_influence_vals(influence_df, analysis_name, range_filter, group_key):
    mask = (
        (influence_df["analysis_name"] == analysis_name) &
        (influence_df["group"] == group_key) &
        (influence_df["range_label"].apply(range_filter))
    )
    sub = influence_df[mask]
    return {var: sub.loc[sub["variable"] == var, "influence_pct"].dropna().values for var in CANON_VARS}


def get_r2_vals(r2_df, analysis_name, range_filter, group_key):
    mask = (
        (r2_df["analysis_name"] == analysis_name) &
        (r2_df["group"] == group_key) &
        (r2_df["range_label"].apply(range_filter))
    )
    return r2_df.loc[mask, "R2_num"].dropna().values


# Plotting
def draw_period_row(ax, period_def, influence_df, r2_df):
    group_keys   = cfg["group_keys"]
    group_labels = cfg["group_labels"]
    group_shade  = cfg["group_shade"]

    n_vars   = len(CANON_VARS)
    n_box    = n_vars + 1
    n_grps   = len(group_keys)
    n_total  = 2 * n_grps

    starts     = [i * (n_box + GROUP_GAP) for i in range(n_total)]
    total_span = n_total * n_box + (n_total - 1) * GROUP_GAP

    condA_analysis, condA_filter = period_def["condA"]
    condB_analysis, condB_filter = period_def["condB"]

    all_groups = (
        [(gk, condA_analysis, condA_filter) for gk in group_keys] +
        [(gk, condB_analysis, condB_filter) for gk in group_keys]
    )

    # Separator positions
    sep_AB = (starts[n_grps - 1] + n_box - 0.5 + starts[n_grps] - 0.5) / 2.0
    inner_seps = [
        (starts[i] + n_box - 0.5 + starts[i + 1] - 0.5) / 2.0
        for i in range(n_total - 1) if i != n_grps - 1
    ]

    # Shaded backgrounds
    sep_boundaries = sorted(inner_seps + [sep_AB, total_span - 0.2])
    for gi, (gkey, _, _) in enumerate(all_groups):
        color, alpha = group_shade[gkey]
        if color is None:
            continue
        left = -0.8 if gi == 0 else (starts[gi - 1] + n_box - 0.5 + starts[gi] - 0.5) / 2.0
        right = next((s for s in sep_boundaries if s > starts[gi]), total_span - 0.2)
        ax.axvspan(left, right, color=color, alpha=alpha, zorder=0)

    # Influence boxplots
    box_data, positions, colors = [], [], []
    for gi, (gkey, analysis, rfilter) in enumerate(all_groups):
        dist = get_influence_vals(influence_df, analysis, rfilter, gkey)
        for vi, var in enumerate(CANON_VARS):
            vals = dist[var]
            box_data.append(vals if len(vals) > 0 else np.array([np.nan]))
            positions.append(starts[gi] + vi)
            colors.append(VAR_COLORS[var])

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.65, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.45, markeredgecolor="none"),
        boxprops=dict(linewidth=0.9), zorder=2,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.88)
    for fi, fliers in enumerate(bp["fliers"]):
        fliers.set_markerfacecolor(colors[fi])
        fliers.set_markeredgecolor(colors[fi])

    # R² boxplots on secondary axis
    r2_box_data, r2_positions = [], []
    for gi, (gkey, analysis, rfilter) in enumerate(all_groups):
        vals = get_r2_vals(r2_df, analysis, rfilter, gkey)
        r2_box_data.append(vals if len(vals) > 0 else np.array([np.nan]))
        r2_positions.append(starts[gi] + n_vars)

    ax2 = ax.twinx()
    bp_r2 = ax2.boxplot(
        r2_box_data, positions=r2_positions, widths=0.65, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.45,
                        markerfacecolor="black", markeredgecolor="none"),
        boxprops=dict(linewidth=1.2, color="black"), zorder=2,
    )
    for patch in bp_r2["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("R²", fontsize=FS_YLABEL, labelpad=4)
    ax2.yaxis.set_tick_params(labelsize=FS_YTICK)

    # Separator lines
    ax.axvline(sep_AB, color="black", linewidth=1.8, linestyle="--", alpha=0.6, zorder=3)
    for s in inner_seps:
        ax.axvline(s, color="#444444", linewidth=0.9, linestyle=":", alpha=0.5, zorder=3)

    # Group labels
    for gi, (gkey, _, _) in enumerate(all_groups):
        ax.text(
            starts[gi] + (n_box - 1) / 2.0, 1.02,
            group_labels[gkey],
            ha="center", va="bottom", fontsize=FS_GRP_LBL, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # Condition labels
    condA_mid = (starts[0] + starts[n_grps - 1] + n_box - 1) / 2.0
    condB_mid = (starts[n_grps] + starts[n_total - 1] + n_box - 1) / 2.0
    for mid, label in [(condA_mid, period_def["condA_label"]), (condB_mid, period_def["condB_label"])]:
        ax.text(mid, 1.10, label, ha="center", va="bottom",
                fontsize=FS_COND_LBL, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Period title
    ax.text(0.5, 1.22, period_def["title"], ha="center", va="bottom",
            fontsize=FS_TITLE, fontweight="bold", transform=ax.transAxes)

    ax.set_xticks(positions + r2_positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.8, total_span - 0.2)
    ax.set_ylim(-1, 100)
    ax.set_ylabel("Rel. Influence [%]", fontsize=FS_YLABEL, labelpad=4)
    ax.yaxis.set_tick_params(labelsize=FS_YTICK)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)


# Main
def run_mode(mode):
    global cfg, ROOT, OUT_PATH
    cfg = MODE_CONFIG[mode]
    ROOT = cfg["root"]
    PLOTS_DIR = os.path.join(DATA_DIR, "plots", "period_influence_boxplots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    OUT_PATH = os.path.join(PLOTS_DIR, cfg["out_name"])

    print(f"Mode: {mode}")

    print("Loading regression CSVs...")
    master_df = load_all_regression_data()

    print("Computing influence percentages...")
    influence_df = compute_influence_pct(master_df)

    print("Extracting R² values...")
    r2_df = compute_r2_data(master_df)

    n_periods = len(PERIODS)
    fig_w = 24 if mode == "temperature_curve" else 28
    fig, axes = plt.subplots(n_periods, 1, figsize=(fig_w, 2.8 * n_periods), squeeze=True)
    fig.subplots_adjust(hspace=0.55, top=0.99, bottom=0.05)

    for ax, period_def in zip(axes, PERIODS):
        draw_period_row(ax, period_def, influence_df, r2_df)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=VAR_COLORS[v], ec="black", lw=0.9, alpha=0.88, label=VAR_LABELS[v])
        for v in LEGEND_VARS
    ]
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=1.2, label="R²"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=FS_LEGEND, bbox_to_anchor=(0.5, 0.0), frameon=True)

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    modes = list(MODE_CONFIG.keys()) if MODE == "both" else [MODE]
    for m in modes:
        run_mode(m)
        print()
