import os
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

ROOT_REG     = os.path.join(DATA_DIR, "analysis", "anomaly_persistence_regression")
CLUSTER_ROOT = os.path.join(DATA_DIR, "analysis", "temperature_curve_regression_k2")
ATM_CSV      = os.path.join(DATA_DIR, "CSVs", "pixel_timeseries.csv")

ZNORM_OUT_DIR   = os.path.join(DATA_DIR, "plots", "relative_influence_plots")
CLUSTER_OUT_DIR = os.path.join(CLUSTER_ROOT, "atm_plots")

ZNORM_CSV_NAME   = "daily_window_group_regression_qtiles.csv"
CLUSTER_CSV_NAME = "daily_window_cluster_regression_k2.csv"

ANALYSIS_FAMILIES = [
    "Transition",
    "Day_Night",
    "Rain_NoRain",
    "Snow_NoSnow",
]

FORCE_RERUN_ALL   = True
SKIP_EXISTING_PNG = True

# Variables loaded from the atmospheric CSV
ATM_VARS = ["TA", "RH", "SWRup", "LWRup", "VWND"]

ATM_COLORS = {
    "TA":    "#57b5e9",
    "RH":    "#0172b2",
    "SWRup": "#ece033",
    "LWRup": "#d55f01",
    "VWND":  "#039e73",
}

N_ATM_PANELS = 2   # panel 0: TA+RH, panel 1: SWRup+LWRup+VWND

CANON_VARS = ["TA", "VWND", "RH", "DWND_cos", "SWRup", "DWND_sin", "LWRup"]

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
    "VWND":     "Wind Speed",
    "RH":       "Relative Humidity",
    "DWND_cos": "Wind Direction (N - S)",
    "SWRup":    "Downwelling Shortwave Radiation",
    "DWND_sin": "Wind Direction (E - W)",
    "LWRup":    "Downwelling Longwave Radiation",
}

GROUP_KEYS = ["Tcold", "Tmid", "Twarm", "Tc0", "Tc1"]

GROUP_TITLES = {
    "Tcold": "Persistently Cold: Standardized Relative Influence",
    "Tmid":  "Intermediate: Standardized Relative Influence",
    "Twarm": "Persistently Warm: Standardized Relative Influence",
    "Tc0":   "Cluster 1: Standardized Relative Influence",
    "Tc1":   "Cluster 2: Standardized Relative Influence",
}

GROUP_DISPLAY_LABELS = {
    "Tcold": "Persistently Cold",
    "Tmid":  "Intermediate",
    "Twarm": "Persistently Warm",
    "Tc0":   "Cluster 1",
    "Tc1":   "Cluster 2",
}

RANGE_TITLES = {
    "Transition_2021_Autumn": "Transition: Snow-Free to Snow-Covered",
    "Transition_2021_Spring": "Transition: Snow-Covered to Snow-Free",
    "Transition_2022_Autumn": "Transition: Snow-Free to Snow-Covered",
    "Transition_2022_Spring": "Transition: Snow-Covered to Snow-Free",
    "Transition_2023_Spring": "Transition: Snow-Covered to Snow-Free",
    "Day_Night_2022_NoSnow_day":   "Day: Snow-Free",
    "Day_Night_2022_NoSnow_night": "Night: Snow-Free",
    "Day_Night_2022_Snow_day":     "Day: Snow-Covered",
    "Day_Night_2022_Snow_night":   "Night: Snow-Covered",
    "Snow_NoSnow_2021_NoSnow": "Snow-Free (2021)",
    "Snow_NoSnow_2022_NoSnow": "Snow-Free (2022)",
    "Snow_NoSnow_2022_Snow":   "Snow-Covered",
    "Rain_NoRain_2022_NoSnow_NoRain": "No Precipitation: Snow-Free",
    "Rain_NoRain_2022_NoSnow_Rain":   "Precipitation: Snow-Free",
    "Rain_NoRain_2022_Snow_NoRain":   "No Precipitation: Snow-Covered",
    "Rain_NoRain_2022_Snow_Rain":     "Precipitation: Snow-Covered",
}

GROUP_COLORS = {
    "Tcold": "#2C7BB6",
    "Tmid":  "#d4a017",
    "Twarm": "#D7191C",
    "Tc0":   "#6baed6",
    "Tc1":   "#fb6a4a",
}

VAR_ALIASES = {
    "TA":       ["TA", "Tair", "TAIR", "T_A", "AirTemp"],
    "RH":       ["RH", "RelHum", "RelHumidity", "RHum"],
    "VWND":     ["VWND", "VWND_mean", "VWIND", "WSPD", "WindSpeed"],
    "LWRup":    ["LWRup", "LWR_up", "LWUp", "LW_up"],
    "SWRup":    ["SWRup", "SWR_up", "SWUp", "SW_up"],
    "DWND_sin": ["DWND_sin", "sin(DWND)", "sin_DWND", "sinDWND", "sindwnd"],
    "DWND_cos": ["DWND_cos", "cos(DWND)", "cos_DWND", "cosDWND", "cosdwnd"],
}

# Helpers

def _time_key(v: str):
    s = str(v)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return (0, pd.to_datetime(s).to_pydatetime())
        except Exception:
            return (9, datetime.datetime.max)
    if "_to_" in s:
        try:
            return (1, pd.to_datetime(s.split("_to_")[0]).to_pydatetime())
        except Exception:
            return (9, datetime.datetime.max)
    return (9, datetime.datetime.max)


def get_day_axis_labels(n: int):
    xticks = np.arange(n)
    labels = [str(i + 1) if (i == 0 or (i + 1) % 5 == 0) else "" for i in range(n)]
    return xticks, labels


def get_range_folders(analysis_root: str, csv_name: str = ZNORM_CSV_NAME):
    if not os.path.exists(analysis_root):
        return []
    return [
        name for name in sorted(os.listdir(analysis_root))
        if os.path.isdir(os.path.join(analysis_root, name))
        and os.path.exists(os.path.join(analysis_root, name, csv_name))
    ]


def detect_time_column(df: pd.DataFrame):
    for c in ["target_day", "window_label", "period", "week"]:
        if c in df.columns:
            return c
    return None


def safe_filename_label(s: str):
    return re.sub(r"[\\/:*?\"<>|]", "_", str(s).strip()).replace(" ", "_")


def to_date(s: str):
    try:
        return str(pd.to_datetime(str(s).split("_to_")[0]).date())
    except Exception:
        return None


def load_atm_daily(atm_csv: str) -> pd.DataFrame:
    df = pd.read_csv(atm_csv, sep=";", low_memory=False)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).copy()
    df["date"] = df["Time"].dt.date.astype(str)
    for var in ATM_VARS:
        df[var] = pd.to_numeric(df[var], errors="coerce")
    return df.groupby("date")[ATM_VARS].mean().reset_index()


def build_atm_series(target_days: list, atm_indexed: pd.DataFrame) -> dict:
    day_dates = [to_date(d) for d in target_days]
    return {
        var: np.array(
            [atm_indexed.loc[date, var] if (date and date in atm_indexed.index) else np.nan
             for date in day_dates],
            dtype=float
        )
        for var in ATM_VARS
    }


def normalize_var(s: str):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def resolve_canon_to_actual(df: pd.DataFrame) -> dict:
    available_norm = {normalize_var(v): v for v in df["variable"].dropna().astype(str).unique()}
    mapping = {}
    for canon in CANON_VARS:
        found = None
        for alias in VAR_ALIASES.get(canon, [canon]):
            if normalize_var(alias) in available_norm:
                found = available_norm[normalize_var(alias)]
                break
        mapping[canon] = found
    return mapping


def add_influence_percent(block: pd.DataFrame, value_col: str = "coef") -> pd.DataFrame:
    d = block.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["abscoef"] = d[value_col].abs()
    total = d["abscoef"].sum(skipna=True)
    d["influence_pct"] = 100.0 * d["abscoef"] / total if (pd.notna(total) and total > 0) else np.nan
    return d


def get_r2_series(df: pd.DataFrame, group_key: str, xcol: str) -> pd.DataFrame:
    if "R2" not in df.columns or "group" not in df.columns:
        return pd.DataFrame(columns=[xcol, "R2"])
    d = df[(df["group"] == group_key) & (df["variable"] == "const")].copy()
    if d.empty:
        return pd.DataFrame(columns=[xcol, "R2"])
    d["R2"] = pd.to_numeric(d["R2"], errors="coerce")
    return d[[xcol, "R2"]].dropna(subset=[xcol, "R2"]).drop_duplicates(subset=[xcol]).copy()


def build_influence_data(reg: pd.DataFrame, xcol: str) -> dict:
    # Normalise column name: both approaches use "group" internally
    if "group" not in reg.columns and "cluster" in reg.columns:
        reg = reg.rename(columns={"cluster": "group"})
    if "group" not in reg.columns:
        return {}

    canon2actual_rev = {
        actual: canon
        for canon, actual in resolve_canon_to_actual(reg).items()
        if actual is not None
    }

    present = set(reg["group"].dropna().astype(str).unique())
    active_keys = [gk for gk in GROUP_KEYS if gk in present]

    group_data = {}
    for group_key in active_keys:
        dfc = reg[reg["group"] == group_key].copy()
        if dfc.empty:
            continue
        pct_rows = [add_influence_percent(block) for _, block in dfc.groupby(xcol)]
        if not pct_rows:
            continue
        dfc_pct = pd.concat(pct_rows, ignore_index=True)
        dfc_pct["canon"] = dfc_pct["variable"].map(canon2actual_rev)
        dfc_pct = dfc_pct[dfc_pct["canon"].isin(CANON_VARS)].copy()
        if dfc_pct.empty:
            continue
        dfc_pct[xcol] = dfc_pct[xcol].astype(str)
        r2 = get_r2_series(reg, group_key, xcol)
        if not r2.empty:
            r2[xcol] = r2[xcol].astype(str)
        group_data[group_key] = {"dfc_pct": dfc_pct, "r2": r2}
    return group_data


def build_legend_handles(dfc_pct: pd.DataFrame) -> list:
    present = set(dfc_pct["canon"].dropna())
    return [
        plt.Line2D([0], [0], color=VAR_COLORS[v], lw=2, label=VAR_LABELS[v])
        for v in CANON_VARS if v in present
    ]


# Atmospheric panel drawing

def _grid(ax):
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()


def _draw_atm_panels(axes_atm, atm_series: dict, xticks):
    # Panel 0: TA (left, °C) + RH (right, %)
    ax0 = axes_atm[0]
    ax0.plot(xticks, atm_series["TA"], color=ATM_COLORS["TA"], linewidth=2)
    ax0.set_ylabel("Air Temperature [°C]")
    ax0.tick_params(axis="y", labelsize=7)
    ax0r = ax0.twinx()
    ax0r.plot(xticks, atm_series["RH"], color=ATM_COLORS["RH"], linewidth=2)
    ax0r.set_ylabel("Relative Humidity [%]")
    ax0r.tick_params(axis="y", labelsize=7)
    _grid(ax0)

    # Panel 1: SWRup + LWRup (left, W/m²) + VWND (right, m/s)
    ax1 = axes_atm[1]
    ax1.plot(xticks, atm_series["SWRup"], color=ATM_COLORS["SWRup"], linewidth=2)
    ax1.plot(xticks, atm_series["LWRup"], color=ATM_COLORS["LWRup"], linewidth=2)
    ax1.set_ylabel("Radiation [W/m²]")
    ax1.tick_params(labelsize=7)
    ax1r = ax1.twinx()
    ax1r.plot(xticks, atm_series["VWND"], color=ATM_COLORS["VWND"], linewidth=2)
    ax1r.set_ylabel("Wind Speed [m/s]")
    ax1r.tick_params(axis="y", labelsize=7)
    _grid(ax1)
    return ax0r, ax1r


# Plot 1: atmospheric variables only

def plot_atm_variables(reg: pd.DataFrame, atm_indexed: pd.DataFrame, out_dir: str, range_label: str):
    out_path = os.path.join(out_dir, f"{safe_filename_label(range_label)}_atm_variables.png")

    if (not FORCE_RERUN_ALL) and SKIP_EXISTING_PNG and os.path.exists(out_path):
        print(f"[SKIP] Already exists: {out_path}")
        return out_path

    xcol = detect_time_column(reg)
    if xcol is None:
        print(f"[SKIP] No time column for {range_label}")
        return None

    target_days = sorted(reg[xcol].dropna().astype(str).unique().tolist(), key=_time_key)
    if not target_days:
        return None

    atm_series = build_atm_series(target_days, atm_indexed)
    dates = [d for d in (to_date(t) for t in target_days) if d]
    title_date = f"{min(dates)} – {max(dates)}" if dates else range_label

    n = len(target_days)
    xticks, xticklabels = get_day_axis_labels(n)

    fig, axes = plt.subplots(N_ATM_PANELS, 1, figsize=(11.25, 3.5 * N_ATM_PANELS), sharex=True)
    _draw_atm_panels(axes, atm_series, xticks)
    axes[0].set_title(f"{title_date}\nAtmospheric Variables")
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=0)
    axes[-1].set_xlabel("Day")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    return out_path


# Plot 2: combined influence + R² + residuals + atmospheric

def plot_combined_with_atm(reg: pd.DataFrame, atm_indexed: pd.DataFrame, out_dir: str,
                           range_label: str, residuals: pd.DataFrame | None = None,
                           suptitle_y: float = 0.985, fixed_total_h: float = None):
    out_path = os.path.join(out_dir, f"{safe_filename_label(range_label)}_combined_std_influence_with_atm.png")

    if (not FORCE_RERUN_ALL) and SKIP_EXISTING_PNG and os.path.exists(out_path):
        print(f"[SKIP] Already exists: {out_path}")
        return out_path

    xcol = detect_time_column(reg)
    if xcol is None:
        print(f"[SKIP] No time column for {range_label}")
        return None

    group_data = build_influence_data(reg, xcol)
    if not group_data:
        print(f"[WARN] No influence data for {range_label}")
        return None

    all_xvals = set()
    for gd in group_data.values():
        all_xvals.update(gd["dfc_pct"][xcol].dropna().tolist())
        if not gd["r2"].empty:
            all_xvals.update(gd["r2"][xcol].dropna().tolist())

    target_days = sorted([str(v) for v in all_xvals], key=_time_key)
    if not target_days:
        return None

    xmap = {v: i for i, v in enumerate(target_days)}
    xticks, xticklabels = get_day_axis_labels(len(target_days))

    for gd in group_data.values():
        gd["dfc_pct"]["xpos"] = gd["dfc_pct"][xcol].map(xmap)
        if not gd["r2"].empty:
            gd["r2"]["xpos"] = gd["r2"][xcol].map(xmap)

    atm_series = build_atm_series(target_days, atm_indexed)
    dates = [d for d in (to_date(t) for t in target_days) if d]
    title_date = f"{min(dates)} – {max(dates)}" if dates else range_label
    period_title = RANGE_TITLES.get(range_label, range_label)

    # Y-axis ceiling: max influence rounded up to nearest 10
    max_pct = 0.0
    for gd in group_data.values():
        m = gd["dfc_pct"]["influence_pct"].dropna().max()
        if pd.notna(m):
            max_pct = max(max_pct, float(m))
    y_max_inf = max(int(np.ceil(max_pct / 10.0)) * 10, 10)

    active_groups = [gk for gk in GROUP_KEYS if gk in group_data]
    n_groups = len(active_groups)
    if n_groups == 0:
        print(f"[WARN] No active groups for {range_label}")
        return None

    # Layout: influence×n_groups | R² | residuals | atm×2
    ratios = [3] * n_groups + [1.5, 2.0] + [1.5] * N_ATM_PANELS
    total_h = fixed_total_h if fixed_total_h is not None else (3.5 * n_groups + 2.0 + 2.5 + 2.0 * N_ATM_PANELS)

    fig, axes = plt.subplots(
        len(ratios), 1,
        figsize=(11.25, total_h),
        sharex=True,
        gridspec_kw={"height_ratios": ratios}
    )
    fig.suptitle(period_title, fontsize=13, y=suptitle_y)

    # Influence panels
    sample_pct = None
    for i, group_key in enumerate(active_groups):
        ax_inf = axes[i]

        if group_key not in group_data:
            ax_inf.text(0.5, 0.5, f"No data for {group_key}", ha="center", va="center",
                        transform=ax_inf.transAxes)
            continue

        dfc_pct = group_data[group_key]["dfc_pct"]
        if sample_pct is None:
            sample_pct = dfc_pct

        for canon in CANON_VARS:
            d = dfc_pct[dfc_pct["canon"] == canon].sort_values("xpos")
            if d.empty:
                continue
            ax_inf.plot(d["xpos"], d["influence_pct"], color=VAR_COLORS[canon], linewidth=2)

        ax_inf.set_ylabel("Relative Influence [%]")
        ax_inf.tick_params(axis="x", labelsize=7)
        title_str = GROUP_TITLES.get(group_key, f"{group_key}: Standardized Relative Influence")
        ax_inf.set_title(f"{title_date}\n\n{title_str}" if i == 0 else title_str)
        ax_inf.set_ylim(0, y_max_inf)
        _grid(ax_inf)

    # Variable legend
    if sample_pct is not None:
        legend_handles = build_legend_handles(sample_pct)
        if legend_handles:
            axes[n_groups - 1].legend(
                handles=legend_handles, loc="upper center",
                bbox_to_anchor=(0.5, -0.12), ncol=4,
                frameon=True, fontsize=8
            )

    # R² panel
    ax_r2 = axes[n_groups]
    for group_key in active_groups:
        if group_key not in group_data:
            continue
        r2 = group_data[group_key]["r2"]
        if r2.empty:
            continue
        r2_plot = r2[r2[xcol].isin(target_days)].sort_values("xpos")
        ax_r2.plot(r2_plot["xpos"], r2_plot["R2"],
                   color=GROUP_COLORS.get(group_key, "gray"), linewidth=2,
                   label=GROUP_DISPLAY_LABELS.get(group_key, group_key))
    ax_r2.set_ylabel("R²")
    ax_r2.tick_params(axis="x", labelsize=7)
    ax_r2.set_title("R²")
    ax_r2.set_ylim(0, 1)
    _grid(ax_r2)

    # Residuals panel
    ax_res = axes[n_groups + 1]
    has_residuals = False
    if residuals is not None and not residuals.empty:
        xcol_res = detect_time_column(residuals)
        if xcol_res and xcol_res in residuals.columns:
            res = residuals.copy()
            res[xcol_res] = res[xcol_res].astype(str)
            for gk in active_groups:
                r = res[(res["group"] == gk) & (res[xcol_res].isin(xmap))].copy()
                if r.empty:
                    continue
                r["xpos"] = r[xcol_res].map(xmap)
                r = r.sort_values("xpos")
                ax_res.plot(r["xpos"], r["residual"], color=GROUP_COLORS.get(gk, "gray"), linewidth=2,
                            label=GROUP_DISPLAY_LABELS.get(gk, gk))
                has_residuals = True
    if not has_residuals:
        ax_res.text(0.5, 0.5, "No residuals available", ha="center", va="center",
                    transform=ax_res.transAxes, fontsize=9)
    ax_res.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_res.set_ylabel("Residual [°C]")
    ax_res.tick_params(axis="x", labelsize=7)
    ax_res.set_title("Residuals (target day)")
    _grid(ax_res)

    # Group legend (shared for R² and residuals panels)
    group_handles = [
        plt.Line2D([0], [0], color=GROUP_COLORS.get(gk, "gray"), lw=2,
                   label=GROUP_DISPLAY_LABELS.get(gk, gk))
        for gk in active_groups
    ]
    ax_res.legend(
        handles=group_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.12), ncol=len(active_groups),
        frameon=True, fontsize=8
    )

    # Atmospheric panels
    axes_atm = axes[n_groups + 2:]
    atm_twins = _draw_atm_panels(axes_atm, atm_series, xticks)
    axes_atm[0].set_title("Atmospheric Variables", pad=4)

    # Atmospheric variable legend
    atm_handles = [
        plt.Line2D([0], [0], color=ATM_COLORS[v], lw=2, label=VAR_LABELS[v])
        for v in ["TA", "RH", "SWRup", "LWRup", "VWND"]
    ]
    axes[-1].legend(
        handles=atm_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.45), ncol=5,
        frameon=True, fontsize=8
    )

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=0)
    for ax in axes:
        ax.tick_params(labelbottom=True)
    axes[-1].set_xlabel("Day")

    # Grey shading for days with low R² or large residuals
    r2_by_group = {}
    for gk in active_groups:
        if gk not in group_data:
            continue
        r2_gk = group_data[gk]["r2"]
        if r2_gk.empty:
            continue
        r2_by_group[gk] = {}
        for _, row in r2_gk[r2_gk[xcol].isin(target_days)].iterrows():
            xp = xmap.get(str(row[xcol]))
            if xp is not None and pd.notna(row["R2"]):
                r2_by_group[gk][xp] = float(row["R2"])

    res_by_group = {}
    if residuals is not None and not residuals.empty:
        xcol_res = detect_time_column(residuals)
        if xcol_res:
            res_tmp = residuals.copy()
            res_tmp[xcol_res] = res_tmp[xcol_res].astype(str)
            for gk in active_groups:
                rows = res_tmp[res_tmp["group"] == gk]
                if rows.empty:
                    continue
                res_by_group[gk] = {}
                for _, row in rows.iterrows():
                    xp = xmap.get(row.get(xcol_res))
                    if xp is not None and pd.notna(row.get("residual")):
                        res_by_group[gk][xp] = float(row["residual"])

    def _shade(ax, xp):
        ax.axvspan(xp - 0.5, xp + 0.5, color="lightgrey", alpha=0.3, zorder=0)

    for xp in range(len(target_days)):
        # Shade influence panel only for the group that has a bad R² or residual
        for i, gk in enumerate(active_groups):
            r2_bad  = r2_by_group.get(gk, {}).get(xp, 1.0) < 0.6
            res_val = res_by_group.get(gk, {}).get(xp)
            res_bad = res_val is not None and abs(res_val) > 2.0
            if r2_bad or res_bad:
                _shade(axes[i], xp)

        # Shade R² panel if any group is below threshold
        for gk in active_groups:
            if r2_by_group.get(gk, {}).get(xp, 1.0) < 0.6:
                _shade(axes[n_groups], xp)
                break

        # Shade residuals panel if any group exceeds threshold
        for gk in active_groups:
            res_val = res_by_group.get(gk, {}).get(xp)
            if res_val is not None and abs(res_val) > 2.0:
                _shade(axes[n_groups + 1], xp)
                break

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    # Compress inter-panel gaps to bring rows closer together
    _gap_squeeze = 0.03          # applied to all non-legend panels
    _squeeze_res_to_atm = 0.015  # extra squeeze between residuals and atmospheric panels

    original_pos = [ax.get_position() for ax in axes]
    atm_twin_pos = [tw.get_position() for tw in atm_twins]

    panels_with_legend_below = {n_groups - 1, n_groups + 1, len(axes) - 1}
    cumulative_at = [0] * len(axes)
    cumulative = 0
    for i in range(len(axes) - 1):
        if i not in panels_with_legend_below:
            cumulative += _gap_squeeze
        cumulative_at[i + 1] = cumulative

    # Extra squeeze for residuals → atmospheric gap
    atm0_idx = n_groups + 2
    for i in range(atm0_idx, len(axes)):
        cumulative_at[i] += _squeeze_res_to_atm

    for i, ax in enumerate(axes):
        if cumulative_at[i] > 0:
            p = original_pos[i]
            axes[i].set_position([p.x0, p.y0 + cumulative_at[i], p.width, p.height])

    for j, twin in enumerate(atm_twins):
        panel_idx = atm0_idx + j
        if cumulative_at[panel_idx] > 0:
            p = atm_twin_pos[j]
            twin.set_position([p.x0, p.y0 + cumulative_at[panel_idx], p.width, p.height])

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")
    return out_path


# Main

print("Loading atmospheric CSV...")
atm_daily = load_atm_daily(ATM_CSV)
atm_indexed = atm_daily.set_index("date")
print(f"  Loaded {len(atm_daily)} daily rows ({atm_daily['date'].iloc[0]} – {atm_daily['date'].iloc[-1]})")

def process_root(root_dir: str, out_dir: str, csv_name: str,
                 res_csv_name: str = "daily_window_group_residuals_qtiles.csv",
                 suptitle_y: float = 0.985, fixed_total_h: float = None):
    os.makedirs(out_dir, exist_ok=True)

    for analysis_name in ANALYSIS_FAMILIES:
        analysis_root = os.path.join(root_dir, analysis_name)

        if not os.path.exists(analysis_root):
            print(f"\n[SKIP] Not found: {analysis_root}")
            continue

        range_labels = get_range_folders(analysis_root, csv_name)
        if not range_labels:
            print(f"\n[SKIP] No range folders in: {analysis_root}")
            continue

        print(f"\n{'#' * 80}")
        print(f"ANALYSIS: {analysis_name}  ({len(range_labels)} ranges)")

        for range_label in range_labels:
            base = os.path.join(analysis_root, range_label)

            reg_file = os.path.join(base, csv_name)
            if not os.path.exists(reg_file):
                print(f"[SKIP] Missing CSV: {reg_file}")
                continue

            try:
                reg = pd.read_csv(reg_file)
            except Exception as e:
                print(f"[SKIP] Could not read {reg_file}: {e}")
                continue

            if reg.empty:
                print(f"[SKIP] Empty CSV: {reg_file}")
                continue

            res_file  = os.path.join(base, res_csv_name)
            residuals = pd.DataFrame()
            if os.path.exists(res_file):
                try:
                    residuals = pd.read_csv(res_file)
                except Exception as e:
                    print(f"  [WARN] Could not load residuals: {e}")

            label = f"{analysis_name}_{range_label}"
            print(f"  {label}")
            plot_combined_with_atm(reg, atm_indexed, out_dir, label, residuals=residuals,
                                   suptitle_y=suptitle_y, fixed_total_h=fixed_total_h)


_znorm_h = 3.5 * 3 + 2.0 + 2.5 + 2.0 * N_ATM_PANELS  # fixed height for 3-group znorm layout

print(f"\n{'=' * 80}")
print("CLUSTER REGRESSION K2")
process_root(CLUSTER_ROOT, CLUSTER_OUT_DIR, CLUSTER_CSV_NAME,
             res_csv_name="daily_window_cluster_residuals_k2.csv",
             suptitle_y=0.98, fixed_total_h=_znorm_h)

print(f"\n{'=' * 80}")
print("ZNORM REGRESSION")
process_root(ROOT_REG, ZNORM_OUT_DIR, ZNORM_CSV_NAME)

print("\nDONE.")
