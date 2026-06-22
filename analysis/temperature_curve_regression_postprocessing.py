import os
import re
import gc
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

ROOT_REG = os.path.join(DATA_DIR, "analysis", "temperature_curve_regression_k2")

ANALYSIS_FAMILIES = [
    "Transition",
    "Day_Night",
    "Rain_NoRain",
    "Snow_NoSnow",
]

FORCE_RERUN_ALL   = True
SKIP_EXISTING_PNG = True

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

GROUP_KEYS = ["Tc0", "Tc1"]

VAR_LABELS = {
    "TA":       "Air Temperature",
    "VWND":     "Wind Speed",
    "RH":       "Relative Humidity",
    "DWND_cos": "Wind Direction (N - S)",
    "SWRup":    "Downwelling Shortwave Radiation",
    "DWND_sin": "Wind Direction (E - W)",
    "LWRup":    "Downwelling Longwave Radiation",
}

GROUP_COLORS = {
    "Tc0": "#6baed6",
    "Tc1": "#fb6a4a",
}

GROUP_TITLES = {
    "Tc0": "Cluster 0: Standardized Relative Influence",
    "Tc1": "Cluster 1: Standardized Relative Influence",
}

# Helpers

def normalize_var(s: str):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

VAR_ALIASES = {
    "TA":       ["TA", "Tair", "TAIR", "T_A", "AirTemp"],
    "RH":       ["RH", "RelHum", "RelHumidity", "RHum"],
    "VWND":     ["VWND", "VWND_mean", "VWIND", "WSPD", "WindSpeed"],
    "LWRup":    ["LWRup", "LWR_up", "LWUp", "LW_up"],
    "SWRup":    ["SWRup", "SWR_up", "SWUp", "SW_up"],
    "DWND_sin": ["DWND_sin", "sin(DWND)", "sin_DWND", "sinDWND", "sindwnd"],
    "DWND_cos": ["DWND_cos", "cos(DWND)", "cos_DWND", "cosDWND", "cosdwnd"],
}

def safe_filename_label(s: str):
    return str(s).strip().replace(":", "-").replace("/", "-").replace("\\", "-").replace(" ", "_")

def parse_window_label_start(s: str):
    try:
        return pd.to_datetime(str(s).split("_to_")[0])
    except Exception:
        return pd.NaT

def parse_window_label_end(s: str):
    try:
        return pd.to_datetime(str(s).split("_to_")[1])
    except Exception:
        return pd.NaT

def _time_key(v: str):
    s = str(v)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return (0, pd.to_datetime(s).to_pydatetime())
        except Exception:
            return (9, datetime.datetime.max)
    if "_to_" in s:
        try:
            return (1, parse_window_label_start(s).to_pydatetime())
        except Exception:
            return (9, datetime.datetime.max)
    m = re.match(r"(\d{4})-W(\d{2})$", s)
    if m:
        year, week = int(m.group(1)), int(m.group(2))
        return (2, datetime.datetime(year, 1, 1) + datetime.timedelta(days=(week - 1) * 7))
    return (9, datetime.datetime.max)

def get_date_span_title(range_label: str, xvals: list, xcol: str):
    if not xvals:
        return f"{range_label}: Standardized Relative Influence"
    start_dt = pd.NaT
    end_dt   = pd.NaT
    if xcol == "target_day":
        parsed = pd.to_datetime(pd.Series(xvals), errors="coerce").dropna()
        if len(parsed) > 0:
            start_dt, end_dt = parsed.min(), parsed.max()
    elif xcol == "window_label":
        starts = pd.Series([parse_window_label_start(x) for x in xvals]).dropna()
        ends   = pd.Series([parse_window_label_end(x)   for x in xvals]).dropna()
        if len(starts) > 0: start_dt = starts.min()
        if len(ends)   > 0: end_dt   = ends.max()
    if pd.isna(start_dt) or pd.isna(end_dt):
        return f"{range_label}: Standardized Relative Influence"
    return f"{start_dt.strftime('%d.%m.%Y')} - {end_dt.strftime('%d.%m.%Y')}"

def get_day_axis_labels(xvals: list, xcol: str):
    xpos   = np.arange(len(xvals))
    labels = [str(i + 1) if (i == 0 or (i + 1) % 5 == 0) else "" for i in range(len(xvals))]
    return xpos, labels, "Day"

def get_range_folders(analysis_root: str):
    if not os.path.exists(analysis_root):
        return []
    return [
        name for name in sorted(os.listdir(analysis_root))
        if os.path.isdir(os.path.join(analysis_root, name))
        and os.path.exists(os.path.join(analysis_root, name, "daily_window_cluster_regression_k2.csv"))
    ]

def resolve_canon_to_actual(drivers_df: pd.DataFrame):
    available_norm = {normalize_var(v): v for v in drivers_df["variable"].dropna().astype(str).unique()}
    mapping = {}
    for canon in CANON_VARS:
        found = None
        for alias in VAR_ALIASES.get(canon, [canon]):
            if normalize_var(alias) in available_norm:
                found = available_norm[normalize_var(alias)]
                break
        mapping[canon] = found
    return mapping

def detect_time_column(df: pd.DataFrame):
    for c in ["target_day", "window_label", "period", "week"]:
        if c in df.columns:
            return c
    return None

def add_influence_percent(block: pd.DataFrame, value_col: str = "coef") -> pd.DataFrame:
    d = block.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["abscoef"] = d[value_col].abs()
    total_abs = d["abscoef"].sum(skipna=True)
    d["influence_pct"] = 100.0 * d["abscoef"] / total_abs if (pd.notna(total_abs) and total_abs > 0) else np.nan
    return d

def compute_residuals(reg: pd.DataFrame, ts: pd.DataFrame, xcol: str) -> pd.DataFrame:
    if ts.empty or reg.empty:
        return pd.DataFrame()

    ts = ts.copy()
    ts["Time"]  = pd.to_datetime(ts["Time"], errors="coerce")
    ts["_date"] = ts["Time"].dt.date

    residual_rows = []

    for (xval, cluster), coef_block in reg.groupby([xcol, "cluster"]):
        y_mean_vals = coef_block["y_mean"].dropna()
        y_std_vals  = coef_block["y_std"].dropna()
        if y_mean_vals.empty or y_std_vals.empty:
            continue
        y_mean = float(y_mean_vals.iloc[0])
        y_std  = float(y_std_vals.iloc[0])
        if pd.isna(y_mean) or pd.isna(y_std) or y_std <= 1e-12:
            continue

        const_rows = coef_block[coef_block["variable"] == "const"]
        intercept  = float(const_rows["coef"].iloc[0]) if not const_rows.empty else 0.0
        pred_vars  = coef_block[coef_block["variable"] != "const"]

        ts_window = ts[ts[xcol].astype(str) == str(xval)].copy()
        if ts_window.empty:
            continue

        if "target_day" in ts_window.columns:
            try:
                target_date = pd.to_datetime(ts_window["target_day"].iloc[0]).date()
                ts_day = ts_window[ts_window["_date"] == target_date].copy()
            except Exception:
                ts_day = ts_window
        else:
            ts_day = ts_window

        if ts_day.empty or cluster not in ts_day.columns:
            continue

        y_actual = float(ts_day[cluster].dropna().mean())
        if pd.isna(y_actual):
            continue

        pred_z = intercept
        for _, vrow in pred_vars.iterrows():
            var    = vrow["variable"]
            coef   = vrow["coef"]
            x_mean = vrow["x_mean"]
            x_std  = vrow["x_std"]
            if pd.isna(coef) or pd.isna(x_mean) or pd.isna(x_std) or x_std <= 1e-12:
                continue
            if var not in ts_day.columns:
                continue
            x_day = float(ts_day[var].dropna().mean())
            if pd.isna(x_day):
                continue
            pred_z += coef * (x_day - x_mean) / x_std

        y_predicted = pred_z * y_std + y_mean

        residual_rows.append({
            xcol:          xval,
            "group":       cluster,
            "y_actual":    y_actual,
            "y_predicted": y_predicted,
            "residual":    y_actual - y_predicted,
            "n_obs_day":   int(len(ts_day)),
        })

    return pd.DataFrame(residual_rows)


def get_r2_series_for_group(df: pd.DataFrame, group_key: str, xcol: str):
    if "R2" not in df.columns or "cluster" not in df.columns:
        return pd.DataFrame(columns=[xcol, "R2"])
    d = df[(df["cluster"] == group_key) & (df["variable"] == "const")].copy()
    if d.empty:
        return pd.DataFrame(columns=[xcol, "R2"])
    d["R2"] = pd.to_numeric(d["R2"], errors="coerce")
    return d[[xcol, "R2"]].dropna(subset=[xcol, "R2"]).drop_duplicates(subset=[xcol]).copy()

def build_legend_handles(dfc_pct: pd.DataFrame):
    present = set(dfc_pct["canon"].dropna())
    return [
        plt.Line2D([0], [0], color=VAR_COLORS[v], lw=2, label=VAR_LABELS[v])
        for v in CANON_VARS if v in present
    ]

# Plotting

def _grid(ax):
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()


def plot_group_timeseries(drivers: pd.DataFrame, reg: pd.DataFrame, out_dir: str,
                          group_key: str, xcol: str, range_label: str):
    out_path = os.path.join(out_dir, f"{group_key}_std_influence_percent.png")
    if (not FORCE_RERUN_ALL) and SKIP_EXISTING_PNG and os.path.exists(out_path):
        print(f"[SKIP] {out_path}")
        return out_path

    dfc = drivers[drivers["cluster"] == group_key].copy()
    if dfc.empty:
        print(f"[WARN] No rows for cluster {group_key}")
        return None

    pct_rows = [add_influence_percent(block) for _, block in dfc.groupby(xcol)]
    dfc_pct  = pd.concat(pct_rows, ignore_index=True) if pct_rows else pd.DataFrame()
    if dfc_pct.empty:
        return None

    canon2actual_rev = {a: c for c, a in resolve_canon_to_actual(dfc).items() if a is not None}
    dfc_pct["canon"] = dfc_pct["variable"].map(canon2actual_rev)
    dfc_pct = dfc_pct[dfc_pct["canon"].isin(CANON_VARS)].copy()
    if dfc_pct.empty:
        return None

    xvals = sorted(dfc_pct[xcol].dropna().astype(str).unique(), key=_time_key)
    xmap  = {v: i for i, v in enumerate(xvals)}
    dfc_pct["xpos"] = dfc_pct[xcol].astype(str).map(xmap)

    xticks, xticklabels, xlabel = get_day_axis_labels(xvals, xcol)
    date_title = get_date_span_title(range_label, xvals, xcol)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    for canon in CANON_VARS:
        d = dfc_pct[dfc_pct["canon"] == canon].sort_values("xpos")
        if d.empty:
            continue
        ax1.plot(d["xpos"], d["influence_pct"], label=VAR_LABELS.get(canon, canon),
                 color=VAR_COLORS.get(canon), linewidth=2)

    ax1.set_ylabel("Relative Influence [%]")
    ax1.set_title(f"{date_title}\n{GROUP_TITLES[group_key]}")
    ax1.set_ylim(0, 100)
    _grid(ax1)

    legend_handles = build_legend_handles(dfc_pct)
    if legend_handles:
        ax1.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=True)

    r2 = get_r2_series_for_group(reg, group_key, xcol)
    if not r2.empty:
        r2[xcol] = r2[xcol].astype(str)
        r2 = r2[r2[xcol].isin(xvals)].sort_values("xpos" if "xpos" in r2.columns else xcol)
        if "xpos" not in r2.columns:
            r2["xpos"] = r2[xcol].map(xmap)
        ax2.plot(r2["xpos"], r2["R2"], linestyle="--", linewidth=2, color="black")

    ax2.set_ylabel("R²")
    ax2.set_title("R²")
    ax2.set_ylim(0, 1)
    _grid(ax2)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, rotation=0)
    ax2.set_xlabel(xlabel)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"[OK] Saved: {out_path}")
    return out_path


def plot_combined_timeseries(drivers: pd.DataFrame, reg: pd.DataFrame, out_dir: str,
                             xcol: str, range_label: str | None = None):
    out_name = "combined_std_influence.png" if not range_label else f"{safe_filename_label(range_label)}_combined_std_influence.png"
    out_path = os.path.join(out_dir, out_name)
    if (not FORCE_RERUN_ALL) and SKIP_EXISTING_PNG and os.path.exists(out_path):
        print(f"[SKIP] {out_path}")
        return out_path

    group_data = {}
    all_xvals  = set()

    for group_key in GROUP_KEYS:
        dfc = drivers[drivers["cluster"] == group_key].copy()
        if dfc.empty:
            continue
        pct_rows = [add_influence_percent(block) for _, block in dfc.groupby(xcol)]
        dfc_pct  = pd.concat(pct_rows, ignore_index=True) if pct_rows else pd.DataFrame()
        if dfc_pct.empty:
            continue
        canon2actual_rev = {a: c for c, a in resolve_canon_to_actual(dfc).items() if a is not None}
        dfc_pct["canon"] = dfc_pct["variable"].map(canon2actual_rev)
        dfc_pct = dfc_pct[dfc_pct["canon"].isin(CANON_VARS)].copy()
        if dfc_pct.empty:
            continue
        dfc_pct[xcol] = dfc_pct[xcol].astype(str)
        all_xvals.update(dfc_pct[xcol].dropna().unique())
        r2 = get_r2_series_for_group(reg, group_key, xcol)
        if not r2.empty:
            r2[xcol] = r2[xcol].astype(str)
            all_xvals.update(r2[xcol].dropna().unique())
        group_data[group_key] = {"dfc_pct": dfc_pct, "r2": r2}

    if not group_data:
        print("[WARN] No valid group data for combined plot.")
        return None

    xvals = sorted([str(v) for v in all_xvals], key=_time_key)
    if not xvals:
        return None

    xticks, xticklabels, xlabel = get_day_axis_labels(xvals, xcol)
    xmap       = {v: i for i, v in enumerate(xvals)}
    date_title = get_date_span_title(range_label, xvals, xcol) if range_label else ""

    for gk in group_data:
        group_data[gk]["dfc_pct"]["xpos"] = group_data[gk]["dfc_pct"][xcol].map(xmap)
        if not group_data[gk]["r2"].empty:
            group_data[gk]["r2"]["xpos"] = group_data[gk]["r2"][xcol].map(xmap)

    n = len(GROUP_KEYS)
    fig, axes = plt.subplots(n * 2, 1, figsize=(15, 5 * n), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1] * n})

    for i, group_key in enumerate(GROUP_KEYS):
        ax_inf = axes[i * 2]
        ax_r2  = axes[i * 2 + 1]

        if group_key not in group_data:
            ax_inf.text(0.5, 0.5, f"No data for {group_key}", ha="center", va="center",
                        transform=ax_inf.transAxes)
            ax_r2.text(0.5, 0.5, "No R² data", ha="center", va="center",
                       transform=ax_r2.transAxes)
            continue

        dfc_pct = group_data[group_key]["dfc_pct"]
        r2      = group_data[group_key]["r2"]

        for canon in CANON_VARS:
            d = dfc_pct[dfc_pct["canon"] == canon].sort_values("xpos")
            if d.empty:
                continue
            ax_inf.plot(d["xpos"], d["influence_pct"], label=VAR_LABELS.get(canon, canon),
                        color=VAR_COLORS.get(canon), linewidth=2)

        ax_inf.set_ylabel("Relative Influence [%]")
        title_str = GROUP_TITLES[group_key]
        ax_inf.set_title(f"{date_title}\n{title_str}" if (i == 0 and date_title) else title_str)
        ax_inf.set_ylim(0, 100)
        _grid(ax_inf)

        if not r2.empty:
            r2_plot = r2[r2[xcol].isin(xvals)].sort_values("xpos")
            ax_r2.plot(r2_plot["xpos"], r2_plot["R2"], linestyle="-", linewidth=2, color="black")
        ax_r2.set_ylabel("R²")
        ax_r2.set_title("R²")
        ax_r2.set_ylim(0, 1)
        _grid(ax_r2)

    sample = next(iter(group_data.values()))["dfc_pct"]
    legend_handles = build_legend_handles(sample)
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=0)
    axes[-1].set_xlabel(xlabel)

    plt.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"[OK] Saved: {out_path}")
    return out_path


def plot_residuals_per_group(residuals: pd.DataFrame, out_dir: str, group_key: str,
                             xcol: str, range_label: str):
    out_path = os.path.join(out_dir, f"{group_key}_residuals.png")
    if (not FORCE_RERUN_ALL) and SKIP_EXISTING_PNG and os.path.exists(out_path):
        print(f"[SKIP] {out_path}")
        return out_path

    dfc = residuals[residuals["group"] == group_key].copy()
    if dfc.empty:
        print(f"[WARN] No residual rows for group {group_key}")
        return None

    dfc[xcol] = dfc[xcol].astype(str)
    xvals = sorted(dfc[xcol].dropna().unique(), key=_time_key)
    xmap  = {v: i for i, v in enumerate(xvals)}
    dfc["xpos"] = dfc[xcol].map(xmap)
    dfc = dfc.sort_values("xpos")

    xticks, xticklabels, xlabel = get_day_axis_labels(xvals, xcol)
    color = GROUP_COLORS[group_key]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(dfc["xpos"], dfc["residual"], color=color, linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Residual [°C]")
    ax1.set_title(f"{get_date_span_title(range_label, xvals, xcol)}\n{GROUP_TITLES[group_key]} — Residuals (target day)")
    _grid(ax1)

    ax2.bar(dfc["xpos"], dfc["n_obs_day"], color=color, alpha=0.6, width=0.7)
    ax2.set_ylabel("N obs.")
    ax2.set_title("N observations on target day")
    _grid(ax2)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, rotation=0)
    ax2.set_xlabel(xlabel)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"[OK] Saved: {out_path}")
    return out_path


def plot_residuals_combined(residuals: pd.DataFrame, out_dir: str, xcol: str,
                            range_label: str | None = None):
    out_name = "combined_residuals.png" if not range_label else f"{safe_filename_label(range_label)}_combined_residuals.png"
    out_path = os.path.join(out_dir, out_name)
    if (not FORCE_RERUN_ALL) and SKIP_EXISTING_PNG and os.path.exists(out_path):
        print(f"[SKIP] {out_path}")
        return out_path

    group_data = {}
    all_xvals  = set()

    for group_key in GROUP_KEYS:
        dfc = residuals[residuals["group"] == group_key].copy()
        if dfc.empty:
            continue
        dfc[xcol] = dfc[xcol].astype(str)
        all_xvals.update(dfc[xcol].dropna().unique())
        group_data[group_key] = dfc

    if not group_data:
        print("[WARN] No residual data for combined residuals plot.")
        return None

    xvals = sorted([str(v) for v in all_xvals], key=_time_key)
    if not xvals:
        return None

    xmap = {v: i for i, v in enumerate(xvals)}
    xticks, xticklabels, xlabel = get_day_axis_labels(xvals, xcol)
    date_title = get_date_span_title(range_label, xvals, xcol) if range_label else ""

    for gk in group_data:
        group_data[gk]["xpos"] = group_data[gk][xcol].map(xmap)

    n = len(GROUP_KEYS)
    fig, axes = plt.subplots(n * 2, 1, figsize=(15, 5 * n), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1] * n})

    for i, group_key in enumerate(GROUP_KEYS):
        ax_res = axes[i * 2]
        ax_n   = axes[i * 2 + 1]
        color  = GROUP_COLORS[group_key]
        title  = (f"{date_title}\n{GROUP_TITLES[group_key]} — Residuals (target day)"
                  if (i == 0 and date_title)
                  else f"{GROUP_TITLES[group_key]} — Residuals (target day)")

        if group_key not in group_data:
            ax_res.text(0.5, 0.5, f"No data for {group_key}", ha="center", va="center",
                        transform=ax_res.transAxes)
            ax_n.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_n.transAxes)
            continue

        s = group_data[group_key].sort_values("xpos")
        ax_res.plot(s["xpos"], s["residual"], color=color, linewidth=2)
        ax_res.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax_res.set_ylabel("Residual [°C]")
        ax_res.set_title(title)
        _grid(ax_res)

        ax_n.bar(s["xpos"], s["n_obs_day"], color=color, alpha=0.6, width=0.7)
        ax_n.set_ylabel("N obs.")
        ax_n.set_title("N observations on target day")
        _grid(ax_n)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels, rotation=0)
    axes[-1].set_xlabel(xlabel)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"[OK] Saved: {out_path}")
    return out_path


# Main

print(f"FORCE_RERUN_ALL = {FORCE_RERUN_ALL}")

for analysis_name in ANALYSIS_FAMILIES:
    analysis_root = os.path.join(ROOT_REG, analysis_name)

    if not os.path.exists(analysis_root):
        print(f"\n[SKIP] Not found: {analysis_root}")
        continue

    range_labels = get_range_folders(analysis_root)
    if not range_labels:
        print(f"\n[SKIP] No valid range folders in: {analysis_root}")
        continue

    print("\n" + "#" * 100)
    print(f"PROCESSING: {analysis_name}  ({len(range_labels)} ranges)")
    print("#" * 100)

    for range_label in range_labels:
        print("\n" + "=" * 90)
        print(f"RANGE: {analysis_name} | {range_label}")
        print("=" * 90)

        base     = os.path.join(analysis_root, range_label)
        reg_file = os.path.join(base, "daily_window_cluster_regression_k2.csv")
        out_dir  = os.path.join(base, "plots_std_influence")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(reg_file) or os.path.getsize(reg_file) == 0:
            print(f"[SKIP] Missing or empty: {reg_file}")
            continue

        try:
            reg = pd.read_csv(reg_file)
        except Exception as e:
            print(f"[SKIP] Could not read {reg_file}: {e}")
            continue

        missing_cols = {"cluster", "variable", "coef", "R2"} - set(reg.columns)
        if missing_cols:
            print(f"[SKIP] Missing columns: {sorted(missing_cols)}")
            continue

        if reg.empty:
            print(f"[SKIP] Empty: {reg_file}")
            continue

        xcol = detect_time_column(reg)
        if xcol is None:
            print("[SKIP] Could not detect time column.")
            continue

        reg[xcol] = reg[xcol].astype(str)
        print(f"Loaded {len(reg)} rows | time column: {xcol} | {reg[xcol].nunique()} unique periods")

        # Residuals
        ts_file = os.path.join(base, "daily_window_cluster_timeseries_k2.csv")
        if os.path.exists(ts_file) and os.path.getsize(ts_file) > 0:
            try:
                ts = pd.read_csv(ts_file)
                if xcol in ts.columns:
                    ts[xcol] = ts[xcol].astype(str)
                    residuals = compute_residuals(reg, ts, xcol)
                    if not residuals.empty:
                        res_path = os.path.join(base, "daily_window_cluster_residuals_k2.csv")
                        residuals.to_csv(res_path, index=False)
                        print(f"[OK] Residuals saved: {res_path} ({len(residuals)} rows)")
                        for gk in GROUP_KEYS:
                            plot_residuals_per_group(residuals, out_dir, gk, xcol, range_label)
                        plot_residuals_combined(residuals, out_dir, xcol, range_label=range_label)
                    else:
                        print("[WARN] No residuals computed.")
                else:
                    print(f"[WARN] Timeseries CSV missing column '{xcol}', skipping residuals.")
            except Exception as e:
                print(f"[SKIP] Could not compute residuals: {e}")
        else:
            print(f"[SKIP] Timeseries CSV not found: {ts_file}")

        # Influence plots
        for gk in GROUP_KEYS:
            plot_group_timeseries(reg, reg, out_dir, gk, xcol, range_label)
        plot_combined_timeseries(reg, reg, out_dir, xcol, range_label=range_label)

print("\nDONE.")
print("All cluster analysis families processed.")
