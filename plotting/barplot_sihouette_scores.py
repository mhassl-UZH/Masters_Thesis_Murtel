import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings

CLUSTERS_ROOT = os.path.join(DATA_DIR, "corrected", "analysis", "temperature_curve_clusters")
OUT_PATH      = os.path.join(CLUSTERS_ROOT, "best_k_barplot.png")

K_RANGE = range(2, 8)

# Load silhouette summaries

csv_files = glob.glob(os.path.join(CLUSTERS_ROOT, "**", "silhouette_summary_*.csv"), recursive=True)
print(f"Found {len(csv_files)} silhouette summary files.")

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f, sep=";")
        dfs.append(df)
    except Exception as e:
        print(f"  [SKIP] {f}: {e}")

if not dfs:
    raise RuntimeError("No silhouette data found.")

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(data)}")

# Drop missing silhouette and exclude Clouds_NoClouds
data = data.dropna(subset=["silhouette"])
data = data[data["analysis_name"] != "Clouds_NoClouds"]
data["k"] = data["k"].astype(int)

# Find best k per window
group_cols = ["analysis_name", "range_label", "target_day", "window_label"]

best_k = (
    data.loc[data.groupby(group_cols)["silhouette"].idxmax()]
    [group_cols + ["k", "silhouette"]]
    .reset_index(drop=True)
)

print(f"Unique windows evaluated: {len(best_k)}")

counts = best_k["k"].value_counts().reindex(K_RANGE, fill_value=0).sort_index()

print("\nBest-k counts:")
for k, c in counts.items():
    print(f"  k={k}: {c} times ({100*c/len(best_k):.1f}%)")

# Plot

fig, ax = plt.subplots(figsize=(7, 4))

bars = ax.bar(
    counts.index,
    counts.values,
    color="#4d80af",
    edgecolor="white",
    linewidth=0.6,
    width=0.6,
)

# Count and percentage labels on each bar
for bar, (k, count) in zip(bars, counts.items()):
    pct = 100 * count / len(best_k)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts) * 0.01,
        f"{count}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=9
    )

ax.set_xlabel("Cluster Count k")
ax.set_ylabel("Number of Windows")
ax.set_title("Distribution of Optimal k across All Windows")
ax.set_xticks(list(K_RANGE))
ax.set_ylim(0, max(counts) * 1.18)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
