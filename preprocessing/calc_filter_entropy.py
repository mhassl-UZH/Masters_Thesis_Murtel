#Calulates entropy threshold based on labels.csv and then copies all good images to a new subfolder. 
# Input: labels.csv; raw TIR CSVs path 
# Output: New subfolder with only good images

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
CSV_FOLDER   = os.path.join(DATA_DIR, "decoded_images")
LABELS_CSV   = os.path.join(DATA_DIR, "decoded_images", "labels.csv")
OUT_FOLDER   = os.path.join(DATA_DIR, "decoded_images_filtered")
OPTIMIZE_FOR     = "roc"   # "f1" or "roc"
CUSTOM_THRESHOLD = 5.550    # set to a float (e.g. 5.5) to skip calculation, or None to calculate from labels
W, H             = 336, 252
SKIP_ROWS        = 8

os.makedirs(OUT_FOLDER, exist_ok=True)


def compute_entropy(arr: np.ndarray) -> float:
    pixels = arr.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(np.nanmin(pixels), np.nanmax(pixels)))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def read_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None, sep=";", skiprows=SKIP_ROWS)
    return df.to_numpy(dtype=np.float32).reshape(H, W)


if CUSTOM_THRESHOLD is not None:
    threshold = float(CUSTOM_THRESHOLD)
    print(f"Using custom threshold = {threshold:.3f}")
else:
    # Read labels
    df_labels = pd.read_csv(LABELS_CSV, sep=";", header=None)
    if df_labels.shape[1] == 1:
        df_labels = pd.read_csv(LABELS_CSV, sep=",", header=None)
    df_labels = df_labels.iloc[:, :2].copy()
    df_labels.columns = ["file", "label"]

    # Compute entropy for labeled files
    files, entropies, labels = [], [], []

    for _, row in df_labels.iterrows():
        stem = os.path.splitext(os.path.basename(str(row["file"])))[0]
        csv_path = os.path.join(CSV_FOLDER, stem + ".csv")

        if not os.path.isfile(csv_path):
            print(f"Missing: {csv_path}")
            continue

        try:
            arr = read_csv(csv_path)
            entropies.append(compute_entropy(arr))
            labels.append(int(row["label"]))
            files.append(stem + ".csv")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    df = pd.DataFrame({"file": files, "entropy": entropies, "label": labels})

    if df.empty:
        raise RuntimeError("No matching CSV files found.")

    x = df["entropy"].to_numpy()
    y = df["label"].to_numpy()

    # Find optimal threshold
    if OPTIMIZE_FOR.lower() == "f1":
        thresholds = np.linspace(float(x.min()), float(x.max()), 200)
        best_thr, best_f1 = thresholds[0], -1.0
        for t in thresholds:
            f1 = f1_score(y, (x >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        threshold = float(best_thr)
        print(f"Best F1 = {best_f1:.3f} at threshold {threshold:.3f}")
    else:
        fpr, tpr, thr_vals = roc_curve(y, x)
        threshold = float(thr_vals[np.argmax(tpr - fpr)])
        print(f"Best Youden-J threshold = {threshold:.3f}")

    # Plot entropy distributions
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df.label == 1, "entropy"], bins=50, alpha=0.6, label="Good")
    plt.hist(df.loc[df.label == 0, "entropy"], bins=50, alpha=0.6, label="Bad")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Filter all CSVs using threshold
all_csvs = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith(".csv")]
print(f"\nApplying threshold to {len(all_csvs)} CSVs in {CSV_FOLDER}")

n_good, n_bad = 0, 0

for fname in sorted(all_csvs):
    fpath = os.path.join(CSV_FOLDER, fname)
    try:
        arr = read_csv(fpath)
        ent = compute_entropy(arr)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        continue

    if ent >= threshold:
        shutil.copy2(fpath, os.path.join(OUT_FOLDER, fname))
        n_good += 1
    else:
        n_bad += 1

print(f"\nDone. {n_good} kept, {n_bad} filtered out.")
print(f"Output folder: {OUT_FOLDER}")
