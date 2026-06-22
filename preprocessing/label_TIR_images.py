# Labeling tool for TIR images (random selection) to calculate entropy threshold

# Key 1 = good imagew
# Key 0 = bad image
# Key q = quit

import os
import cv2
import csv
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
IMG_DIR = os.path.join(DATA_DIR, "decoded_images")   # TIR
RGB_DIR = os.path.join(DATA_DIR, "TIR_camera", "RGB_images")
OUT_CSV = os.path.join(IMG_DIR, "labels.csv")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
N_SAMPLES = 100

TZ = ZoneInfo("Europe/Zurich")

def parse_ts(fname):
    try:
        ts = fname[1:13]  # YYMMDDhhmmss
        return datetime.strptime(ts, "%y%m%d%H%M%S")
    except Exception:
        return None

def parse_ts_dst_corrected(fname):
    dt = parse_ts(fname)
    if dt is None:
        return None
    dst = dt.replace(tzinfo=TZ).dst()
    if dst is not None and dst != timedelta(0):
        dt = dt - timedelta(hours=1)
    return dt


# Collect files
tir_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if os.path.splitext(f)[1].lower() in EXTS
])

rgb_files = sorted([
    f for f in os.listdir(RGB_DIR)
    if os.path.splitext(f)[1].lower() in EXTS
])

if not tir_files:
    print("No TIR images found.")
    raise SystemExit

# Prepare RGB timestamps
rgb_ts = []
for f in rgb_files:
    t = parse_ts_dst_corrected(f)
    if t:
        rgb_ts.append((t, f))

# Random selection
if len(tir_files) > N_SAMPLES:
    files = random.sample(tir_files, N_SAMPLES)
else:
    files = tir_files

print(f"{len(files)} images selected — labeling starts!")

# Load existing labels
labeled = {}
if os.path.isfile(OUT_CSV):
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) == 2:
                labeled[row[0]] = row[1]

# GUI
cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Label Tool", 1600, 750)

with open(OUT_CSV, "a", newline="", encoding="utf-8") as fcsv:
    writer = csv.writer(fcsv, delimiter=";")

    for i, fname in enumerate(files, 1):
        if fname in labeled:
            continue

        tir_path = os.path.join(IMG_DIR, fname)
        tir_img = cv2.imread(tir_path)

        if tir_img is None:
            print(f"Could not load {fname}.")
            continue

        # Find nearest RGB
        t_tir = parse_ts(fname)
        rgb_img = None

        if t_tir and rgb_ts:
            nearest = min(rgb_ts, key=lambda x: abs((x[0] - t_tir).total_seconds()))
            rgb_path = os.path.join(RGB_DIR, nearest[1])
            rgb_img = cv2.imread(rgb_path)

        # Fallback: white image
        if rgb_img is None:
            rgb_img = 255 * np.ones_like(tir_img)

        # Resize to matching height
        h = min(tir_img.shape[0], rgb_img.shape[0])
        tir_res = cv2.resize(tir_img, (int(tir_img.shape[1]*h/tir_img.shape[0]), h))
        rgb_res = cv2.resize(rgb_img, (int(rgb_img.shape[1]*h/rgb_img.shape[0]), h))

        combined = cv2.hconcat([tir_res, rgb_res])

        cv2.imshow("Label Tool", combined)
        print(f"[{i}/{len(files)}] {fname}")

        key = cv2.waitKey(0)

        if key == ord("1"):
            label = 1
        elif key == ord("0"):
            label = 0
        elif key == ord("q"):
            print("\nAborted.")
            break
        else:
            print("Invalid key.") 
            continue

        writer.writerow([fname, label])
        fcsv.flush()
        print(f"{fname} -> {label}\n")

cv2.destroyAllWindows()
print(f"\nSaved: {OUT_CSV}")
print(f"Done at {datetime.now().strftime('%H:%M:%S')}")
