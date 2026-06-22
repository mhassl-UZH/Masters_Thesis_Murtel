#Extracts internal camera temperature from TIR PNGs. 
# Input: TIR PNG folder 
# Output: internal_camera_temps.csv

import os
import re
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Zurich")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Settings
IN_FOLDER  = os.path.join(DATA_DIR, "TIR_camera", "TIR_raw_data")
OUT_FOLDER = os.path.join(DATA_DIR, "CSVs")
os.makedirs(OUT_FOLDER, exist_ok=True)
OUT_CSV = os.path.join(OUT_FOLDER, "internal_camera_temps.csv")

def extract_tou_from_jpg(filepath):
    pattern = re.compile(rb"SECTION\s+SENSORS.*?TOU\s*=\s*(-?\d+).*?ENDSECTION\s+SENSORS", re.S)
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        match = pattern.search(data)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None


def format_tou(tou):
    if tou is None:
        return None
    return f"{tou / 10:.1f}"


def extract_timestamp_from_filename(fname):
    match = re.search(r"m(\d{15})", fname)  # YYMMDDhhmmssxxx
    if not match:
        return None

    digits = match.group(1)
    date_digits = digits[:12]   # YYMMDDhhmmss
    millis = digits[12:]        # xxx (milliseconds)

    try:
        dt = datetime.strptime(date_digits, "%y%m%d%H%M%S")
        dst = dt.replace(tzinfo=TZ).dst()
        if dst is not None and dst != timedelta(0):
            dt = dt - timedelta(hours=1)
        timestamp = f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{millis}"
        return timestamp
    except Exception:
        return None


def process_folder(folder, output_csv):
    results = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(folder, fname)
        tou_raw = extract_tou_from_jpg(fpath)
        tou_formatted = format_tou(tou_raw)
        timestamp = extract_timestamp_from_filename(fname)
        results.append({
            "filename": fname,
            "timestamp": timestamp,
            "cam_temp": tou_formatted
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "timestamp", "cam_temp"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {output_csv}")
    print(f"Entries written: {len(results)}")
    if len(results) > 0:
        print("\nExample entries:")
        for r in results[:3]:
            print(r)

if __name__ == "__main__":
    print(f"Extracting TOU from JPGs in: {IN_FOLDER}")
    process_folder(IN_FOLDER, OUT_CSV)
