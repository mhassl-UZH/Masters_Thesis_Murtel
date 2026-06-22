#Decodes output PNGs from the mobotix camera to CSVs with °C per pixel
#and corrects for daylight saving time, restricted to the analysis periods
#defined in temperature_curve_grouping.py (plus their rolling-window lookback).
# Input: MOBOTIX decoder path; Raw TIR PNGs
# Output: TIR CSVs containing °C values per pixel, for the analysis periods only

import os
import re
import glob
import shutil
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

# Input/output paths
DECODER_EXE = os.path.join(DATA_DIR, "TIR_camera", "MOBOTIX_decoder", "decoder.exe")
TIR_DIR     = os.path.join(DATA_DIR, "TIR_camera", "TIR_raw_data")
OUT_DIR     = os.path.join(DATA_DIR, "decoded_images")

# Rolling-window lookback used in temperature_curve_grouping.py — images this many
# days before a range's start are also needed to build the first daily windows
ROLLING_WINDOW_DAYS = 5

# Analysis date ranges, mirroring ANALYSIS_RANGES_snow_nosnow / _day_night /
# _transition / _rain_norain in analysis/temperature_curve_grouping.py
ANALYSIS_RANGES = [
    # Snow_NoSnow
    {"start": "05.08.21", "end": "23.08.21"},
    {"start": "07.06.22", "end": "19.07.22"},
    {"start": "26.02.22", "end": "01.04.22"},
    # Day_Night
    {"start": "07.06.22", "end": "19.07.22"},
    {"start": "26.02.22", "end": "14.03.22"},
    # Transition
    {"start": "14.06.21", "end": "25.07.21"},
    {"start": "13.09.21", "end": "14.11.21"},
    {"start": "09.05.22", "end": "26.06.22"},
    {"start": "31.10.22", "end": "18.12.22"},
    {"start": "15.05.23", "end": "02.07.23"},
    # Rain_NoRain
    {"start": "05.07.22", "end": "22.07.22"},
    {"start": "23.07.22", "end": "08.08.22"},
    {"start": "24.02.22", "end": "28.03.22"},
    {"start": "29.01.22", "end": "23.02.22"},
]

def parse_range_date(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%d.%m.%y")

# Build merged (start, end) intervals, expanded backwards by the rolling-window
# lookback so the first daily windows of each range have enough preceding data
_intervals = []
for r in ANALYSIS_RANGES:
    start = parse_range_date(r["start"]) - timedelta(days=ROLLING_WINDOW_DAYS - 1)
    end   = parse_range_date(r["end"]).replace(hour=23, minute=59, second=59, microsecond=999999)
    _intervals.append((start, end))

_intervals.sort()
PERIOD_INTERVALS = []
for start, end in _intervals:
    if PERIOD_INTERVALS and start <= PERIOD_INTERVALS[-1][1]:
        PERIOD_INTERVALS[-1] = (PERIOD_INTERVALS[-1][0], max(PERIOD_INTERVALS[-1][1], end))
    else:
        PERIOD_INTERVALS.append((start, end))

def in_analysis_period(dt: datetime) -> bool:
    return any(start <= dt <= end for start, end in PERIOD_INTERVALS)

# Timezone and filename pattern
TZ      = ZoneInfo("Europe/Zurich")
PATTERN = re.compile(r"m(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{3})")

os.makedirs(OUT_DIR, exist_ok=True)

# Find all jpg images (case-insensitive deduplication for Windows)
tir_files = sorted({p.lower(): p for p in
    glob.glob(os.path.join(TIR_DIR, "*.jpg")) +
    glob.glob(os.path.join(TIR_DIR, "*.JPG"))
}.values())

for src in tir_files:
    base = os.path.splitext(os.path.basename(src))[0]

    # Parse timestamp and apply daylight saving time correction
    m = PATTERN.search(base)
    if not m:
        print(f"Skip (no timestamp in name): {base}")
        continue

    yy, MM, DD, hh, mm, ss, ms = map(int, m.groups())
    dt = datetime(2000 + yy, MM, DD, hh, mm, ss, ms * 1000)

    dt_local = dt.replace(tzinfo=TZ)
    dst = dt_local.dst()
    if dst is not None and dst != timedelta(0):
        dt = dt - timedelta(hours=1)

    if not in_analysis_period(dt):
        print(f"Skip (outside analysis periods): {base}")
        continue

    corrected_name = dt.strftime("m%y%m%d%H%M%S") + f"{dt.microsecond // 1000:03d}"

    print(f"Processing: {base} -> {corrected_name}")

    # Run decoder in a dedicated temp folder so outputs are easy to locate
    temp_dir = os.path.join(OUT_DIR, "_tmp_" + base)
    os.makedirs(temp_dir, exist_ok=True)
    out_base = os.path.join(temp_dir, base)

    subprocess.run(
        [DECODER_EXE, src, out_base],
        check=False
    )

    matches = glob.glob(os.path.join(temp_dir, "*thermal.clecius.csv"))

    if matches:
        src_csv = matches[0]
        dst_csv = os.path.join(OUT_DIR, corrected_name + ".csv")

        if os.path.exists(dst_csv):
            os.remove(dst_csv)

        shutil.copy2(src_csv, dst_csv)
        print(f"Saved: {dst_csv}")
    else:
        print(f"No CSV found for: {base}")

    # Delete temp folder with all decoder outputs
    shutil.rmtree(temp_dir, ignore_errors=True)

print("Done.")
