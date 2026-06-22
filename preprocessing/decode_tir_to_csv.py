#Decodes output PNGs from the mobotix camera to CSVs with °C per pixel 
#and corrects for daylight saving time. 
# Input: MOBOTIX decoder path; Raw TIR PNGs
# Output: TIR CSVs containing °C values per pixel

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

# Only process images on or after this date (set to None to process all)
START_DATE = "2021-01-01"

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

    if START_DATE and dt < datetime.strptime(START_DATE, "%Y-%m-%d"):
        print(f"Skip (before {START_DATE}): {base}")
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
