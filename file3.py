import csv
import gzip
import json
import pathlib
import shutil
import subprocess
import time
import urllib.request

# 1. Folder for eICU-CRD Demo v2.0.1 CSV files
output_folder = pathlib.Path("eicu_demo_data")
output_folder.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://physionet.org/files/eicu-crd-demo/2.0.1/"

TABLES = [
    "patient.csv.gz",              # 2,520 ICU stays, timestamps & offsets (133.6 KB)
    "hospital.csv.gz",             # 20 hospitals, bed capacity & region (759 B)
    "admissionDx.csv.gz",          # Admission diagnoses (93.6 KB)
    "apachePatientResult.csv.gz",  # APACHE IV severity & LOS predictions (209.1 KB)
    "treatment.csv.gz",            # Active ICU treatments with timestamps (450.1 KB)
    "diagnosis.csv.gz",            # Time-stamped active clinical problems (352.1 KB)
    "vitalAperiodic.csv.gz",       # Non-invasive blood pressure & vitals (2.5 MB)
]

print(f"Checking & repairing eICU-CRD Demo v2.0.1 in: {output_folder.resolve()}\n")

for gz_filename in TABLES:
    url = BASE_URL + gz_filename
    gz_path = output_folder / gz_filename
    csv_filename = gz_filename.replace(".csv.gz", ".csv")
    csv_path = output_folder / csv_filename
    tmp_csv_path = output_folder / (csv_filename + ".tmp")

    # If a leftover .gz file exists from a previous interrupted run, the .csv is incomplete
    if gz_path.exists():
        print(f"[REPAIR] Found interrupted download for {gz_filename}, cleaning up partial files...")
        gz_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    # Skip files that are already cleanly extracted
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"[SKIP] {csv_filename} is already complete ({csv_path.stat().st_size:,} bytes).")
        continue

    for attempt in range(1, 5):
        try:
            print(f"1. Downloading {gz_filename} (Attempt {attempt}/4) ...")
            # Use built-in Windows curl.exe with auto-resume & retries first, fallback to urllib
            curl_cmd = [
                "curl.exe", "-L", "--fail", "--retry", "5", "--retry-delay", "2",
                "-A", "Mozilla/5.0", "-o", str(gz_path), url
            ]
            res = subprocess.run(curl_cmd, capture_output=True, text=True)
            if res.returncode != 0 or not gz_path.exists() or gz_path.stat().st_size == 0:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=180) as response, open(gz_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)

            print(f"2. Verifying & Extracting {gz_filename} -> {csv_filename} ...")
            with gzip.open(gz_path, "rb") as f_in, open(tmp_csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            # Only move to final .csv after 100% clean decompression
            tmp_csv_path.replace(csv_path)
            gz_path.unlink(missing_ok=True)
            print(f"   [DONE] Saved clean {csv_path} ({csv_path.stat().st_size:,} bytes)\n")
            break
        except Exception as e:
            print(f"   [WARN] Attempt {attempt} failed for {gz_filename}: {e}")
            gz_path.unlink(missing_ok=True)
            tmp_csv_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)
            time.sleep(3)

# 2. Folder for CMS Timely and Effective Care - Hospital (yv7e-xc69)
cms_folder = pathlib.Path("cms_data")
cms_folder.mkdir(parents=True, exist_ok=True)
cms_csv_path = cms_folder / "cms_timely_and_effective_care_hospital.csv"

if not (cms_csv_path.exists() and cms_csv_path.stat().st_size > 0):
    print(f"\nDownloading CMS Timely and Effective Care (yv7e-xc69) into: {cms_csv_path.resolve()} ...")
    cms_api_url = "https://data.cms.gov/provider-data/api/1/datastore/query/yv7e-xc69/0?limit=1500&offset=0"
    try:
        req = urllib.request.Request(cms_api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
            results = payload.get("results", [])
            if results:
                fieldnames = list(results[0].keys())
                with open(cms_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
                print(f"   [DONE] Saved {len(results):,} CMS hospital rows to {cms_csv_path}")
    except Exception as e:
        print(f"   [WARN] Could not fetch CMS API directly: {e}")
        print("   -> You can download the full CSV in browser at: https://data.cms.gov/provider-data/dataset/yv7e-xc69")
else:
    print(f"\n[SKIP] {cms_csv_path.name} already exists.")

print("\nAll done! Check your 'eicu_demo_data' and 'cms_data' folders.")