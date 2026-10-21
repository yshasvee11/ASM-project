import gzip
import pathlib
import shutil
import urllib.request

# Folder where clean .csv files will be saved on your computer
output_folder = pathlib.Path("eicu_demo_data")
output_folder.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://physionet.org/files/eicu-crd-demo/2.0.1/"

# Core eICU-CRD Demo v2.0.1 tables needed for Person 2 & Queueing/Markov Modelling
TABLES = [
    "patient.csv.gz",              # 2,520 ICU stays, timestamps & offsets (133.6 KB)
    "hospital.csv.gz",             # 20 hospitals, bed capacity & region (759 B)
    "admissionDx.csv.gz",          # Admission diagnoses (93.6 KB)
    "apachePatientResult.csv.gz",  # APACHE IV severity & LOS predictions (209.1 KB)
    "treatment.csv.gz",            # Active ICU treatments with timestamps (450.1 KB)
    "diagnosis.csv.gz",            # Time-stamped active clinical problems (352.1 KB)
    "vitalAperiodic.csv.gz",       # Non-invasive blood pressure & vitals (2.5 MB)
    # Uncomment the line below if you also want the 18.5 MB 5-minute vitalPeriodic table:
    # "vitalPeriodic.csv.gz",
]

print(f"Downloading and extracting eICU-CRD Demo v2.0.1 into: {output_folder.resolve()}\n")

for gz_filename in TABLES:
    url = BASE_URL + gz_filename
    gz_path = output_folder / gz_filename
    csv_filename = gz_filename.replace(".csv.gz", ".csv")
    csv_path = output_folder / csv_filename

    try:
        print(f"1. Downloading {gz_filename} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(gz_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        print(f"2. Extracting  {gz_filename} -> {csv_filename} ...")
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Remove compressed .gz file after extracting clean .csv
        gz_path.unlink(missing_ok=True)
        print(f"   [DONE] Saved {csv_path}\n")
    except Exception as e:
        print(f"   [ERROR] Could not download {gz_filename}: {e}\n")

print("All done! You can now open patient.csv and hospital.csv directly in Excel or Python/Pandas.")