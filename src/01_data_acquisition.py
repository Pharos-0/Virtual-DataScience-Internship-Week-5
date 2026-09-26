# %% [markdown]
# ## Part 1 - Data acquisition
#
# The IBM Telco Customer Churn dataset is published by IBM as part of the
# `telco-customer-churn-on-icp4d` Code Pattern on GitHub under the Apache 2.0
# licence. The CSV is downloaded straight from the repository's raw URL, its
# SHA-256 checksum is recorded (and checked on later runs), and the licence
# text is saved next to the data because Apache 2.0 requires it to travel with
# redistributed copies.

# %%
import hashlib
import json

import pandas as pd
import requests
from IPython.display import display

from config import (DATASET_URL, EXPECTED_SHA256, LICENSE_PATH, LICENSE_URL,
                    PROJECT_ROOT, RAW_DATA_PATH, RESULTS_DIR, SOURCE_REPOSITORY)


def download_file(url, destination, timeout=60):
    """Download a file over HTTPS and write it to ``destination``."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def sha256_of(path):
    """Return the SHA-256 hex digest of a file (used to verify the download)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# %% tags=["shot-acquisition"]
# Download the dataset only if it is not already present locally
if RAW_DATA_PATH.exists():
    print(f"Found existing file: {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
else:
    download_file(DATASET_URL, RAW_DATA_PATH)
    print(f"Downloaded {DATASET_URL}")

if not LICENSE_PATH.exists():
    download_file(LICENSE_URL, LICENSE_PATH)

checksum = sha256_of(RAW_DATA_PATH)
if EXPECTED_SHA256 is not None and checksum != EXPECTED_SHA256:
    raise ValueError("Checksum mismatch: the source file has changed since this project was run.")

df_raw = pd.read_csv(RAW_DATA_PATH)
print(f"File size : {RAW_DATA_PATH.stat().st_size:,} bytes")
print(f"SHA-256   : {checksum}")
print(f"Rows      : {df_raw.shape[0]:,}")
print(f"Columns   : {df_raw.shape[1]}")
# First rows, first eight columns (the full inspection is in Part 2)
display(df_raw.head(3).iloc[:, :8])

# %%
# Record acquisition metadata so the report and later weeks can cite exact figures
acquisition_log = {
    "dataset_name": "IBM Telco Customer Churn",
    "source_repository": SOURCE_REPOSITORY,
    "download_url": DATASET_URL,
    "license": "Apache License 2.0",
    "file_name": RAW_DATA_PATH.name,
    "file_size_bytes": RAW_DATA_PATH.stat().st_size,
    "sha256": checksum,
    "n_rows": int(df_raw.shape[0]),
    "n_columns": int(df_raw.shape[1]),
    "columns": list(df_raw.columns),
}
with open(RESULTS_DIR / "acquisition_log.json", "w") as f:
    json.dump(acquisition_log, f, indent=2)
print("Saved outputs/results/acquisition_log.json")
