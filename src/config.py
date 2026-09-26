"""Project configuration for the consolidated final project (Weeks 1-5).

Every script imports its paths from here, so the whole pipeline can be run from
any working directory (terminal, run_all.py or the notebook in notebooks/).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "models"

# Source: IBM Code Pattern "telco-customer-churn-on-icp4d" (Apache License 2.0)
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
LICENSE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/LICENSE"
)
SOURCE_REPOSITORY = "https://github.com/IBM/telco-customer-churn-on-icp4d"

RAW_DATA_PATH = RAW_DIR / "Telco-Customer-Churn.csv"
LICENSE_PATH = RAW_DIR / "LICENSE-Apache-2.0.txt"
CLEAN_DATA_PATH = PROCESSED_DIR / "telco_churn_clean.csv"

# SHA-256 of the raw file as first downloaded, and of the cleaned file produced
# by src/02_data_cleaning.py. Both are checked when the files are used.
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
CLEAN_DATA_SHA256 = "ec3ef03fbe050a90c06893621f92720beaa3136176ef5e91e1b5fc4d40164fec"

RANDOM_SEED = 42

for _directory in (RAW_DIR, PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, RESULTS_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
