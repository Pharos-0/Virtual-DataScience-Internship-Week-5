"""Run the complete project pipeline (Weeks 1-5) in order.

Usage (from the repository root):
    python src/run_all.py

Each stage runs as a separate Python process, exactly as if it were started from
the terminal, and the pipeline stops at the first failure.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGES = [
    ("Week 1", "01_data_acquisition.py"),
    ("Week 1", "02_data_cleaning.py"),
    ("Week 1", "03_eda.py"),
    ("Week 2", "04_visualization.py"),
    ("Week 3", "05_statistical_analysis.py"),
    ("Week 4", "06_modeling.py"),
    ("Week 5", "07_final_analysis.py"),
]

if __name__ == "__main__":
    for week, script in STAGES:
        start = time.time()
        print(f"[{week}] running {script} ...", flush=True)
        completed = subprocess.run([sys.executable, str(SRC_DIR / script)], capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-4000:])
            sys.exit(f"Stage {script} failed.")
        print(f"[{week}] {script} finished in {time.time() - start:.1f} s")
    print("Pipeline complete: all outputs are in outputs/")
