"""Assemble the week's analysis notebook from the scripts in src/ and execute it.

The scripts in src/ are written in Jupytext "percent" format (cells separated
by ``# %%``), so they run as ordinary Python files and can also be converted
cell-by-cell into a notebook. This keeps a single copy of the analysis code.

Usage (from the repository root):
    python notebooks/build_notebook.py
"""
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent
SRC_DIR = NOTEBOOK_DIR.parent / "src"

NOTEBOOK_NAME = "Week_5_Final_Project.ipynb"
SCRIPTS = ["07_final_analysis.py"]
INTRO = """# Week 5 - Final Project: Integrated Findings and Evidence for Recommendations

**Dataset:** IBM Telco Customer Churn (7,043 customers), acquired and cleaned in Week 1.

This notebook runs `src/07_final_analysis.py`, the final stage of the project pipeline.
It reads the saved results of stages 01-06 (EDA, visual story, hypothesis tests and the
churn model), so run `python src/run_all.py` first on a fresh checkout. The code cells
are taken directly from `src/`.
"""
SETUP = """import sys
sys.path.insert(0, "../src")  # make the project modules in src/ importable
%config InlineBackend.figure_format = 'retina'"""


def build_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook.cells.append(nbformat.v4.new_markdown_cell(INTRO))
    notebook.cells.append(nbformat.v4.new_code_cell(SETUP))
    for script in SCRIPTS:
        notebook.cells.extend(jupytext.read(SRC_DIR / script).cells)

    # Stable cell ids keep git diffs small when the notebook is rebuilt
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{index:03d}"
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    NotebookClient(notebook, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK_DIR)}}).execute()
    nbformat.write(notebook, NOTEBOOK_DIR / NOTEBOOK_NAME)
    print(f"Executed and saved notebooks/{NOTEBOOK_NAME}")
