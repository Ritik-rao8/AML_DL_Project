"""Shared project paths, dataset columns and reproducibility seed."""
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
SEED = 42
COL_INPUT = "input"
COL_LABEL = "label"
COL_NUM_VARS = "num_variables"
COL_TEMPLATE = "template"
