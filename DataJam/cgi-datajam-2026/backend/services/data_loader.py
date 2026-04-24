import os
from pathlib import Path

import pandas as pd

# Resolve `backend/data/raw` relative to this file, with optional env override.
_BASE_DIR = Path(__file__).resolve().parents[1]  # .../backend
DATA_DIR = Path(os.getenv("DATA_DIR", _BASE_DIR / "data" / "raw")).resolve()


def get_data_path(filename: str) -> Path:
    return DATA_DIR / filename


def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(get_data_path(filename), **kwargs)
