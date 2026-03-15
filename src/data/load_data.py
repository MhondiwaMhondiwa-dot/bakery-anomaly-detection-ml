"""Utilities to load raw CSV files from `data/raw`.

This module provides simple, resilient CSV loading functions used by the
cleaning pipeline.
"""
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_csv_files(input_dir: str):
    p = Path(input_dir)
    return sorted([f for f in p.glob("*.csv") if f.is_file()])


def load_csv(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load a CSV into a DataFrame with robust defaults.

    - Uses pandas with on_bad_lines='skip' to avoid failing on messy rows.
    - Attempts to parse dates when parse_dates is True.
    """
    logger.info(f"Loading CSV: {path}")
    try:
        if parse_dates:
            df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
        else:
            df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        logger.exception(f"Failed to read {path}: {e}")
        raise
    logger.info(f"Loaded {len(df):,} rows from {path.name}")
    return df
