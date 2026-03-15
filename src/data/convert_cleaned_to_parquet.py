"""Convert any `*.cleaned.csv` files in `data/processed` into Parquet files.

This script is defensive: it will attempt to coerce timestamps and numeric columns
before writing Parquet. It writes a `<stem>.parquet` file next to the CSV and
prints progress.

Usage:
    python src/data/convert_cleaned_to_parquet.py --processed_dir data/processed
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # parse timestamp-like columns
    time_cols = [c for c in df.columns if any(k in c for k in ('time', 'timestamp', 'date', 'arrival'))]
    for c in time_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            df[c] = pd.to_datetime(df[c].astype(str), infer_datetime_format=True, errors='coerce')

    # numeric coercion: try to coerce object columns with many numeric-like values
    for c in df.columns:
        if df[c].dtype == object:
            ser = df[c].astype(str).str.replace(',', '')
            coerced = pd.to_numeric(ser.replace('', np.nan), errors='coerce')
            if coerced.notnull().mean() > 0.9:
                df[c] = coerced
    return df


def convert_all(processed_dir: Path):
    processed_dir = Path(processed_dir)
    files = list(processed_dir.glob('*.cleaned.csv'))
    if not files:
        logger.info('No .cleaned.csv files found in %s', processed_dir)
        return
    for f in files:
        logger.info('Reading %s', f)
        try:
            df = pd.read_csv(f, engine='python', on_bad_lines='skip')
        except Exception as e:
            logger.exception('Failed to read %s: %s', f, e)
            continue
        df = coerce_types(df)
        out_path = processed_dir / (f.stem.replace('.cleaned','') + '.parquet')
        try:
            df.to_parquet(out_path, index=False)
            logger.info('Wrote parquet: %s', out_path)
        except Exception as e:
            logger.exception('Failed to write parquet for %s: %s', f, e)
            # as a final fallback, write a CSV with a _parquet_try prefix
            fallback = processed_dir / (f.stem + '.parquet_failed.csv')
            df.to_csv(fallback, index=False)
            logger.info('Wrote fallback CSV: %s', fallback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed')
    args = parser.parse_args()
    convert_all(args.processed_dir)
