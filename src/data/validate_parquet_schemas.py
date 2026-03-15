"""Validate and summarize Parquet files in `data/processed`.

Prints for each parquet file:
- file path
- number of rows
- columns with dtypes
- null counts per column
- up to 3 example values per column

Usage:      
    python src/data/validate_parquet_schemas.py --processed_dir data/processed
"""
from pathlib import Path
import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def summarize_parquet(path: Path, max_examples: int = 3):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.exception('Failed to read parquet %s: %s', path, e)
        return
    print('\n' + '='*80)
    print(f'File: {path.name}')
    print(f'Rows: {len(df):,}')
    print('Columns and dtypes:')
    # df.dtypes is a Series (column -> dtype); use .items() for iteration
    for col, dtype in df.dtypes.items():
        try:
            nulls = int(df[col].isnull().sum())
            pct_null = nulls / len(df) if len(df) > 0 else 0
            print(f' - {col}: {dtype} (nulls: {nulls} | {pct_null:.1%})')
            # show examples (convert to python types for readable printing)
            examples = list(pd.unique(df[col].dropna()))[:max_examples]
            ex_list = ', '.join([str(x) for x in examples]) if len(examples) else '[]'
            print(f'    examples: {ex_list}')
        except Exception as e:
            print(f' - {col}: <unable to summarize: {e}>')


def main(processed_dir: str):
    p = Path(processed_dir)
    files = sorted(p.glob('*.parquet'))
    if not files:
        print('No parquet files found in', processed_dir)
        return
    for f in files:
        summarize_parquet(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed')
    args = parser.parse_args()
    main(args.processed_dir)
