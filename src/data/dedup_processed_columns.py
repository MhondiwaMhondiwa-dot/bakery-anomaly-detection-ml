"""Normalize and remove duplicate-suffixed columns in processed parquet files.

- Scans `data/processed/*.parquet` and for each file:
  - Detect columns that end with a numeric suffix like `.1`, `_1`, `.2`, etc.
  - If the base column (without suffix) exists, drop the suffixed columns.
  - If the base column does not exist, rename the first suffixed column to the base name and drop subsequent suffixed columns.
- Writes a backup copy of the original parquet to `data/processed/backup/` before overwriting.
- Logs actions to console and to `reports/dedup_logs.txt`.

Usage:
    python src/data/dedup_processed_columns.py --processed_dir data/processed
"""
from pathlib import Path
import re
import argparse
import pandas as pd
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:[._](?P<num>\d+))$")


def process_file(path: Path, backup_dir: Path):
    logger.info(f"Processing {path}")
    df = pd.read_parquet(path)
    cols = list(df.columns)
    groups = {}
    for c in cols:
        m = SUFFIX_RE.match(c)
        if m:
            base = m.group('base')
            groups.setdefault(base, []).append(c)
    actions = []
    for base, suffixed in groups.items():
        if base in cols:
            # Drop all suffixed columns
            for s in suffixed:
                df.drop(columns=[s], inplace=True)
                actions.append(f"dropped {s} because {base} exists")
        else:
            # Rename the first suffixed to base, drop others
            suffixed_sorted = sorted(suffixed)
            first = suffixed_sorted[0]
            df.rename(columns={first: base}, inplace=True)
            actions.append(f"renamed {first} -> {base}")
            for s in suffixed_sorted[1:]:
                df.drop(columns=[s], inplace=True)
                actions.append(f"dropped {s} (duplicate)")
    if actions:
        # backup original
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        logger.info(f"Backed up {path.name} -> {backup_path}")
        # write cleaned parquet (overwrite)
        df.to_parquet(path, index=False)
        logger.info(f"Wrote cleaned parquet to {path}")
    else:
        logger.info(f"No duplicate-suffixed columns found in {path.name}")
    return actions


def main(processed_dir: str):
    p = Path(processed_dir)
    backup_dir = p / 'backup'
    log_path = Path('reports') / 'dedup_logs.txt'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf8') as logf:
        for f in sorted(p.glob('*.parquet')):
            actions = process_file(f, backup_dir)
            if actions:
                logf.write(f"File: {f.name}\n")
                for a in actions:
                    logf.write(f" - {a}\n")
                logf.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed')
    args = parser.parse_args()
    main(args.processed_dir)
