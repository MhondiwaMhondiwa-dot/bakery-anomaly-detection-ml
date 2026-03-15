"""Script to iterate over CSVs in `data/raw` and produce cleaned files.

Usage:
    python src/data/prepare_data.py --input_dir data/raw --output_dir data/processed
"""
from pathlib import Path
import argparse
from load_data import list_csv_files
from clean_data import clean_file


def main(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = list_csv_files(input_dir)
    if not files:
        print(f"No CSV files found in {input_dir}")
        return
    for f in files:
        try:
            out = clean_file(f, output_dir)
            print(f"Processed {f.name} -> {out}")
        except Exception as e:
            print(f"Failed to process {f.name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/raw')
    parser.add_argument('--output_dir', default='data/processed')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
