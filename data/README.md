Data directory layout

- `raw/`  : Place original unprocessed CSV/PDF/Excel files here. These files are treated as immutable sources.
- `processed/` : Cleaned and preprocessed files produced by `src/data/prepare_data.py`.
- `interim/` : Temporary or intermediate datasets created during processing.

How to run the cleaning pipeline

1. Activate your virtual environment.
2. From the project root run:

```bash
python src/data/prepare_data.py --input_dir data/raw --output_dir data/processed
```

This will read all `.csv` files from `data/raw`, clean them, and write cleaned files to `data/processed`.

Notes

- The cleaning pipeline attempts to write Parquet files; if Parquet writing fails (missing optional dependencies), a cleaned CSV is written instead.
- Inspect the cleaned files in `data/processed` and report any unexpected column transformations.
 
Convert any `.cleaned.csv` files to Parquet

```bash
python src/data/convert_cleaned_to_parquet.py --processed_dir data/processed
```

This will attempt to coerce timestamps and numeric types and write `<name>.parquet` files next to the cleaned CSVs.
