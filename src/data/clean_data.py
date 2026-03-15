"""Cleaning and preprocessing helpers for the bakery datasets.

The functions here attempt to standardize column names, parse common timestamp
fields, unify quantity column names (qty variants), remove duplicate columns,
and write cleaned outputs to `data/processed`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _standardize_colname(col: str) -> str:
    col = col.strip()
    col = col.replace('\n', '_').replace('\r', '_')
    col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
    col = ''.join(c for c in col if c.isalnum() or c == '_')
    return col.lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_standardize_colname(c) for c in df.columns]
    return df


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Append numeric suffixes to duplicate column names to make them unique.

    This avoids pandas returning a DataFrame when indexing by a duplicate column
    label (which causes `.dtype` attribute errors)."""
    cols = list(df.columns)
    counts = {}
    new_cols = []
    for c in cols:
        if c in counts:
            counts[c] += 1
            new_cols.append(f"{c}_{counts[c]}")
        else:
            counts[c] = 0
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Try to identify common timestamp-like column names
    # include arrival/departure/eta/etd keywords so arrival columns are parsed
    time_keywords = ('time', 'timestamp', 'date', 'arrival', 'depart', 'eta', 'etd')
    time_cols = [c for c in df.columns if any(k in c for k in time_keywords)]
    for c in time_cols:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors='coerce')

        # Normalize pure date columns (named `date` or ending with `_date`) to date dtype
        if c == 'date' or c.endswith('_date'):
            df[c] = parsed.dt.date
        else:
            df[c] = parsed
    return df


def unify_qty_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Common qty column name patterns
    qty_candidates = [c for c in df.columns if ('qty' in c or 'quantity' in c) and not c.startswith('qty_')]
    # We will not rename every qty column blindly, but create canonical names where appropriate
    # For dispatch and sales tables we commonly want `qty_dispatched`, `quantity_sold`, `qty_waste`, `qty_returned`.
    mapping = {}
    for c in df.columns:
        if c in ['qty_dispatched', 'qty_dispatched']:
            continue
        if 'dispat' in c and 'qty' in c:
            mapping[c] = 'qty_dispatched'
        if 'dispatch' in c and ('qty' in c or 'quantity' in c):
            mapping[c] = 'qty_dispatched'
        if ('quantity_sold' in c) or (c in ['qty_sold', 'quantity_sold', 'quantity_sold']):
            mapping[c] = 'quantity_sold'
        if 'qty_return' in c or 'returned' in c:
            mapping[c] = 'qty_returned'
        # Fix: only rename columns that are actually quantity fields, not IDs
        if ('quantity_wasted' in c or 'qty_wasted' in c) and 'waste_id' not in c:
            if 'qty_waste' not in df.columns:
                mapping[c] = 'qty_waste'
    df = df.rename(columns=mapping)
    return df


def _coerce_object_columns(df: pd.DataFrame, numeric_threshold: float = 0.9) -> pd.DataFrame:
    """Try to coerce object columns to numeric where most values convert cleanly.

    This reduces mixed-type object columns that cause parquet serialization errors.
    """
    df = df.copy()
    for c in df.columns:
        ser = df[c]
        # if duplicate column labels returned a DataFrame, use first occurrence
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        if getattr(ser, 'dtype', None) == object:
            ser = ser.astype(str).str.strip()
            # try numeric coercion
            coerced = pd.to_numeric(ser.str.replace(',', ''), errors='coerce')
            ratio = coerced.notnull().mean()
            if ratio >= numeric_threshold:
                df[c] = coerced
    return df


def drop_all_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are exact duplicates of other columns."""
    df = df.copy()
    cols_to_drop = []
    seen_hashes = {}
    for c in df.columns:
        ser = df[c]
        try:
            h = hash(tuple(ser.fillna('__NA__').astype(str).values[:1000]))
        except Exception:
            h = None
        if h is not None and h in seen_hashes:
            cols_to_drop.append(c)
        elif h is not None:
            seen_hashes[h] = c
    if cols_to_drop:
        logger.info(f"Dropping duplicate columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column labels are unique by appending numeric suffixes to duplicates.

    This is used after canonical renames (which can map several source columns to
    the same target name) to avoid duplicate column labels that break parquet.
    """
    df = df.copy()
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def clean_file(input_path: Path, output_dir: Path) -> Path:
    logger.info(f"Cleaning file: {input_path.name}")
    df = pd.read_csv(input_path, sep=None, engine='python', on_bad_lines='skip')
    df = standardize_columns(df)
    df = make_columns_unique(df)
    df = parse_timestamps(df)
    df = unify_qty_columns(df)
    # ensure column labels are unique after any canonical renames
    df = ensure_unique_columns(df)
    df = drop_all_duplicate_columns(df)
    
    # Enrich dispatch with route_id from route_transport metadata
    if 'vehicle_id' in df.columns and 'route_id' not in df.columns and 'dispatch' in input_path.stem.lower():
        # Read route metadata from RAW directory to ensure we use latest data
        route_file_raw = input_path.parent / 'route_transport_multivehicle.csv'
        route_file_processed = output_dir / 'route_transport_multivehicle.parquet'
        
        # Prefer raw CSV (latest data), fallback to processed parquet
        route_file = route_file_raw if route_file_raw.exists() else route_file_processed
        
        if route_file.exists():
            try:
                if route_file.suffix == '.csv':
                    routes = pd.read_csv(route_file)
                else:
                    routes = pd.read_parquet(route_file)
                # Get unique vehicle→route mapping (pick first route per vehicle if multiple)
                vehicle_route_map = routes[['vehicle_id', 'route_id']].drop_duplicates('vehicle_id', keep='first')
                df = df.merge(vehicle_route_map, on='vehicle_id', how='left')
                logger.info(f"Enriched dispatch with route_id: {df['route_id'].notna().sum()} of {len(df)} records mapped (source: {route_file.name})")
            except Exception as e:
                logger.warning(f"Could not enrich dispatch with route_id: {e}")
    
    # Remove duplicate dispatch records: one-truck-per-day
    if 'vehicle_id' in df.columns and ('departure_time' in df.columns or 'timestamp' in df.columns):
        date_col = 'timestamp' if 'timestamp' in df.columns else 'departure_time'
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        before = len(df)
        df = df.drop_duplicates(subset=["vehicle_id", "date"], keep="first")
        after = len(df)
        logger.info(f"Removed dispatch duplicates: {before} -> {after} rows (one-truck-per-day)")

    # Convert numeric-like columns
    for c in df.columns:
        ser = df[c]
        # If indexing by label returned a DataFrame (duplicate column names), pick the first column
        if isinstance(ser, pd.DataFrame):
            logger.warning(f"Column label '{c}' refers to multiple columns; using first occurrence for type coercion")
            ser = ser.iloc[:, 0]

        # proceed only if we have a Series-like object
        if hasattr(ser, 'dtype') and ser.dtype == object:
            # try to coerce to numeric where ~90% convertible
            coerced = pd.to_numeric(ser.astype(str).str.replace(',', '').replace('', np.nan), errors='coerce')
            notnull_ratio = coerced.notnull().mean()
            if notnull_ratio > 0.9:
                df[c] = coerced

    # attempt to coerce other object columns that look numeric to avoid parquet write issues
    df = _coerce_object_columns(df, numeric_threshold=0.9)

    # Prepare output path
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_path.stem + '.parquet')
    try:
        df.to_parquet(out_path, index=False)
        logger.info(f"Wrote cleaned parquet to {out_path}")
    except Exception:
        logger.warning("Parquet write failed — attempting to coerce object columns and retry")
        try:
            df2 = _coerce_object_columns(df, numeric_threshold=0.6)
            df2.to_parquet(out_path, index=False)
            logger.info(f"Wrote cleaned parquet to {out_path} after coercion")
            out_path = out_path
        except Exception as e:
            out_csv = output_dir / (input_path.stem + '.cleaned.csv')
            df.to_csv(out_csv, index=False)
            logger.info(f"Parquet write failed after retry ({e}); wrote CSV to {out_csv}")
            out_path = out_csv

    return out_path
