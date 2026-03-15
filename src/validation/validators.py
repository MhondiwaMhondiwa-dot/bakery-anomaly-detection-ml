from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

from src.utils.constants import (
    SHEPPERTON_PLANT_ID,
    STD_SKU_COLS,
    ALLOWED_SKUS,
    ALLOWED_DEPOTS,
    CRITICAL_FIELDS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ensure_plant_scope(df: pd.DataFrame, plant_col: str = "plant_id") -> pd.DataFrame:
    """Filter DataFrame to Shepperton plant and log scope enforcement.

    Falls back to 'plant' column if 'plant_id' absent.
    """
    col = plant_col if plant_col in df.columns else ("plant" if "plant" in df.columns else None)
    if col:
        before = len(df)
        df = df[df[col] == SHEPPERTON_PLANT_ID].copy()
        logger.info(f"Plant scope enforced: {before:,} -> {len(df):,} rows ({SHEPPERTON_PLANT_ID}) using column '{col}'")
    else:
        logger.info(f"Plant column '{plant_col}' not found; skipping plant scope filter")
    return df


def check_allowed_skus_column(df: pd.DataFrame, sku_col: str = "sku") -> pd.DataFrame:
    """Validate SKU values for datasets with a single `sku` column.

    Drops rows with invalid SKU and logs issues.
    """
    if sku_col not in df.columns:
        logger.info(f"SKU column '{sku_col}' not in DataFrame; skipping allowed SKU check")
        return df
    valid_mask = df[sku_col].isin(ALLOWED_SKUS)
    invalid = df.loc[~valid_mask, sku_col].dropna().unique().tolist()
    if invalid:
        logger.warning(f"Found invalid SKUs (removed): {invalid}")
    return df.loc[valid_mask].copy()


def check_dispatch_sku_columns(df: pd.DataFrame) -> None:
    """Ensure dispatch dataset has standardized SKU load columns.

    Warn if expected SKU columns missing or unexpected SKU columns present.
    """
    if not set(STD_SKU_COLS).issubset(df.columns):
        missing = [c for c in STD_SKU_COLS if c not in df.columns]
        if missing:
            logger.warning(f"Dispatch missing expected SKU columns: {missing}")
    # Identify unexpected sku-like columns
    unexpected = [
        c for c in df.columns
        if any(k in c for k in ["soft", "brown", "grain", "seed"]) and c not in STD_SKU_COLS
    ]
    if unexpected:
        logger.warning(f"Dispatch has unexpected SKU-related columns: {unexpected}")


def check_critical_nulls(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Drop rows with nulls in critical fields and log count removed."""
    fields = CRITICAL_FIELDS.get(dataset_key, [])
    if not fields:
        logger.info(f"No critical field list found for '{dataset_key}'; skipping null check")
        return df
    before = len(df)
    df = df.dropna(subset=[c for c in fields if c in df.columns])
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.warning(f"Removed {removed:,} rows with critical nulls in {dataset_key}")
    return df


def check_dispatch_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    """Respect one-truck-per-day rule: detect and flag duplicates.

    Returns a DataFrame of duplicate rows (anomalies) and logs counts.
    """
    if "vehicle_id" not in df.columns or "timestamp" not in df.columns:
        logger.info("Dispatch uniqueness check skipped: missing 'vehicle_id' or 'timestamp'")
        return pd.DataFrame(columns=df.columns)
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["timestamp"], errors="coerce").dt.date
    dup_mask = tmp.duplicated(subset=["vehicle_id", "date"], keep=False)
    dup_df = tmp.loc[dup_mask].sort_values(["vehicle_id", "date"])
    if not dup_df.empty:
        logger.warning(f"One-truck-per-day violations: {len(dup_df):,} records detected")
        # Optionally remove duplicates
        df_cleaned = tmp.drop_duplicates(subset=["vehicle_id", "date"], keep="first")
        logger.info(f"Dispatch duplicates removed: {len(tmp)} -> {len(df_cleaned)} rows")
        return df_cleaned
    else:
        logger.info("Dispatch uniqueness OK: no violations detected")
        return tmp


def enforce_depot_list(df: pd.DataFrame, depot_col: str = "depot_id") -> pd.DataFrame:
    """Filter dispatch/sales to allowed depots only."""
    if depot_col in df.columns:
        before = len(df)
        df = df[df[depot_col].isin(ALLOWED_DEPOTS)].copy()
        logger.info(f"Depot filter applied: {before:,} -> {len(df):,} rows")
    return df


def compute_dispatch_load_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dispatch SKU composition percentages using standardized SKU columns."""
    required = [c for c in STD_SKU_COLS if c in df.columns]
    if not required:
        logger.info("No standardized SKU load columns found; skipping composition computation")
        return pd.DataFrame(index=df.index)
    total = df[required].sum(axis=1).replace(0, pd.NA)
    comp = pd.DataFrame(index=df.index)
    for c in required:
        comp[f"pct_{c}"] = (df[c] / total) * 100
    comp["total_from_components"] = df[required].sum(axis=1)
    return comp
