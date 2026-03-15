"""EDA Restart Orchestrator (Guide-Aligned)

Scope strictly to Shepperton plant and enforce dispatch constraints.
This script:
- Prepares processed parquet files from data/raw if missing
- Loads processed datasets
- Applies plant scope and depot filters
- Validates allowed SKUs and critical nulls
- Checks one-truck-per-day rule for dispatch
- Computes dispatch multi-SKU load composition
- Produces summary CSVs and simple figures under `reports/`

Run:
    python -m src.analysis.eda_restart_guide_aligned --prepare true
"""
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np

from src.utils.constants import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    REPORTS_DIR,
    SUMMARIES_DIR,
    FIGURES_DIR,
)
from src.validation.validators import (
    ensure_plant_scope,
    check_allowed_skus_column,
    check_critical_nulls,
    check_dispatch_uniqueness,
    enforce_depot_list,
    compute_dispatch_load_composition,
)
from src.data.clean_data import clean_file

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def ensure_processed_files():
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for fname in [
        "production_dataset.csv",
        "quality_control_dataset.csv",
        "dispatch_dataset.csv",
        "route_transport_multivehicle.csv",
        "sales_dataset.csv",
        "sales_pos_dataset.csv",
        "returns_dataset.csv",
        "waste_dataset.csv",
        "inventory_stock_movements.csv",
        "holiday_production_sales.csv",
        "equipment_iot_sensor.csv",
    ]:
        raw_path = DATA_RAW_DIR / fname
        out_parquet = DATA_PROCESSED_DIR / (raw_path.stem + ".parquet")
        if not out_parquet.exists():
            if raw_path.exists():
                logger.info(f"Processing {raw_path} -> {out_parquet}")
                clean_file(raw_path, DATA_PROCESSED_DIR)
            else:
                logger.warning(f"Missing raw file: {raw_path}")


def _safe_read(name: str) -> pd.DataFrame:
    p = DATA_PROCESSED_DIR / f"{name}.parquet"
    if not p.exists():
        logger.warning(f"Processed file not found: {p}; attempting to read CSV")
        csv_p = DATA_RAW_DIR / f"{name}.csv"
        if csv_p.exists():
            return pd.read_csv(csv_p)
        logger.error(f"Neither parquet nor CSV found for {name}")
        return pd.DataFrame()
    return pd.read_parquet(p)


def analyze_dispatch():
    df = _safe_read("dispatch_dataset")
    if df.empty:
        return
    df = ensure_plant_scope(df, "plant_id")
    df = enforce_depot_list(df, "depot_id")
    df = check_critical_nulls(df, "dispatch_dataset")
    check_dispatch_sku_columns = True  # validation warnings handled in validators
    dup_df = check_dispatch_uniqueness(df)
    if not dup_df.empty:
        (REPORTS_DIR / "models").mkdir(parents=True, exist_ok=True)
        dup_path = REPORTS_DIR / "models" / "flagged_anomalies.csv"
        dup_df.to_csv(dup_path, index=False)
        logger.info(f"Wrote dispatch uniqueness anomalies to {dup_path}")

    # Composition percentages (only when multi-SKU load columns exist)
    comp = compute_dispatch_load_composition(df)
    if not comp.empty:
        comp_out = SUMMARIES_DIR / "dispatch_load_composition.csv"
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        comp.to_csv(comp_out, index=False)
        logger.info(f"Wrote dispatch load composition to {comp_out}")

    # Basic time summaries
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour
        qty_col = "total_quantity" if "total_quantity" in df.columns else ("qty_dispatched" if "qty_dispatched" in df.columns else None)
        if qty_col:
            by_hour = df.groupby("hour")[qty_col].sum().reset_index()
            by_hour_out = SUMMARIES_DIR / "dispatch_by_hour.csv"
            by_hour.to_csv(by_hour_out, index=False)
            logger.info(f"Wrote {by_hour_out}")

    # By depot
    if "depot_id" in df.columns:
        qty_col = "total_quantity" if "total_quantity" in df.columns else ("qty_dispatched" if "qty_dispatched" in df.columns else None)
        if qty_col:
            by_depot = df.groupby("depot_id")[qty_col].sum().reset_index().sort_values(qty_col, ascending=False)
            by_depot_out = SUMMARIES_DIR / "dispatch_by_depot.csv"
            by_depot.to_csv(by_depot_out, index=False)
            logger.info(f"Wrote {by_depot_out}")


def analyze_production_qc_waste():
    prod = _safe_read("production_dataset")
    qc = _safe_read("quality_control_dataset")
    waste = _safe_read("waste_dataset")
    if prod.empty:
        return
    prod = ensure_plant_scope(prod, "plant_id")
    prod = check_allowed_skus_column(prod, "sku")
    prod = check_critical_nulls(prod, "production_dataset")
    # waste rate per batch (if batch_id aligns)
    if not waste.empty and "batch_id" in waste.columns:
        waste = ensure_plant_scope(waste, "plant_id")
        waste = check_allowed_skus_column(waste, "sku")
        waste = check_critical_nulls(waste, "waste_dataset")
        qty_col = "quantity_wasted" if "quantity_wasted" in waste.columns else ("qty_waste" if "qty_waste" in waste.columns else None)
        if qty_col is None:
            logger.info("Waste quantity column not found; skipping waste rate computation")
        else:
            wr = waste.groupby("batch_id")[qty_col].sum().rename("quantity_wasted").reset_index()
            prod_wr = prod.merge(wr, on="batch_id", how="left")
            if "quantity_produced" in prod_wr.columns:
                prod_wr["waste_rate_pct"] = (prod_wr["quantity_wasted"].fillna(0) / prod_wr["quantity_produced"]) * 100
            out = SUMMARIES_DIR / "production_waste_rate_by_batch.csv"
            SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
            prod_wr[["batch_id", "sku", "quantity_produced", "quantity_wasted", "waste_rate_pct"]].to_csv(out, index=False)
            logger.info(f"Wrote {out}")
    # QC fail rate per batch
    if not qc.empty and "batch_id" in qc.columns:
        qc = check_allowed_skus_column(qc, "sku")
        qc = check_critical_nulls(qc, "quality_control_dataset")
        qc_fail = qc.groupby("batch_id")["pass_fail"].apply(lambda s: (s == "fail").mean() * 100).rename("qc_fail_rate_pct").reset_index()
        out = SUMMARIES_DIR / "qc_fail_rate_by_batch.csv"
        qc_fail.to_csv(out, index=False)
        logger.info(f"Wrote {out}")


def analyze_sales_inventory_returns():
    sales = _safe_read("sales_dataset")
    pos = _safe_read("sales_pos_dataset")
    inv = _safe_read("inventory_stock_movements")
    returns = _safe_read("returns_dataset")

    # Sales dataset (aggregated by depot/sku/date)
    if not sales.empty:
        sales = ensure_plant_scope(sales, "plant_id")
        sales = enforce_depot_list(sales, "depot_id")
        sales = check_allowed_skus_column(sales, "sku")
        # normalize date
        if "date" in sales.columns:
            sales["date"] = pd.to_datetime(sales["date"], errors="coerce").dt.date
        by_depot_sku = sales.groupby(["depot_id", "sku"]).agg(quantity_sold=("quantity_sold", "sum")).reset_index()
        out = SUMMARIES_DIR / "sales_by_depot_sku.csv"
        by_depot_sku.to_csv(out, index=False)
        logger.info(f"Wrote {out}")

    # POS dataset (retail demand)
    if not pos.empty:
        pos = check_allowed_skus_column(pos, "sku")
        if "timestamp" in pos.columns:
            pos["timestamp"] = pd.to_datetime(pos["timestamp"], errors="coerce")
            pos["date"] = pos["timestamp"].dt.date
            by_day = pos.groupby("date")["quantity_sold"].sum().reset_index()
            out = SUMMARIES_DIR / "pos_units_by_day.csv"
            by_day.to_csv(out, index=False)
            logger.info(f"Wrote {out}")

    # Inventory reconciliation (negative balance anomalies)
    if not inv.empty:
        inv = ensure_plant_scope(inv, "plant_id")
        inv = check_allowed_skus_column(inv, "sku")
        inv = check_critical_nulls(inv, "inventory_stock_movements")
        neg = inv[inv.get("balance").astype(float) < 0]
        if not neg.empty:
            out = REPORTS_DIR / "inventory_negative_balance.csv"
            neg.to_csv(out, index=False)
            logger.info(f"Wrote {out}")

    # Returns analysis (surges)
    if not returns.empty:
        returns = check_allowed_skus_column(returns, "sku")
        if "timestamp" in returns.columns:
            returns["timestamp"] = pd.to_datetime(returns["timestamp"], errors="coerce")
            returns["date"] = returns["timestamp"].dt.date
            by_day = returns.groupby("date")["qty_returned"].sum().reset_index()
            out = SUMMARIES_DIR / "returns_by_day.csv"
            by_day.to_csv(out, index=False)
            logger.info(f"Wrote {out}")


def run(prepare: bool = False):
    if prepare:
        ensure_processed_files()
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    analyze_dispatch()
    analyze_production_qc_waste()
    analyze_sales_inventory_returns()

    logger.info("EDA restart complete (guide-aligned)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=lambda x: str(x).lower() in ["1","true","yes"], default=True)
    args = parser.parse_args()
    run(prepare=args.prepare)
