"""Feature engineering for anomaly detection analytics.

This module computes the derived signals described in the project plan and
builds a unified analytical table at a chosen grain.  It is meant to be run
once after the processed datasets are available; the outputs are written to
`data/analytic/` for modelling.

The core anomaly targets (production defect, route delay, retailer demand
collapse, route waste spike) are all supported by the features computed
here.

Usage:
    python -m src.features.engineer
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/processed")
ANALYTIC = Path("data/analytic")
ANALYTIC.mkdir(exist_ok=True)


def production_features() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "production_dataset.parquet")
    # timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date

    # defect rate       
    defect_cols = [
        "stacked_before_robot",
        "squashed",
        "torn",
        "undersized_small",
        "valleys",
        "loose_packs",
        "pale_underbaked",
    ]
    df["total_defects"] = df[defect_cols].sum(axis=1)
    df["defect_rate"] = df["total_defects"] / df["quantity_produced"].replace(0, np.nan)

    # rolling output for sudden drop
    df = df.sort_values("timestamp")
    df["seven_day_avg_output"] = (
        df.groupby("sku")["quantity_produced"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["output_drop"] = df["quantity_produced"] < 0.6 * df["seven_day_avg_output"]

    # operator-specific defect ratio
    df["operator_defect_rate"] = (
        df.groupby("operator_id")["total_defects"]
        .transform("sum")
        / df.groupby("operator_id")["quantity_produced"].transform("sum")
    )

    return df


def dispatch_features() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "dispatch_dataset.parquet")
    # timestamps: ensure we have a common `timestamp` field to group by
    if "timestamp" not in df.columns:
        if "departure_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["departure_time"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT
    for col in ["expected_arrival", "actual_arrival", "departure_time", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # delay
    if "actual_arrival" in df.columns and "expected_arrival" in df.columns:
        df["delay_minutes"] = (df["actual_arrival"] - df["expected_arrival"]).dt.total_seconds() / 60.0
    else:
        df["delay_minutes"] = np.nan
    # flags used in anomaly plan
    df["late_flag"] = df["delay_minutes"] >= 60
    df["early_arrival_flag"] = df["delay_minutes"] < -30
    # load utilization
    sku_cols = [c for c in ["soft_white", "high_energy_brown", "whole_grain_loaf", "low_gi_seed_loaf"] if c in df.columns]
    if sku_cols:
        df["total_load"] = df[sku_cols].sum(axis=1)
    else:
        df["total_load"] = np.nan
    cap_cols = [c for c in ["vehicle_capacity", "capacity_units", "truck_capacity"] if c in df.columns]
    df["load_capacity"] = df[cap_cols].iloc[:, 0] if cap_cols else np.nan
    df["utilization"] = df["total_load"] / df["load_capacity"].replace(0, np.nan)

    return df


def sales_pos_features() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "sales_pos_dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    # compute quantity_sold if multi-SKU
    sku_cols = [c for c in ["soft_white", "high_energy_brown", "whole_grain_loaf", "low_gi_seed_loaf"] if c in df.columns]
    if sku_cols:
        df["quantity_sold"] = df[sku_cols].sum(axis=1)
    else:
        df["quantity_sold"] = df.get("quantity_sold", 0)
    # rolling mean per retailer
    df = df.sort_values("timestamp")
    df["retailer_7d_avg"] = (
        df.groupby("retailer_id")["quantity_sold"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["demand_collapse"] = df["quantity_sold"] < 0.5 * df["retailer_7d_avg"]
    # promotion uplift
    df["promotion_flag"] = df.get("promotion_flag", 0)
    return df


def waste_features() -> pd.DataFrame:
    # Now reading from processed parquet instead of raw CSV (fixed in cleaning)
    df = pd.read_parquet(RAW / "waste_dataset.parquet")
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["date"] = df["timestamp"].dt.date
    # qty_waste already exists in processed data (cleaned from quantity_wasted)
    if "qty_waste" not in df.columns:
        sku_cols = [c for c in ["soft_white","high_energy_brown","whole_grain_loaf","low_gi_seed_loaf"] if c in df.columns]
        df["qty_waste"] = df[sku_cols].sum(axis=1)
    else:
        # Ensure numeric type
        df["qty_waste"] = pd.to_numeric(df["qty_waste"], errors="coerce").fillna(0)
    # route spike: compute route mean/std
    stats = df.groupby("route_id")["qty_waste"].agg(["mean","std"]).rename(columns={"mean":"route_mean","std":"route_std"})
    df = df.join(stats, on="route_id")
    df["route_spike"] = df["qty_waste"] > df["route_mean"] + 2 * df["route_std"]
    # reason code counts and temperature flags (waste_reason already exists)
    if "waste_reason" in df.columns:
        df["reason_count"] = df.groupby("waste_reason")["waste_id"].transform("count")
        df = df.rename(columns={"waste_reason":"waste_reason_code"})
    if "temperature_at_check" in df.columns:
        df["high_temp_flag"] = df["temperature_at_check"] > 30
    return df


def returns_features() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "returns_dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    # quantity returned from SKU columns if present
    sku_cols = [c for c in ["soft_white","high_energy_brown","whole_grain_loaf","low_gi_seed_loaf"] if c in df.columns]
    df["qty_returned"] = df[sku_cols].sum(axis=1) if sku_cols else 0
    # attach dispatch volume for return rate calculation when possible
    # Note: dispatch now has route_id after cleaning enrichment
    try:
        disp = pd.read_parquet(RAW / "dispatch_dataset.parquet")
        # Compute total dispatched per route_id (where available)
        if "route_id" in disp.columns:
            sku_disp = [c for c in ["soft_white","high_energy_brown","whole_grain_loaf","low_gi_seed_loaf"] if c in disp.columns]
            disp["qty_dispatched"] = disp[sku_disp].sum(axis=1) if sku_disp else disp.get("total_quantity", 0)
            route_disp = disp.groupby("route_id")["qty_dispatched"].sum().rename("route_qty_dispatched")
            df = df.join(route_disp, on="route_id")
            df["route_return_rate"] = df["qty_returned"] / df["route_qty_dispatched"].replace(0, np.nan)
        else:
            df["route_return_rate"] = np.nan
    except Exception:
        df["route_return_rate"] = np.nan
    # rolling statistics for spikes
    stats = df.groupby("route_id")["qty_returned"].agg(["mean","std"]).rename(columns={"mean":"route_ret_mean","std":"route_ret_std"})
    df = df.join(stats, on="route_id")
    df["route_return_spike"] = df["qty_returned"] > df["route_ret_mean"] + 2 * df["route_ret_std"]
    return df


def inventory_features() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "inventory_stock_movements_dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["negative_balance_flag"] = df["balance_after"] < 0
    # expiry tracking
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
        df["days_to_expiry"] = (df["expiry_date"] - df["timestamp"]).dt.days
        df["nearing_expiry_flag"] = df["days_to_expiry"] < 2
    # sales-inventory mismatch via simple proxy: later
    return df


def qc_features() -> pd.DataFrame:
    """Quality Control features with batch-level pass rates and parameter failures."""
    df = pd.read_parquet(RAW / "quality_control_dataset.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    
    # Binary pass/fail flag
    df["qc_pass"] = df["pass_fail"] == "pass"
    df["qc_fail"] = df["pass_fail"] == "fail"
    
    # Parameter-specific failure flags
    df["moisture_fail"] = (df["parameter"] == "moisture") & (df["pass_fail"] == "fail")
    df["seal_strength_fail"] = (df["parameter"] == "seal_strength") & (df["pass_fail"] == "fail")
    df["temperature_fail"] = (df["parameter"] == "temperature") & (df["pass_fail"] == "fail")
    df["weight_fail"] = (df["parameter"] == "weight") & (df["pass_fail"] == "fail")
    
    # Batch-level QC pass rate (critical for anomaly detection)
    batch_stats = df.groupby("batch_id").agg(
        batch_qc_pass_rate=("qc_pass", "mean"),
        batch_qc_tests=("qc_id", "count"),
        batch_qc_fails=("qc_fail", "sum")
    )
    df = df.join(batch_stats, on="batch_id")
    
    # Batch fails QC if pass rate < 60%
    df["batch_qc_fail_flag"] = df["batch_qc_pass_rate"] < 0.6
    
    return df


def holiday_context() -> pd.DataFrame:
    """Load holidays and create contextual flags for pre/post holiday periods."""
    holidays = pd.read_parquet(RAW / "holidays_calendar.parquet")
    holidays["date"] = pd.to_datetime(holidays["date"])
    
    # Create a date range covering the full dataset period (2025)
    date_range = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    df = pd.DataFrame({'date': date_range})
    
    # Mark holiday dates
    df["is_holiday"] = df["date"].isin(holidays["date"])
    
    # Pre-holiday flag (day before holiday)
    holiday_dates = set(holidays["date"])
    df["is_pre_holiday"] = df["date"].apply(lambda d: (d + pd.Timedelta(days=1)) in holiday_dates)
    
    # Post-holiday flag (day after holiday)
    df["is_post_holiday"] = df["date"].apply(lambda d: (d - pd.Timedelta(days=1)) in holiday_dates)
    
    # Merge holiday names for reference
    df = df.merge(holidays.rename(columns={'date': 'holiday_date'}), 
                  left_on='date', right_on='holiday_date', how='left')
    df = df.drop('holiday_date', axis=1, errors='ignore')
    
    # Convert date to date object for joining
    df["date"] = df["date"].dt.date
    
    return df


def build_daily_route_table() -> pd.DataFrame:
    """Merge production, dispatch, waste, returns, QC, sales POS, inventory and holiday context at route-day level."""
    prod = production_features()
    disp = dispatch_features()
    waste = waste_features()
    returns = returns_features()
    qc = qc_features()
    sales = sales_pos_features()
    inventory = inventory_features()
    holidays = holiday_context()

    # aggregate
    prod_agg = prod.groupby(prod["timestamp"].dt.date).agg(
        total_prod=("quantity_produced","sum"),
        avg_defect=("defect_rate","mean"),
        high_defect_count=("defect_rate",lambda x: (x>0.1).sum())
    )
    disp_agg = disp.groupby(disp["timestamp"].dt.date).agg(
        avg_delay=("delay_minutes","mean"),
        late_pct=("late_flag","mean"),
        early_pct=("early_arrival_flag","mean")
    )
    waste_agg = waste.groupby(waste["date"]).agg(
        total_waste=("qty_waste","sum"),
        route_spike_pct=("route_spike","mean")
    )
    ret_agg = returns.groupby(returns["timestamp"].dt.date).agg(
        total_return=("qty_returned","sum"),
        return_spike_pct=("route_return_spike","mean")
    )
    qc_agg = qc.groupby(qc["date"]).agg(
        qc_pass_rate=("qc_pass","mean"),
        qc_fail_pct=("qc_fail","mean"),
        qc_tests=("qc_id","count"),
        batch_fail_count=("batch_qc_fail_flag","sum"),
        moisture_fail_count=("moisture_fail","sum"),
        seal_fail_count=("seal_strength_fail","sum"),
        temp_fail_count=("temperature_fail","sum"),
        weight_fail_count=("weight_fail","sum")
    )
    sales_agg = sales.groupby(sales["date"]).agg(
        total_sold=("quantity_sold","sum"),
        demand_collapse_pct=("demand_collapse","mean"),
        demand_collapse_count=("demand_collapse","sum"),
        promotion_days=("promotion_flag","sum"),
        avg_retailer_sales=("quantity_sold","mean")
    )
    inventory_agg = inventory.groupby(inventory["date"]).agg(
        negative_balance_count=("negative_balance_flag","sum"),
        stock_movements=("movement_id","count") if "movement_id" in inventory.columns else ("timestamp","count"),
        nearing_expiry_count=("nearing_expiry_flag","sum") if "nearing_expiry_flag" in inventory.columns else ("timestamp",lambda x: 0)
    )
    
    # Start with holidays as base (includes all dates in 2025)
    analytic = holidays.set_index('date')
    
    # Join all feature aggregations
    analytic = analytic.join([prod_agg, disp_agg, waste_agg, ret_agg, qc_agg, sales_agg, inventory_agg], how='outer')

    # compute simple anomaly flags based on project thresholds and z-scores
    analytic = flag_anomalies(analytic)

    analytic.to_parquet(ANALYTIC / "plant_daily.parquet")
    return analytic



def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Attach columns indicating potential anomalies.

    Thresholds come from the key‑signals document and are deliberately
    simple so they can be inspected by a human.  A future model can
    replace or augment these rules.
    """
    df = df.copy()
    # production defect: critical when average defect exceeds 15%
    df["prod_defect_anomaly"] = df["avg_defect"] > 0.15
    # delay anomalies: >60 min average or >30% of routes late
    df["delay_anomaly"] = df["avg_delay"] > 60
    df["late_route_anomaly"] = df["late_pct"] > 0.3
    # return anomalies based on rolling spike proportion
    df["return_anomaly"] = df["return_spike_pct"] > 0.1
    # waste spikes
    df["waste_anomaly"] = df["route_spike_pct"] > 0.1    # QC anomaly: fail rate > 40% (baseline is 38.15%)
    if "qc_fail_pct" in df.columns:
        df["qc_anomaly"] = df["qc_fail_pct"] > 0.4
    # Sales anomaly: demand collapse > 10% of retailers
    if "demand_collapse_pct" in df.columns:
        df["sales_anomaly"] = df["demand_collapse_pct"] > 0.1
    # Inventory anomaly: negative balance detected
    if "negative_balance_count" in df.columns:
        df["inventory_anomaly"] = df["negative_balance_count"] > 0
    
    # Adjust anomaly detection for holiday context
    # Don't flag sales spikes on holidays/pre-holidays as anomalies (expected demand surge)
    if "is_holiday" in df.columns and "sales_anomaly" in df.columns:
        df["sales_anomaly"] = df["sales_anomaly"] & ~(df["is_holiday"] | df["is_pre_holiday"])
    
    # z‑score based flag for any numeric signal (extreme value)
    for col in ["avg_defect","avg_delay","total_return","total_waste","qc_fail_pct","demand_collapse_pct","total_sold"]:
        if col in df.columns:
            z = (df[col] - df[col].mean()) / df[col].std()
            df[col + "_z"] = z
            df[col + "_extreme"] = z.abs() > 2
    return df


if __name__ == "__main__":
    print("Generating analytic dataset...")
    table = build_daily_route_table()
    print(table.head())
    print("Written to", ANALYTIC / "plant_daily.parquet")
