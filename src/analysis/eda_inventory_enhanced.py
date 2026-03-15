"""
Enhanced EDA for Inventory / Stock Movements Dataset

Business Context:
================
The Inventory Dataset is THE STATE OF THE SYSTEM. It represents the final reconciliation
of all supply chain activities:

Production → created stock
QC → approved/rejected stock
Dispatch → moved stock
Sales/POS → consumed stock
Returns → stock came back
Waste → stock destroyed
INVENTORY → "After everything... how much stock do we have, and where?"

Inventory is not just another dataset — it is the LEDGER that reconciles all activities.

Critical Business Questions:
1. Does balance_after = balance_before + qty_in - qty_out? (reconciliation check)
2. Are there negative balances? (data integrity issue)
3. Where is stock aging past expiry? (waste risk)
4. What is inventory turnover rate? (efficiency)
5. Are there unexplained stock adjustments? (shrinkage/theft)
6. Which SKUs have highest inventory risk? (slow-moving stock)
7. Do plant vs store inventories match dispatch/sales patterns?

Anomaly Detection Opportunities:
- Negative stock (missing records, double-counting)
- Stock without production (ghost inventory)
- Stock aging past expiry (dispatch failure)
- High adjustments (theft, miscounts)
- Balance reconciliation failures
- Unexplained losses (shrinkage)

Key Insight: Inventory is the "final truth" - if numbers don't add up here, 
something upstream failed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Paths
DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports')
FIGURES_DIR = REPORTS_DIR / 'figures'
SUMMARIES_DIR = REPORTS_DIR / 'summaries'

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare():
    """
    Load Inventory dataset and prepare features.
    
    Returns:
        pd.DataFrame: Cleaned and feature-enriched dataframe
    """
    df = pd.read_parquet(DATA_DIR / 'inventory_stock_movements_dataset.parquet')
    logging.info(f"Loaded {len(df):,} inventory movement records")
    
    # ensure expiry_date is a datetime so downstream calculations work
    if 'expiry_date' in df.columns:
        df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
    
    # Handle missing timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Derive time features
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['month'] = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.month_name()
    
    # Derive location type
    df['location_type'] = df.apply(
        lambda row: 'Plant' if pd.notna(row.get('plant_id')) else 'Store' if pd.notna(row.get('store_id')) else 'Depot' if pd.notna(row.get('depot_id')) else 'Unknown',
        axis=1
    )
    df['location_id'] = df.apply(
        lambda row: row.get('plant_id') if pd.notna(row.get('plant_id')) else (row.get('store_id') if pd.notna(row.get('store_id')) else row.get('depot_id')),
        axis=1
    )
    
    # Reconciliation check: balance_after should = balance_before + qty_in - qty_out
    # Derive qty_in/qty_out from quantity_moved sign
    df['qty_in'] = df['quantity_moved'].where(df['quantity_moved'] > 0, 0)
    df['qty_out'] = df['quantity_moved'].where(df['quantity_moved'] < 0, 0).abs()
    df['net_movement'] = df['qty_in'] - df['qty_out']

    # Reconciliation check using balance_before/after if available
    if {'balance_before','balance_after'}.issubset(df.columns):
        df['calculated_balance'] = df['balance_before'] + df['net_movement']
        df['balance_mismatch'] = df['balance_after'] != df['calculated_balance']
        df['balance_diff'] = df['balance_after'] - df['calculated_balance']
    else:
        df['balance_mismatch'] = False
        df['balance_diff'] = 0
    
    # Net movement
    # Days to expiry
    df['days_to_expiry'] = (df['expiry_date'] - df['timestamp']).dt.days
    df['expired_flag'] = df['days_to_expiry'] < 0
    df['expiry_risk'] = df['days_to_expiry'].apply(
        lambda x: 'Expired' if x < 0 else 'Critical (0-2 days)' if x <= 2 else 'Warning (3-5 days)' if x <= 5 else 'Safe'
    )
    
    # Anomaly flags
    df['negative_balance_flag'] = df['balance_after'] < 0
    df['large_adjustment_flag'] = (df['movement_type'] == 'stock_adjustment') & (df['net_movement'].abs() > 100)
    
    logging.info(f"Derived features complete")
    logging.info(f"Negative balances: {df['negative_balance_flag'].sum():,} ({df['negative_balance_flag'].mean()*100:.1f}%)")
    logging.info(f"Balance mismatches: {df['balance_mismatch'].sum():,}")
    logging.info(f"Expired stock movements: {df['expired_flag'].sum():,}")
    
    return df


def summary_stats(df):
    """
    Generate comprehensive summary statistics for Inventory dataset.
    
    Focus Areas:
    - Overall stock movements (in/out/net)
    - Movement type breakdown
    - Balance reconciliation (anomaly detection)
    - Location-specific inventory (plant vs store)
    - SKU inventory levels and turnover
    - Expiry risk analysis
    - Shrinkage and adjustments
    - Data integrity issues
    """
    lines = []
    lines.append("=" * 80)
    lines.append("INVENTORY / STOCK MOVEMENTS - ENHANCED SUMMARY REPORT")
    lines.append("The State of the System: Final Reconciliation")
    lines.append("=" * 80)
    lines.append("")
    
    # === 1. OVERALL METRICS ===
    lines.append("=" * 80)
    lines.append("1. OVERALL INVENTORY MOVEMENT METRICS")
    lines.append("=" * 80)
    
    total_records = len(df)
    total_qty_in = df['qty_in'].sum()
    total_qty_out = df['qty_out'].sum()
    net_movement = total_qty_in - total_qty_out
    n_skus = df['sku'].nunique()
    n_locations = df['location_id'].nunique()
    
    lines.append(f"Total Movement Records: {total_records:,}")
    lines.append(f"Total Qty In (Production, Returns, Adjustments): {total_qty_in:,} units")
    lines.append(f"Total Qty Out (Dispatch, Sales, Waste): {total_qty_out:,} units")
    lines.append(f"Net Movement (In - Out): {net_movement:,} units")
    lines.append(f"Unique SKUs Tracked: {n_skus}")
    lines.append(f"Unique Locations (Plants + Stores): {n_locations}")
    lines.append(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    lines.append("")
    
    # === 2. MOVEMENT TYPE BREAKDOWN ===
    lines.append("=" * 80)
    lines.append("2. MOVEMENT TYPE BREAKDOWN")
    lines.append("=" * 80)
    
    movement_stats = df.groupby('movement_type').agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum'
    }).round(0)
    movement_stats.columns = ['Count', 'Qty_In', 'Qty_Out', 'Net_Movement']
    movement_stats = movement_stats.sort_values('Count', ascending=False)
    movement_stats['Pct'] = (movement_stats['Count'] / total_records * 100).round(1)
    
    lines.append(f"\nMovement Type Summary:")
    for movement_type, row in movement_stats.iterrows():
        lines.append(f"\n{movement_type.upper()}:")
        lines.append(f"   - Records: {row['Count']:,.0f} ({row['Pct']:.1f}%)")
        lines.append(f"   - Qty In: {row['Qty_In']:,.0f} | Qty Out: {row['Qty_Out']:,.0f}")
        lines.append(f"   - Net Movement: {row['Net_Movement']:,.0f}")
    
    lines.append(f"\n📊 Movement Distribution:")
    # Safe lookups depending on movement type labels in dataset
    for label, desc in [
        ('production','Production (stock creation)'),
        ('dispatch','Dispatch (stock moved)'),
        ('sale','Store Sales (stock consumed)'),
        ('return_from_store','Returns (stock recovered)'),
        ('waste','Waste (stock destroyed)'),
        ('stock_adjustment','Adjustments (corrections)'),
    ]:
        if label in movement_stats.index:
            lines.append(f"   - {desc}: {movement_stats.loc[label, 'Count']:,.0f} records")
    lines.append("")
    
    # === 3. DATA INTEGRITY & ANOMALIES ===
    lines.append("=" * 80)
    lines.append("3. DATA INTEGRITY & ANOMALY DETECTION")
    lines.append("=" * 80)
    
    negative_balance_count = df['negative_balance_flag'].sum()
    negative_balance_pct = negative_balance_count / total_records * 100
    balance_mismatch_count = df['balance_mismatch'].sum()
    large_adj_count = df['large_adjustment_flag'].sum()
    
    lines.append(f"\n🚨 CRITICAL ANOMALIES DETECTED:")
    lines.append(f"\n1. NEGATIVE BALANCES: {negative_balance_count:,} records ({negative_balance_pct:.1f}%)")
    if negative_balance_pct > 20:
        lines.append(f"   ⚠️  SEVERE DATA INTEGRITY ISSUE!")
        lines.append(f"   → Causes: Missing sales records, double-counted dispatch, unlogged waste")
        lines.append(f"   → Impact: Cannot trust inventory levels for {negative_balance_pct:.1f}% of movements")
        lines.append(f"   → Action: Urgent reconciliation with source datasets")
    
    # Negative balance by location
    neg_by_location = df[df['negative_balance_flag']].groupby('location_type')['record_id'].count()
    lines.append(f"\n   Negative Balance Distribution:")
    for loc_type, count in neg_by_location.items():
        lines.append(f"   - {loc_type}: {count:,} records")
    
    lines.append(f"\n2. BALANCE RECONCILIATION FAILURES: {balance_mismatch_count:,} records")
    if balance_mismatch_count > 0:
        lines.append(f"   → Formula: balance_after ≠ balance_before + qty_in - qty_out")
        lines.append(f"   → Indicates: Calculation errors or data corruption")
        lines.append(f"   → Action: Audit affected records")
    else:
        lines.append(f"   ✅ All balances reconcile correctly!")
    
    lines.append(f"\n3. LARGE STOCK ADJUSTMENTS: {large_adj_count:,} records (>100 units)")
    if large_adj_count > 0:
        large_adj_pct = large_adj_count / movement_stats.loc['stock_adjustment', 'Count'] * 100
        lines.append(f"   → {large_adj_pct:.1f}% of adjustments are large (>100 units)")
        lines.append(f"   → Potential causes: Theft, miscounts, reporting errors")
        lines.append(f"   → Action: Investigate reason_codes for large adjustments")
    lines.append("")
    
    # === 4. LOCATION-SPECIFIC INVENTORY ===
    lines.append("=" * 80)
    lines.append("4. LOCATION-SPECIFIC INVENTORY (PLANT vs STORE)")
    lines.append("=" * 80)
    
    location_stats = df.groupby('location_type').agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum',
        'balance_after': 'mean',
        'location_id': 'nunique'
    }).round(0)
    location_stats.columns = ['Records', 'Qty_In', 'Qty_Out', 'Net_Movement', 'Avg_Balance', 'Locations']
    
    lines.append(f"\nInventory by Location Type:")
    for loc_type, row in location_stats.iterrows():
        lines.append(f"\n{loc_type.upper()}:")
        lines.append(f"   - Movement Records: {row['Records']:,.0f}")
        lines.append(f"   - Qty In: {row['Qty_In']:,.0f} | Qty Out: {row['Qty_Out']:,.0f}")
        lines.append(f"   - Net Movement: {row['Net_Movement']:,.0f}")
        lines.append(f"   - Avg Balance: {row['Avg_Balance']:,.0f} units")
        lines.append(f"   - Number of {loc_type}s: {row['Locations']:.0f}")
    
    lines.append(f"\n📦 Inventory Flow Pattern:")
    plant_out = location_stats.loc['Plant', 'Qty_Out'] if 'Plant' in location_stats.index else 0
    store_in = location_stats.loc['Store', 'Qty_In'] if 'Store' in location_stats.index else 0
    lines.append(f"   - Plants Qty Out: {plant_out:,.0f} (production → dispatch)")
    lines.append(f"   - Stores Qty In: {store_in:,.0f} (dispatch → inventory)")
    if plant_out > 0:
        lines.append(f"   - Flow efficiency: {store_in/plant_out*100:.1f}% (should be ~100%)")
    lines.append("")
    
    # === 5. SKU INVENTORY ANALYSIS ===
    lines.append("=" * 80)
    lines.append("5. SKU INVENTORY LEVELS & TURNOVER")
    lines.append("=" * 80)
    
    # Get latest balance per SKU (last record for each SKU)
    latest_balance = df.sort_values('timestamp').groupby('sku').last()['balance_after']
    
    sku_stats = df.groupby('sku').agg({
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum',
        'record_id': 'count'
    }).round(0)
    sku_stats.columns = ['Qty_In', 'Qty_Out', 'Net_Movement', 'Movements']
    sku_stats['Latest_Balance'] = latest_balance
    sku_stats['Turnover_Ratio'] = (sku_stats['Qty_Out'] / (sku_stats['Qty_In'] + 1)).round(2)  # +1 to avoid div by 0
    sku_stats = sku_stats.sort_values('Latest_Balance', ascending=False)
    
    lines.append(f"\nTop 10 SKUs by Current Inventory Balance:")
    for idx, (sku, row) in enumerate(sku_stats.head(10).iterrows(), 1):
        lines.append(f"{idx}. {sku}:")
        lines.append(f"   - Current Balance: {row['Latest_Balance']:,.0f} units")
        lines.append(f"   - Total In: {row['Qty_In']:,.0f} | Out: {row['Qty_Out']:,.0f}")
        lines.append(f"   - Turnover Ratio: {row['Turnover_Ratio']:.2f} (Out/In)")
        lines.append(f"   - Movement Records: {row['Movements']:,.0f}")
    
    # Slow-moving SKUs (high balance, low turnover)
    slow_moving = sku_stats[(sku_stats['Latest_Balance'] > 1000) & (sku_stats['Turnover_Ratio'] < 0.5)]
    lines.append(f"\n⚠️  Slow-Moving SKUs (Balance >1000, Turnover <0.5): {len(slow_moving)}")
    if len(slow_moving) > 0:
        lines.append(f"   → Risk: Expiry, waste, tied-up capital")
        lines.append(f"   → Action: Reduce production, promote sales, discount aging stock")
        for sku, row in slow_moving.head(5).iterrows():
            lines.append(f"   - {sku}: Balance {row['Latest_Balance']:,.0f}, Turnover {row['Turnover_Ratio']:.2f}")
    lines.append("")
    
    # === 6. EXPIRY RISK ANALYSIS ===
    lines.append("=" * 80)
    lines.append("6. STOCK EXPIRY RISK ANALYSIS")
    lines.append("=" * 80)
    
    expiry_stats = df['expiry_risk'].value_counts()
    expired_count = df['expired_flag'].sum()
    expired_pct = expired_count / total_records * 100
    
    lines.append(f"\nExpiry Risk Distribution:")
    for risk_level, count in expiry_stats.items():
        pct = count / total_records * 100
        lines.append(f"   - {risk_level}: {count:,} records ({pct:.1f}%)")
    
    lines.append(f"\n🚨 Expired Stock Movements: {expired_count:,} ({expired_pct:.1f}%)")
    if expired_pct > 5:
        lines.append(f"   ⚠️  HIGH EXPIRY RATE: {expired_pct:.1f}% of movements involve expired stock")
        lines.append(f"   → Causes: Slow dispatch, overstocking, demand misforecasting")
        lines.append(f"   → Action: FIFO enforcement, reduce batch sizes, improve turnover")
    
    # Expired stock by movement type
    expired_by_type = df[df['expired_flag']].groupby('movement_type')['record_id'].count().sort_values(ascending=False)
    lines.append(f"\n   Expired Stock by Movement Type:")
    for movement_type, count in expired_by_type.items():
        lines.append(f"   - {movement_type}: {count:,} records")
    
    # Days to expiry stats
    lines.append(f"\n📅 Days to Expiry Statistics:")
    lines.append(f"   - Mean: {df['days_to_expiry'].mean():.1f} days")
    lines.append(f"   - Median: {df['days_to_expiry'].median():.1f} days")
    lines.append(f"   - 25th percentile: {df['days_to_expiry'].quantile(0.25):.1f} days")
    lines.append(f"   - 10th percentile: {df['days_to_expiry'].quantile(0.10):.1f} days")
    lines.append("")
    
    # === 7. SHRINKAGE & ADJUSTMENTS ===
    lines.append("=" * 80)
    lines.append("7. SHRINKAGE & STOCK ADJUSTMENTS")
    lines.append("=" * 80)
    
    adjustments = df[df['movement_type'] == 'stock_adjustment']
    total_adj_in = adjustments['qty_in'].sum()
    total_adj_out = adjustments['qty_out'].sum()
    net_adj = total_adj_in - total_adj_out
    
    lines.append(f"\nStock Adjustment Summary:")
    lines.append(f"   - Total Adjustment Records: {len(adjustments):,}")
    lines.append(f"   - Adjustments In: {total_adj_in:,} units (positive corrections)")
    lines.append(f"   - Adjustments Out: {total_adj_out:,} units (negative corrections)")
    lines.append(f"   - Net Adjustment: {net_adj:,} units")
    
    shrinkage_rate = abs(total_adj_out) / total_qty_in * 100 if total_qty_in > 0 else 0
    lines.append(f"\n📉 Shrinkage Rate: {shrinkage_rate:.2f}% of total inbound stock")
    if shrinkage_rate > 2:
        lines.append(f"   ⚠️  HIGH SHRINKAGE: {shrinkage_rate:.2f}% indicates significant losses")
        lines.append(f"   → Potential causes: Theft, miscounts, unrecorded waste, data errors")
        lines.append(f"   → Industry benchmark: <1% shrinkage")
        lines.append(f"   → Action: Inventory audits, security measures, process improvements")
    
    # Reason codes for adjustments
    if 'reason_code' in adjustments.columns:
        adj_reasons = adjustments['reason_code'].value_counts().head(5)
        lines.append(f"\n   Top 5 Adjustment Reasons:")
        for reason, count in adj_reasons.items():
            lines.append(f"   - {reason}: {count:,} records")
    lines.append("")
    
    # === 8. TEMPORAL PATTERNS ===
    lines.append("=" * 80)
    lines.append("8. TEMPORAL INVENTORY PATTERNS")
    lines.append("=" * 80)
    
    # Daily patterns
    daily_stats = df.groupby('date').agg({
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum',
        'balance_after': 'mean'
    })
    
    lines.append(f"\n📅 Daily Metrics:")
    lines.append(f"   - Average daily qty in: {daily_stats['qty_in'].mean():,.0f} units")
    lines.append(f"   - Average daily qty out: {daily_stats['qty_out'].mean():,.0f} units")
    lines.append(f"   - Average daily net movement: {daily_stats['net_movement'].mean():,.0f} units")
    lines.append(f"   - Average daily balance: {daily_stats['balance_after'].mean():,.0f} units")
    
    # Day of week patterns
    dow_stats = df.groupby('day_of_week')[['qty_in', 'qty_out']].sum()
    lines.append(f"\n📊 Day of Week Patterns:")
    lines.append(f"   - Highest in: {dow_stats['qty_in'].idxmax()} ({dow_stats['qty_in'].max():,.0f} units)")
    lines.append(f"   - Highest out: {dow_stats['qty_out'].idxmax()} ({dow_stats['qty_out'].max():,.0f} units)")
    lines.append("")
    
    # === 9. KEY INSIGHTS & ACTIONS ===
    lines.append("=" * 80)
    lines.append("9. KEY INSIGHTS & CRITICAL ACTION ITEMS")
    lines.append("=" * 80)
    
    lines.append("\n🎯 Critical Findings:")
    
    # Finding 1: Negative balances
    if negative_balance_pct > 20:
        lines.append(f"\n1. SEVERE DATA INTEGRITY CRISIS: {negative_balance_pct:.1f}% NEGATIVE BALANCES")
        lines.append(f"   → Impact: Cannot trust inventory system for production planning")
        lines.append(f"   → Root Cause: Likely missing records in sales/dispatch OR double-counting")
        lines.append(f"   → URGENT ACTION: Halt production decisions until reconciliation complete")
        lines.append(f"   → Fix: Cross-reference with Sales, Dispatch, Waste datasets")
    
    # Finding 2: Expiry risk
    if expired_pct > 5:
        lines.append(f"\n2. HIGH EXPIRY RATE: {expired_pct:.1f}% OF MOVEMENTS INVOLVE EXPIRED STOCK")
        lines.append(f"   → Impact: Waste risk, quality issues, customer dissatisfaction")
        lines.append(f"   → Root Cause: Slow inventory turnover, overstocking")
        lines.append(f"   → Action: FIFO enforcement, reduce batch sizes, improve dispatch speed")
    
    # Finding 3: Shrinkage
    if shrinkage_rate > 2:
        lines.append(f"\n3. HIGH SHRINKAGE RATE: {shrinkage_rate:.2f}%")
        lines.append(f"   → Impact: {abs(total_adj_out):,.0f} units unexplained loss")
        lines.append(f"   → Potential causes: Theft, miscounts, unrecorded waste")
        lines.append(f"   → Action: Inventory audits, security measures, process improvements")
    
    # Finding 4: Slow-moving SKUs
    if len(slow_moving) > 0:
        lines.append(f"\n4. SLOW-MOVING INVENTORY: {len(slow_moving)} SKUs AT RISK")
        lines.append(f"   → Risk: {slow_moving['Latest_Balance'].sum():,.0f} units tied up in slow-movers")
        lines.append(f"   → Impact: Capital tied up, expiry risk, warehouse space")
        lines.append(f"   → Action: Reduce production, promote sales, discount aging stock")
    
    # Finding 5: Balance mismatches
    if balance_mismatch_count > 0:
        lines.append(f"\n5. BALANCE RECONCILIATION FAILURES: {balance_mismatch_count:,} RECORDS")
        lines.append(f"   → Indicates: Calculation errors or data corruption")
        lines.append(f"   → Action: Audit affected records, fix data pipeline")
    
    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Write to file
    summary_path = REPORTS_DIR / 'inventory_enhanced_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logging.info(f"Wrote {summary_path}")
    
    return '\n'.join(lines)


def grouped_summaries(df):
    """
    Generate grouped summary CSVs for pivot analysis.
    """
    # 1. By Movement Type
    movement_summary = df.groupby('movement_type').agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum',
        'balance_after': 'mean'
    }).round(2)
    movement_summary.columns = ['Records', 'Qty_In', 'Qty_Out', 'Net_Movement', 'Avg_Balance']
    movement_summary.to_csv(SUMMARIES_DIR / 'inventory_by_movement_type.csv')
    logging.info("Wrote inventory_by_movement_type.csv")
    
    # 2. By Location
    location_summary = df.groupby(['location_type', 'location_id']).agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'balance_after': 'mean',
        'negative_balance_flag': 'sum'
    }).round(2)
    location_summary.columns = ['Records', 'Qty_In', 'Qty_Out', 'Avg_Balance', 'Negative_Balance_Count']
    location_summary = location_summary.sort_values('Records', ascending=False)
    location_summary.to_csv(SUMMARIES_DIR / 'inventory_by_location.csv')
    logging.info("Wrote inventory_by_location.csv")
    
    # 3. By SKU
    sku_summary = df.groupby('sku').agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum',
        'balance_after': ['mean', 'last'],
        'expired_flag': 'sum'
    }).round(2)
    sku_summary.columns = ['Records', 'Qty_In', 'Qty_Out', 'Net_Movement', 'Avg_Balance', 'Latest_Balance', 'Expired_Movements']
    sku_summary['Turnover_Ratio'] = (sku_summary['Qty_Out'] / (sku_summary['Qty_In'] + 1)).round(2)
    sku_summary = sku_summary.sort_values('Latest_Balance', ascending=False)
    sku_summary.to_csv(SUMMARIES_DIR / 'inventory_by_sku.csv')
    logging.info("Wrote inventory_by_sku.csv")
    
    # 4. By Date
    date_summary = df.groupby('date').agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'net_movement': 'sum',
        'balance_after': 'mean',
        'negative_balance_flag': 'sum'
    }).round(2)
    date_summary.columns = ['Records', 'Qty_In', 'Qty_Out', 'Net_Movement', 'Avg_Balance', 'Negative_Balance_Count']
    date_summary.to_csv(SUMMARIES_DIR / 'inventory_by_date.csv')
    logging.info("Wrote inventory_by_date.csv")
    
    # 5. Expiry Risk Summary
    expiry_summary = df.groupby('expiry_risk').agg({
        'record_id': 'count',
        'qty_in': 'sum',
        'qty_out': 'sum',
        'sku': 'nunique'
    }).round(2)
    expiry_summary.columns = ['Records', 'Qty_In', 'Qty_Out', 'SKUs_Affected']
    expiry_summary.to_csv(SUMMARIES_DIR / 'inventory_expiry_risk.csv')
    logging.info("Wrote inventory_expiry_risk.csv")
    
    # 6. Anomaly Summary
    anomaly_summary = df[df['negative_balance_flag']].groupby(['location_type', 'sku']).agg({
        'record_id': 'count',
        'balance_after': 'min'
    }).reset_index()
    anomaly_summary.columns = ['Location_Type', 'SKU', 'Negative_Balance_Count', 'Min_Balance']
    anomaly_summary = anomaly_summary.sort_values('Negative_Balance_Count', ascending=False).head(50)
    anomaly_summary.to_csv(SUMMARIES_DIR / 'inventory_anomalies_top50.csv', index=False)
    logging.info("Wrote inventory_anomalies_top50.csv")


def visualizations(df):
    """
    Generate 10+ comprehensive visualizations for Inventory dataset.
    """

    def add_caption(text):
        fig = plt.gcf()
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    # 1. Movement Type Distribution
    plt.figure(figsize=(10, 6))
    movement_counts = df['movement_type'].value_counts()
    colors = ['green' if x == 'production' else 'red' if x in ['waste', 'dispatch'] 
              else 'orange' if x == 'stock_adjustment' else 'steelblue' for x in movement_counts.index]
    bars = plt.bar(movement_counts.index, movement_counts.values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Movement Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Records', fontsize=12, fontweight='bold')
    plt.title('Inventory Movement Type Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    add_caption('Shows how often each movement type occurs across the inventory ledger.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_movement_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_movement_types.png")
    
    # 2. Balance Distribution (Histogram)
    plt.figure(figsize=(12, 6))
    # Filter extreme outliers for better visualization
    balance_filtered = df[df['balance_after'].between(df['balance_after'].quantile(0.01), 
                                                        df['balance_after'].quantile(0.99))]
    plt.hist(balance_filtered['balance_after'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Balance')
    plt.axvline(balance_filtered['balance_after'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f"Mean: {balance_filtered['balance_after'].mean():,.0f}")
    plt.xlabel('Balance After Movement', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Inventory Balance Distribution (Filtered for Outliers)', fontsize=14, fontweight='bold')
    plt.legend()
    add_caption('Highlights typical balance levels and the spread of stock positions.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_balance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_balance_distribution.png")
    
    # 3. Negative Balance Analysis
    plt.figure(figsize=(10, 6))
    neg_by_location = df[df['negative_balance_flag']].groupby('location_type')['record_id'].count()
    colors_neg = ['red', 'darkred']
    bars = plt.bar(neg_by_location.index, neg_by_location.values, color=colors_neg, alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Location Type', fontsize=12, fontweight='bold')
    plt.ylabel('Negative Balance Records', fontsize=12, fontweight='bold')
    plt.title('🚨 Negative Balance Anomalies by Location Type', fontsize=14, fontweight='bold', color='red')
    add_caption('Pinpoints where negative stock is most concentrated for reconciliation.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_negative_balances.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_negative_balances.png")
    
    # 4. Qty In vs Qty Out by Movement Type
    plt.figure(figsize=(12, 6))
    movement_flow = df.groupby('movement_type')[['qty_in', 'qty_out']].sum()
    
    x = np.arange(len(movement_flow.index))
    width = 0.35
    
    plt.bar(x - width/2, movement_flow['qty_in'], width, label='Qty In', color='green', alpha=0.7)
    plt.bar(x + width/2, movement_flow['qty_out'], width, label='Qty Out', color='red', alpha=0.7)
    
    plt.xlabel('Movement Type', fontsize=12, fontweight='bold')
    plt.ylabel('Quantity', fontsize=12, fontweight='bold')
    plt.title('Qty In vs Qty Out by Movement Type', fontsize=14, fontweight='bold')
    plt.xticks(x, movement_flow.index, rotation=45, ha='right')
    plt.legend()
    add_caption('Compares inbound vs outbound volume to validate expected flows.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_qty_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_qty_flow.png")
    
    # 5. SKU Balance (Top 15)
    plt.figure(figsize=(12, 8))
    latest_balance = df.sort_values('timestamp').groupby('sku').last()['balance_after'].sort_values(ascending=True).tail(15)
    colors_sku = ['green' if x > 5000 else 'orange' if x > 1000 else 'red' for x in latest_balance]
    latest_balance.plot(kind='barh', color=colors_sku, edgecolor='black')
    plt.xlabel('Current Balance (Units)', fontsize=12, fontweight='bold')
    plt.ylabel('SKU', fontsize=12, fontweight='bold')
    plt.title('Top 15 SKUs by Current Inventory Balance', fontsize=14, fontweight='bold')
    add_caption('Shows which SKUs hold the most stock and may drive carrying risk.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_sku_balances.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_sku_balances.png")
    
    # 6. Daily Inventory Trend
    plt.figure(figsize=(14, 6))
    daily_balance = df.groupby('date')['balance_after'].mean()
    
    plt.fill_between(daily_balance.index, daily_balance.values, alpha=0.3, color='steelblue')
    plt.plot(daily_balance.index, daily_balance.values, color='darkblue', linewidth=2, label='Avg Daily Balance')
    
    # 7-day moving average
    ma7 = daily_balance.rolling(window=7, center=True).mean()
    plt.plot(ma7.index, ma7.values, color='red', linewidth=2, linestyle='--', label='7-Day Moving Avg')
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Average Balance', fontsize=12, fontweight='bold')
    plt.title('Daily Inventory Balance Trend', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    add_caption('Tracks stock level stability and long-term drift.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_daily_trend.png")
    
    # 7. Expiry Risk Pie Chart
    plt.figure(figsize=(10, 10))
    expiry_counts = df['expiry_risk'].value_counts()
    colors_expiry = {'Expired': 'red', 'Critical (0-2 days)': 'orange', 
                     'Warning (3-5 days)': 'yellow', 'Safe': 'green'}
    colors_list = [colors_expiry.get(x, 'gray') for x in expiry_counts.index]
    
    plt.pie(expiry_counts.values, labels=expiry_counts.index, autopct='%1.1f%%',
            colors=colors_list, startangle=90, explode=[0.05 if x == 'Expired' else 0 for x in expiry_counts.index])
    plt.title('Stock Expiry Risk Distribution', fontsize=14, fontweight='bold')
    add_caption('Quantifies how much stock is safe vs near or past expiry.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_expiry_risk_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_expiry_risk_pie.png")
    
    # 8. Days to Expiry Distribution
    plt.figure(figsize=(12, 6))
    days_filtered = df[df['days_to_expiry'].between(-10, 30)]  # Focus on critical range
    plt.hist(days_filtered['days_to_expiry'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Expiry Date')
    plt.axvline(5, color='orange', linestyle='--', linewidth=2, label='5 Days Warning')
    plt.xlabel('Days to Expiry', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Days to Expiry Distribution (-10 to +30 days)', fontsize=14, fontweight='bold')
    plt.legend()
    add_caption('Shows how close current stock is to expiry thresholds.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_days_to_expiry.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_days_to_expiry.png")
    
    # 9. Location Type Comparison (Plant vs Store)
    plt.figure(figsize=(10, 6))
    location_comparison = df.groupby('location_type')[['qty_in', 'qty_out']].sum()
    
    x = np.arange(len(location_comparison.index))
    width = 0.35
    
    plt.bar(x - width/2, location_comparison['qty_in'], width, label='Qty In', color='green', alpha=0.7)
    plt.bar(x + width/2, location_comparison['qty_out'], width, label='Qty Out', color='red', alpha=0.7)
    
    plt.xlabel('Location Type', fontsize=12, fontweight='bold')
    plt.ylabel('Total Quantity', fontsize=12, fontweight='bold')
    plt.title('Inventory Flow: Plant vs Store', fontsize=14, fontweight='bold')
    plt.xticks(x, location_comparison.index)
    plt.legend()
    add_caption('Compares inbound and outbound volumes by location type.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_plant_vs_store.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_plant_vs_store.png")
    
    # 10. Stock Adjustment Distribution
    adjustments = df[df['movement_type'] == 'stock_adjustment']
    plt.figure(figsize=(10, 6))
    plt.hist(adjustments['net_movement'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Adjustment')
    plt.xlabel('Net Adjustment (Qty In - Qty Out)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Stock Adjustment Distribution (Shrinkage Analysis)', fontsize=14, fontweight='bold')
    plt.legend()
    add_caption('Highlights adjustment size distribution to spot unusual corrections.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_adjustments.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_adjustments.png")
    
    # 11. Turnover Ratio by SKU (Top 15)
    plt.figure(figsize=(12, 8))
    sku_turnover = df.groupby('sku').agg({
        'qty_in': 'sum',
        'qty_out': 'sum'
    })
    sku_turnover['turnover'] = (sku_turnover['qty_out'] / (sku_turnover['qty_in'] + 1)).round(2)
    sku_turnover = sku_turnover.sort_values('turnover', ascending=True).tail(15)
    
    colors_turnover = ['green' if x > 0.8 else 'orange' if x > 0.5 else 'red' for x in sku_turnover['turnover']]
    sku_turnover['turnover'].plot(kind='barh', color=colors_turnover, edgecolor='black')
    plt.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='Perfect Turnover (1.0)')
    plt.xlabel('Turnover Ratio (Out / In)', fontsize=12, fontweight='bold')
    plt.ylabel('SKU', fontsize=12, fontweight='bold')
    plt.title('Top 15 SKUs by Inventory Turnover Ratio', fontsize=14, fontweight='bold')
    plt.legend()
    add_caption('Higher turnover indicates faster movement and lower holding risk.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_turnover_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_turnover_ratio.png")
    
    # 12. Net Movement by Day of Week
    plt.figure(figsize=(10, 6))
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_net = df.groupby('day_of_week')['net_movement'].sum().reindex(dow_order)
    
    colors_dow = ['green' if x > 0 else 'red' for x in dow_net]
    bars = plt.bar(dow_net.index, dow_net.values, color=colors_dow, alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
    plt.ylabel('Net Movement (In - Out)', fontsize=12, fontweight='bold')
    plt.title('Net Inventory Movement by Day of Week', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    add_caption('Shows weekly rhythm of stock accumulation vs depletion.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'inventory_net_movement_dow.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved inventory_net_movement_dow.png")

    # 13. Balance reconciliation mismatches by movement type
    if 'balance_mismatch' in df.columns:
        plt.figure(figsize=(10, 6))
        mismatch_counts = df[df['balance_mismatch']].groupby('movement_type')['record_id'].count()
        if not mismatch_counts.empty:
            mismatch_counts = mismatch_counts.sort_values(ascending=False)
            plt.bar(mismatch_counts.index, mismatch_counts.values, color='crimson', alpha=0.7, edgecolor='black')
            plt.xlabel('Movement Type', fontsize=12, fontweight='bold')
            plt.ylabel('Mismatch Records', fontsize=12, fontweight='bold')
            plt.title('Balance Reconciliation Mismatches by Movement Type', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            add_caption('Highlights where balance math fails most often for data repair focus.')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / 'inventory_balance_mismatch_by_type.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved inventory_balance_mismatch_by_type.png")
        else:
            plt.close()

    # 14. Negative balance trend over time (daily counts)
    plt.figure(figsize=(12, 5))
    neg_daily = df[df['negative_balance_flag']].groupby('date')['record_id'].count()
    if not neg_daily.empty:
        plt.plot(neg_daily.index, neg_daily.values, color='red', linewidth=2)
        plt.fill_between(neg_daily.index, neg_daily.values, alpha=0.2, color='red')
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Negative Balance Records', fontsize=12, fontweight='bold')
        plt.title('Daily Negative Balance Incidents', fontsize=14, fontweight='bold')
        add_caption('Tracks spikes in negative stock to locate issue windows.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'inventory_negative_balance_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved inventory_negative_balance_trend.png")
    else:
        plt.close()

    # 15. Expiry risk by location type
    plt.figure(figsize=(10, 6))
    risk_by_loc = df.groupby(['location_type', 'expiry_risk'])['record_id'].count().unstack(fill_value=0)
    if not risk_by_loc.empty:
        risk_by_loc.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.xlabel('Location Type', fontsize=12, fontweight='bold')
        plt.ylabel('Records', fontsize=12, fontweight='bold')
        plt.title('Expiry Risk by Location Type', fontsize=14, fontweight='bold')
        add_caption('Shows where expiry risk is concentrated across the network.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'inventory_expiry_risk_by_location.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved inventory_expiry_risk_by_location.png")
    else:
        plt.close()


def main():
    """
    Main execution function.
    """
    logging.info("=" * 80)
    logging.info("Starting Inventory / Stock Movements Enhanced EDA")
    logging.info("=" * 80)
    
    # Load and prepare data
    df = load_and_prepare()
    
    # Generate summary statistics
    summary_stats(df)
    
    # Generate grouped summaries
    grouped_summaries(df)
    
    # Generate visualizations
    visualizations(df)
    
    logging.info("=" * 80)
    logging.info("✅ Inventory EDA complete!")
    logging.info(f"   - Summary: reports/inventory_enhanced_summary.txt")
    logging.info(f"   - Figures: reports/figures/inventory_*.png (12 visualizations)")
    logging.info(f"   - CSVs: reports/summaries/inventory_*.csv (6 summary files)")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
