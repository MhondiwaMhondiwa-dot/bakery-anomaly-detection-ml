"""
Exploratory Data Analysis for Returns Dataset

This script analyzes products returned from retailers to bakery/depot.
Returns signal downstream failure - products delivered but NOT converted to sales.
Critical for understanding demand mismatches, quality issues, and logistics problems.

Key Analyses:
- Return volume and rates
- Root cause analysis (return reasons)
- Route and retailer performance
- SKU-level return patterns
- Temperature correlation with returns
- Handling condition impact
- Temporal patterns (time-to-return, seasonal trends)
- Linkage to dispatch and quality issues

Author: Baker's Inn Analytics Team
Date: 2025
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(message)s'
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
SUMMARIES_DIR = REPORTS_DIR / 'summaries'

# Create output directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare():
    """
    Load returns dataset and prepare derived fields.
    
    Returns:
        pd.DataFrame: Returns data with derived fields
    """
    df = pd.read_parquet(DATA_DIR / 'returns_dataset.parquet')
    logging.info(f"Loaded {len(df):,} return records")
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Compute total quantity returned from multi-SKU columns
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    sku_cols_present = [c for c in sku_cols if c in df.columns]
    if sku_cols_present:
        df['qty_returned'] = df[sku_cols_present].sum(axis=1)
    else:
        df['qty_returned'] = 0
    
    # Derive time-based features
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    return df

def summary_stats(df):
    """
    Generate comprehensive summary statistics for returns dataset.
    
    Args:
        df: Returns DataFrame
    """
    summary_path = REPORTS_DIR / 'returns_enhanced_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RETURNS DATASET - EXPLORATORY DATA ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset overview
        f.write("↩️ DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Return Records: {len(df):,}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Total Units Returned: {df['qty_returned'].sum():,}\n")
        f.write(f"Average Return Size: {df['qty_returned'].mean():.1f} units\n")
        f.write(f"Median Return: {df['qty_returned'].median():.0f} units\n")
        f.write(f"Routes Affected: {df['route_id'].nunique()}\n")
        f.write(f"Retailers Returning: {df['retailer_id'].nunique()}\n")
        sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
        f.write(f"SKUs Returned (Multi-SKU): {len([c for c in sku_cols if c in df.columns])}\n\n")
        
        # Root cause analysis
        f.write("🔍 RETURN ROOT CAUSE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        reason_returns = df.groupby('reason_code').agg({
            'qty_returned': ['sum', 'mean', 'count']
        }).sort_values(('qty_returned', 'sum'), ascending=False)
        reason_returns.columns = ['Total Units', 'Avg Units/Return', 'Incidents']
        reason_returns['% of Total'] = (reason_returns['Total Units'] / df['qty_returned'].sum() * 100).round(2)
        f.write(f"{reason_returns}\n\n")
        
        top_reason = reason_returns.index[0]
        top_reason_pct = reason_returns.iloc[0]['% of Total']
        top_reason_units = reason_returns.iloc[0]['Total Units']
        f.write(f"🎯 **TOP RETURN REASON:** {top_reason}\n")
        f.write(f"   {top_reason_units:,.0f} units ({top_reason_pct:.1f}% of all returns)\n")
        f.write(f"   Priority action: {'Improve demand forecasting' if 'Unsold' in top_reason else 'Address quality/logistics issues'}\n\n")
        
        # Identify preventable vs non-preventable returns
        expired_damaged = df[df['reason_code'].str.contains('Expired|Damaged|Crushed|Mould', case=False, na=False)]
        unsold = df[df['reason_code'].str.contains('Unsold', case=False, na=False)]
        
        f.write(f"Preventable Returns (Expired/Damaged/Crushed/Mould): {expired_damaged['qty_returned'].sum():,} units ({expired_damaged['qty_returned'].sum()/df['qty_returned'].sum()*100:.1f}%)\n")
        f.write(f"Demand Mismatch (Returned Unsold): {unsold['qty_returned'].sum():,} units ({unsold['qty_returned'].sum()/df['qty_returned'].sum()*100:.1f}%)\n\n")
        
        # Initialize variables for worst performers
        worst_route = "N/A"
        worst_route_returns = 0
        worst_retailer = "N/A"
        worst_retailer_returns = 0
        
        # Route-level returns
        f.write("🚛 ROUTE-LEVEL RETURN ANALYSIS (Top 15)\n")
        f.write("-" * 80 + "\n")
        route_returns = df.groupby('route_id').agg({
            'qty_returned': ['sum', 'mean', 'count'],
            'retailer_id': 'nunique'
        }).sort_values(('qty_returned', 'sum'), ascending=False).head(15)
        route_returns.columns = ['Total Returned', 'Avg per Return', 'Incidents', 'Retailers']
        route_returns['% of Total'] = (route_returns['Total Returned'] / df['qty_returned'].sum() * 100).round(2)
        f.write(f"{route_returns}\n\n")
        
        if not route_returns.empty:
            worst_route = route_returns.index[0]
            worst_route_returns = route_returns.iloc[0]['Total Returned']
            f.write(f"⚠️ **WORST PERFORMING ROUTE:** {worst_route} ({worst_route_returns:,.0f} units returned)\n")
            f.write("   Action: Investigate dispatch timing, cold chain, demand forecasting for this route\n\n")
        else:
            f.write("⚠️ No route-level return data available; skipping route analysis\n\n")
        
        # Retailer-level returns
        f.write("🏪 RETAILER-LEVEL RETURN ANALYSIS (Top 15)\n")
        f.write("-" * 80 + "\n")
        retailer_returns = df.groupby('retailer_id').agg({
            'qty_returned': ['sum', 'mean', 'count']
        }).sort_values(('qty_returned', 'sum'), ascending=False)
        if retailer_returns.empty:
            f.write("No valid retailer_id values present; retailer-level analysis skipped.\n\n")
        else:
            retailer_returns = retailer_returns.head(15)
            retailer_returns.columns = ['Total Returned', 'Avg per Return', 'Incidents']
            retailer_returns['% of Total'] = (retailer_returns['Total Returned'] / df['qty_returned'].sum() * 100).round(2)
            f.write(f"{retailer_returns}\n\n")
            worst_retailer = retailer_returns.index[0]
            worst_retailer_returns = retailer_returns.iloc[0]['Total Returned']
            f.write(f"⚠️ **WORST PERFORMING RETAILER:** {worst_retailer} ({worst_retailer_returns:,.0f} units returned)\n")
            f.write("   Action: Review storage conditions, demand patterns, or consider reducing dispatch volumes\n\n")
        
        # SKU-level returns (multi-SKU columns)
        f.write("🍞 SKU-LEVEL RETURN ANALYSIS\n")
        f.write("-" * 80 + "\n")
        sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
        sku_cols_present = [c for c in sku_cols if c in df.columns]
        if sku_cols_present:
            sku_totals = df[sku_cols_present].sum()
            f.write("SKU Totals Returned:\n")
            for sku, qty in sku_totals.sort_values(ascending=False).items():
                pct = qty / df['qty_returned'].sum() * 100 if df['qty_returned'].sum() > 0 else 0
                f.write(f"  {sku}: {qty:,.0f} units ({pct:.1f}%)\n")
            f.write("\n")
        
        # Temperature analysis (optional if column exists)
        if 'temperature_at_return' in df.columns:
            f.write("🌡️ TEMPERATURE CORRELATION WITH RETURNS\n")
            f.write("-" * 80 + "\n")
            temp_stats = df['temperature_at_return'].describe()
            f.write(f"{temp_stats}\n\n")
            
            # High temperature returns
            high_temp_threshold = 35  # Celsius
            high_temp_returns = df[df['temperature_at_return'] > high_temp_threshold]
            f.write(f"Returns with temp > {high_temp_threshold}°C: {len(high_temp_returns):,} ({len(high_temp_returns)/len(df)*100:.1f}%)\n")
            f.write(f"Units returned at high temp: {high_temp_returns['qty_returned'].sum():,} ({high_temp_returns['qty_returned'].sum()/df['qty_returned'].sum()*100:.1f}% of total)\n\n")
        
        # Handling condition (optional if column exists)
        if 'handling_condition' in df.columns:
            f.write("🤲 HANDLING CONDITION IMPACT\n")
            f.write("-" * 80 + "\n")
            handling_returns = df.groupby('handling_condition').agg({
                'qty_returned': ['sum', 'count']
            }).sort_values(('qty_returned', 'sum'), ascending=False)
            handling_returns.columns = ['Total Returned', 'Incidents']
            handling_returns['% of Total'] = (handling_returns['Total Returned'] / df['qty_returned'].sum() * 100).round(2)
            f.write(f"{handling_returns}\n\n")
        
        
        # Temporal patterns
        f.write("📅 TEMPORAL RETURN PATTERNS\n")
        f.write("-" * 80 + "\n")
        
        # Day of week
        dow_returns = df.groupby('day_name').agg({
            'qty_returned': ['sum', 'mean', 'count']
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        dow_returns.columns = ['Total', 'Avg', 'Incidents']
        f.write("Returns by Day of Week:\n")
        f.write(f"{dow_returns}\n\n")
        
        peak_day = dow_returns['Total'].idxmax()
        f.write(f"Peak Return Day: {peak_day} ({dow_returns.loc[peak_day, 'Total']:.0f} units)\n\n")
        
        # Weekend vs weekday
        weekend_returns = df.groupby('is_weekend').agg({
            'qty_returned': ['sum', 'mean']
        })
        weekend_returns.index = ['Weekday', 'Weekend']
        f.write("Weekday vs Weekend:\n")
        f.write(f"{weekend_returns}\n\n")
        
        # Monthly pattern
        monthly_returns = df.groupby('month')['qty_returned'].sum()
        f.write("Returns by Month:\n")
        f.write(f"{monthly_returns}\n\n")
        
        # ACTION ITEMS
        f.write("🎯 KEY INSIGHTS & ACTION ITEMS\n")
        f.write("=" * 80 + "\n")
        
        f.write(f"1. **Total returns:** {df['qty_returned'].sum():,} units across {len(df):,} incidents\n")
        f.write(f"   Impact: Returns often lead to waste - critical to prevent\n\n")
        
        f.write(f"2. **Top return reason:** {top_reason} ({top_reason_pct:.1f}%)\n")
        f.write(f"   Action: {'Improve demand forecasting and reduce dispatch volumes' if 'Unsold' in top_reason else 'Address quality control and cold chain'}\n\n")
        
        f.write(f"3. **Worst route:** {worst_route} ({worst_route_returns:,.0f} units)\n")
        f.write(f"   Action: Route-specific investigation - dispatch timing, vehicle condition, demand patterns\n\n")
        
        f.write(f"4. **Worst retailer:** {worst_retailer} ({worst_retailer_returns:,.0f} units)\n")
        f.write(f"   Action: Retailer partnership review - storage, forecasting, or reduce allocation\n\n")
        
        if 'temperature_at_return' in df.columns:
            high_temp_threshold = 35
            high_temp_returns = df[df['temperature_at_return'] > high_temp_threshold]
            f.write(f"5. **High-temperature returns:** {high_temp_returns['qty_returned'].sum():,} units at >{high_temp_threshold}°C\n")
            f.write(f"   Action: Cold chain monitoring, refrigeration at retailers\n\n")
        
        f.write(f"6. **Preventable returns:** {expired_damaged['qty_returned'].sum():,} units (quality/logistics issues)\n")
        f.write(f"   Action: QC improvement, dispatch optimization, packaging reinforcement\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("✅ Returns summary complete!\n")
    
    logging.info(f"Wrote {summary_path}")

def grouped_summaries(df):
    """
    Generate grouped summary CSV files.
    
    Args:
        df: Returns DataFrame
    """
    # 1. Returns by reason
    reason_summary = df.groupby('reason_code').agg({
        'qty_returned': ['sum', 'mean', 'count']
    }).round(2)
    reason_summary.columns = ['_'.join(col).strip() for col in reason_summary.columns]
    reason_summary = reason_summary.sort_values('qty_returned_sum', ascending=False)
    reason_summary.to_csv(SUMMARIES_DIR / 'returns_by_reason.csv')
    logging.info("Wrote returns_by_reason.csv")
    
    # 2. Returns by route (top 30)
    # 1. Returns by reason code
    reason_summary = df.groupby('reason_code').agg({
        'qty_returned': ['sum', 'count']
    }).round(2)
    reason_summary.columns = ['_'.join(col).strip() for col in reason_summary.columns]
    reason_summary = reason_summary.sort_values('qty_returned_sum', ascending=False)
    reason_summary.to_csv(SUMMARIES_DIR / 'returns_by_reason.csv')
    logging.info("Wrote returns_by_reason.csv")
    
    # 2. Returns by route (top 30)
    route_summary = df.groupby('route_id').agg({
        'qty_returned': ['sum', 'mean', 'count'],
        'retailer_id': 'nunique'
    }).round(2)
    route_summary.columns = ['_'.join(col).strip() for col in route_summary.columns]
    route_summary = route_summary.sort_values('qty_returned_sum', ascending=False).head(30)
    route_summary.to_csv(SUMMARIES_DIR / 'returns_by_route_top30.csv')
    logging.info("Wrote returns_by_route_top30.csv")
    
    # 3. Returns by retailer (top 30)
    retailer_summary = df.groupby('retailer_id').agg({
        'qty_returned': ['sum', 'mean', 'count']
    }).round(2)
    retailer_summary.columns = ['_'.join(col).strip() for col in retailer_summary.columns]
    retailer_summary = retailer_summary.sort_values('qty_returned_sum', ascending=False).head(30)
    retailer_summary.to_csv(SUMMARIES_DIR / 'returns_by_retailer_top30.csv')
    logging.info("Wrote returns_by_retailer_top30.csv")
    
    # 4. Returns by SKU (multi-column)
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    sku_cols_present = [c for c in sku_cols if c in df.columns]
    if sku_cols_present:
        sku_totals = df[sku_cols_present].sum().to_frame('total_qty_returned')
        sku_totals['pct_of_total'] = sku_totals['total_qty_returned'] / sku_totals['total_qty_returned'].sum() * 100
        sku_totals = sku_totals.sort_values('total_qty_returned', ascending=False)
        sku_totals.to_csv(SUMMARIES_DIR / 'returns_by_sku.csv')
        logging.info("Wrote returns_by_sku.csv")
    
    # 5. Returns by handling condition (if exists)
    if 'handling_condition' in df.columns:
        handling_summary = df.groupby('handling_condition').agg({
            'qty_returned': ['sum', 'mean', 'count']
        }).round(2)
        handling_summary.columns = ['_'.join(col).strip() for col in handling_summary.columns]
        handling_summary = handling_summary.sort_values('qty_returned_sum', ascending=False)
        handling_summary.to_csv(SUMMARIES_DIR / 'returns_by_handling.csv')
        logging.info("Wrote returns_by_handling.csv")
    
    # 6. Daily returns
    daily_summary = df.groupby('date').agg({
        'qty_returned': 'sum',
        'return_id': 'count'
    }).round(2)
    daily_summary.columns = ['total_returned', 'incidents']
    daily_summary.to_csv(SUMMARIES_DIR / 'returns_by_date.csv')
    logging.info("Wrote returns_by_date.csv")

def visualizations(df):
    """
    Generate comprehensive visualizations for returns dataset.
    
    Args:
        df: Returns DataFrame
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Returns by reason
    fig, ax = plt.subplots(figsize=(12, 7))
    reason_returns = df.groupby('reason_code')['qty_returned'].sum().sort_values(ascending=True)
    colors = ['darkred' if x > reason_returns.median() * 1.5 else 'coral' if x > reason_returns.median() else 'gold' for x in reason_returns]
    reason_returns.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Return Volume by Reason', fontsize=16, fontweight='bold')
    ax.set_xlabel('Units Returned', fontsize=12)
    ax.set_ylabel('Return Reason', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'returns_by_reason.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_by_reason.png")
    
    # 2. Top 15 routes by returns
    fig, ax = plt.subplots(figsize=(12, 8))
    route_returns = df.groupby('route_id')['qty_returned'].sum().sort_values(ascending=True).tail(15)
    route_returns.plot(kind='barh', ax=ax, color='darkred')
    ax.set_title('Top 15 Routes by Return Volume', fontsize=16, fontweight='bold')
    ax.set_xlabel('Units Returned', fontsize=12)
    ax.set_ylabel('Route ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'returns_by_route_top15.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_by_route_top15.png")
    
    # 3. Top 15 retailers by returns
    retailer_returns = df.groupby('retailer_id')['qty_returned'].sum().sort_values(ascending=True).tail(15)
    if not retailer_returns.empty and retailer_returns.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        retailer_returns.plot(kind='barh', ax=ax, color='darkred')
        ax.set_title('Top 15 Retailers by Return Volume', fontsize=16, fontweight='bold')
        ax.set_xlabel('Units Returned', fontsize=12)
        ax.set_ylabel('Retailer ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'returns_by_retailer_top15.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved returns_by_retailer_top15.png")
    else:
        logging.info("Skipped returns_by_retailer_top15.png (no valid retailer data)")
    
    # 4. Returns by SKU (multi-column)
    fig, ax = plt.subplots(figsize=(12, 7))
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    sku_cols_present = [c for c in sku_cols if c in df.columns]
    if sku_cols_present:
        sku_returns = df[sku_cols_present].sum().sort_values(ascending=True)
        colors = ['red' if x > sku_returns.median() * 1.5 else 'orange' for x in sku_returns]
        sku_returns.plot(kind='barh', ax=ax, color=colors)
        ax.set_title('Return Volume by SKU', fontsize=16, fontweight='bold')
        ax.set_xlabel('Units Returned', fontsize=12)
        ax.set_ylabel('SKU', fontsize=12)
        ax.axvline(sku_returns.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {sku_returns.median():.0f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'returns_by_sku.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved returns_by_sku.png")
    
    # 5. Daily returns trend
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_returns = df.groupby('date')['qty_returned'].sum().reset_index()
    daily_returns['date'] = pd.to_datetime(daily_returns['date'])
    
    ax.plot(daily_returns['date'], daily_returns['qty_returned'], linewidth=2, color='darkred', alpha=0.7)
    ax.fill_between(daily_returns['date'], daily_returns['qty_returned'], alpha=0.3, color='salmon')
    
    # 7-day moving average
    daily_returns['ma7'] = daily_returns['qty_returned'].rolling(window=7, center=True).mean()
    ax.plot(daily_returns['date'], daily_returns['ma7'], linewidth=3, color='darkblue', label='7-Day Moving Avg')
    
    ax.set_title('Daily Returns Trend', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Units Returned', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'returns_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_daily_trend.png")
    
    # 6. Day of week pattern
    fig, ax = plt.subplots(figsize=(12, 6))
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_returns = df.groupby('day_name')['qty_returned'].sum().reindex(dow_order)
    colors_dow = ['lightcoral' if day not in ['Saturday', 'Sunday'] else 'darkred' for day in dow_order]
    ax.bar(dow_order, dow_returns.values, color=colors_dow)
    ax.set_title('Returns by Day of Week', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Total Units Returned', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(dow_returns.values):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'returns_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_day_of_week.png")
    
    # 7. Temperature distribution (if column exists)
    if 'temperature_at_return' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df['temperature_at_return'].dropna(), bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(35, color='red', linestyle='--', linewidth=2, label='High Temp Threshold (35°C)')
        ax.set_title('Temperature Distribution at Return', fontsize=16, fontweight='bold')
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'returns_temperature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved returns_temperature_distribution.png")
    
    # 8. Handling condition impact (if column exists)
    if 'handling_condition' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        handling_returns = df.groupby('handling_condition')['qty_returned'].sum().sort_values(ascending=False)
        handling_returns.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Returns by Handling Condition', fontsize=16, fontweight='bold')
        ax.set_xlabel('Handling Condition', fontsize=12)
        ax.set_ylabel('Units Returned', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(handling_returns.values):
            ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'returns_by_handling_condition.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved returns_by_handling_condition.png")
    
    # 9. Return quantity distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['qty_returned'], bins=50, color='darkred', alpha=0.7, edgecolor='black')
    ax.axvline(df['qty_returned'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["qty_returned"].mean():.1f}')
    ax.axvline(df['qty_returned'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["qty_returned"].median():.0f}')
    ax.set_title('Return Quantity Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Units Returned per Incident', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'returns_quantity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_quantity_distribution.png")
    
    # 10. Reason breakdown pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    reason_returns = df.groupby('reason_code')['qty_returned'].sum().sort_values(ascending=False).head(6)
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    ax.pie(reason_returns, labels=reason_returns.index, autopct='%1.1f%%', startangle=90,
           colors=colors, textprops={'fontsize': 10})
    ax.set_title('Return Reasons Distribution (Top 6)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'returns_reason_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved returns_reason_pie.png")

def main():
    """
    Main execution function for Returns EDA.
    """
    # Load and prepare data
    df = load_and_prepare()
    
    # Generate summary statistics
    summary_stats(df)
    
    # Generate grouped summaries
    grouped_summaries(df)
    
    # Generate visualizations
    visualizations(df)
    
    logging.info("✅ Returns EDA complete!")

if __name__ == '__main__':
    main()
