"""
Enhanced EDA for Sales Dataset (B2B Channel) - Wholesale/Depot Distribution

Business Context:
================
The Sales Dataset captures B2B wholesale transactions where depots distribute products 
to stores/retailers. This is distinct from Sales POS (B2C retail transactions):

- Sales Dataset (THIS) = B2B channel: Depot → Store (wholesale orders)
- Sales POS = B2C channel: Retailer → Consumer (retail transactions)

Key Business Questions:
1. How do depot-to-store orders differ from retail POS patterns?
2. Which depots are highest volume distributors?
3. Are wholesale order sizes larger than retail transactions?
4. Do stores order different SKU mixes than retail POS demand?
5. Which routes connect depots to stores most efficiently?
6. Is wholesale pricing lower than retail (expected margin structure)?

Critical Links:
- Depots distribute to Stores (this dataset)
- Stores then sell to consumers (Sales POS dataset)
- Can correlate: Depot orders → Store POS sales for inventory optimization
- Route performance impacts both depot distribution and store freshness

Analysis Focus:
- Depot performance (volume, SKU mix, store coverage)
- Store ordering patterns (frequency, volume, SKU preferences)
- Route efficiency (which routes serve which depot-store pairs)
- Pricing analysis (wholesale vs retail margin validation)
- Temporal patterns (B2B ordering behavior vs B2C demand patterns)
- Depot-Store-Route network optimization opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

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
    Load Sales Dataset (B2B channel) and prepare time-based features.
    
    Returns:
        pd.DataFrame: Cleaned and feature-enriched dataframe
    """
    df = pd.read_parquet(DATA_DIR / 'sales_dataset.parquet')
    logging.info(f"Loaded {len(df):,} B2B sales records")

    # The updated aggregated sales schema includes date, plant_id, depot_id, region, sku, quantity_sold, bakers_inn_price
    # Derive time features from 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()

    # Derive revenue using Bakers Inn price
    price_col = 'bakers_inn_price' if 'bakers_inn_price' in df.columns else None
    if price_col:
        df['revenue'] = df['quantity_sold'] * df[price_col]
    else:
        df['revenue'] = df['quantity_sold']

    # Clean depot_id and region
    df['depot_id'] = df['depot_id'].fillna('UNKNOWN')
    df['region'] = df.get('region').fillna('UNKNOWN') if 'region' in df.columns else 'UNKNOWN'

    logging.info(f"Final dataset: {len(df):,} records with {df['quantity_sold'].sum():,} units sold")
    logging.info(f"Total revenue: ${df['revenue'].sum():,.2f}")

    return df


def summary_stats(df):
    """
    Generate comprehensive summary statistics for B2B Sales Dataset.
    
    Focus Areas:
    - Depot distribution performance
    - Store ordering patterns
    - Route efficiency
    - SKU demand by depot/store
    - Pricing structure (wholesale vs retail validation)
    - Temporal ordering patterns
    - Network optimization opportunities (depot-route-store)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SALES DATASET (B2B CHANNEL) - ENHANCED SUMMARY REPORT")
    lines.append("Analysis Period: Wholesale/Depot Distribution to Stores")
    lines.append("=" * 80)
    lines.append("")
    
    # === 1. OVERALL METRICS ===
    lines.append("=" * 80)
    lines.append("1. OVERALL B2B SALES METRICS")
    lines.append("=" * 80)
    
    total_records = len(df)
    total_units = df['quantity_sold'].sum()
    total_revenue = df['revenue'].sum()
    n_depots = df['depot_id'].nunique()
    n_skus = df['sku'].nunique()
    price_col = 'bakers_inn_price' if 'bakers_inn_price' in df.columns else None
    
    lines.append(f"Total B2B Orders: {total_records:,}")
    lines.append(f"Total Units Distributed: {total_units:,}")
    lines.append(f"Total Revenue (Wholesale): ${total_revenue:,.2f}")
    if price_col:
        lines.append(f"Average Wholesale Price: ${df[price_col].mean():.2f}/unit")
    lines.append("")
    
    # === 2. DEPOT PERFORMANCE ===
    lines.append("=" * 80)
    lines.append("2. DEPOT DISTRIBUTION PERFORMANCE")
    lines.append("=" * 80)
    
    depot_stats = df.groupby('depot_id').agg({
        'quantity_sold': ['sum', 'count'],
        'revenue': 'sum'
    }).round(2)
    depot_stats.columns = ['Total_Units', 'Orders', 'Revenue']
    depot_stats = depot_stats.sort_values('Total_Units', ascending=False)
    depot_stats['Units_Pct'] = (depot_stats['Total_Units'] / total_units * 100).round(2)
    depot_stats['Revenue_Pct'] = (depot_stats['Revenue'] / total_revenue * 100).round(2)
    depot_stats['Avg_Order_Size'] = (depot_stats['Total_Units'] / depot_stats['Orders']).round(1)
    
    lines.append(f"\nTop 5 Depots by Volume:")
    for idx, (depot, row) in enumerate(depot_stats.head().iterrows(), 1):
        lines.append(f"{idx}. {depot}:")
        lines.append(f"   - Units Distributed: {row['Total_Units']:,.0f} ({row['Units_Pct']:.1f}%)")
        lines.append(f"   - Revenue: ${row['Revenue']:,.2f} ({row['Revenue_Pct']:.1f}%)")
        # Store and route granularity not available in this aggregated schema
        lines.append(f"   - Avg Order Size: {row['Avg_Order_Size']:.1f} units")
    
    # Depot concentration analysis
    top3_depots_pct = depot_stats.head(3)['Units_Pct'].sum()
    lines.append(f"\n📊 Depot Concentration: Top 3 depots = {top3_depots_pct:.1f}% of volume")
    if top3_depots_pct > 60:
        lines.append(f"   ⚠️  HIGH CONCENTRATION: {top3_depots_pct:.1f}% concentrated in top 3 depots")
        lines.append(f"   → Risk: Over-reliance on few depots (capacity/disruption risk)")
    else:
        lines.append(f"   ✅ BALANCED: Healthy distribution across depot network")
    lines.append("")
    
    # === 3. STORE ORDERING PATTERNS ===
    lines.append("=" * 80)
    lines.append("3. STORE ORDERING PATTERNS")
    lines.append("=" * 80)
    
    # Depot-focused stats (store granularity not in aggregated dataset)
    lines.append(f"\nDepot-focused view only (store granularity not available in aggregated dataset).")
    
    # === 4. ROUTE EFFICIENCY ===
    lines.append("=" * 80)
    lines.append("4. DISTRIBUTION ROUTE EFFICIENCY")
    lines.append("=" * 80)
    
    # Route metrics not available in aggregated dataset; focusing on depot performance.
    
    # === 5. SKU DEMAND ANALYSIS ===
    lines.append("=" * 80)
    lines.append("5. SKU DEMAND IN B2B CHANNEL")
    lines.append("=" * 80)
    
    # Pricing column already determined above
    agg_map = {
        'quantity_sold': 'sum',
        'revenue': 'sum',
        'depot_id': 'nunique'
    }
    if price_col:
        agg_map[price_col] = 'mean'
    sku_stats = df.groupby('sku').agg(agg_map).round(2)
    if price_col:
        sku_stats.columns = ['Total_Units', 'Revenue', 'Depots_Stocking', 'Avg_Price']
    else:
        sku_stats.columns = ['Total_Units', 'Revenue', 'Depots_Stocking']
    sku_stats = sku_stats.sort_values('Total_Units', ascending=False)
    sku_stats['Units_Pct'] = (sku_stats['Total_Units'] / total_units * 100).round(2)
    
    lines.append(f"\nTop 10 SKUs by B2B Volume:")
    for idx, (sku, row) in enumerate(sku_stats.head(10).iterrows(), 1):
        lines.append(f"{idx}. {sku}:")
        lines.append(f"   - Units: {row['Total_Units']:,.0f} ({row['Units_Pct']:.1f}%)")
        lines.append(f"   - Revenue: ${row['Revenue']:,.2f}")
        if price_col:
            lines.append(f"   - Avg Wholesale Price: ${row['Avg_Price']:.2f}/unit")
        lines.append(f"   - Depots Stocking: {row['Depots_Stocking']:.0f}")
    
    # SKU variety analysis
    lines.append(f"\n📊 SKU Portfolio:")
    lines.append(f"   - Total SKUs: {n_skus}")
    lines.append(f"   - Top 5 SKUs: {sku_stats.head(5)['Units_Pct'].sum():.1f}% of volume")
    lines.append(f"   - Top 10 SKUs: {sku_stats.head(10)['Units_Pct'].sum():.1f}% of volume")
    lines.append("")
    
    # === 6. PRICING ANALYSIS ===
    lines.append("=" * 80)
    lines.append("6. WHOLESALE PRICING STRUCTURE")
    lines.append("=" * 80)
    
    # Pricing analysis
    if price_col:
        price_stats = df.groupby('sku')[price_col].agg(['mean', 'std', 'min', 'max']).round(2)
        price_stats = price_stats.sort_values('mean', ascending=False)
    else:
        price_stats = pd.DataFrame()
    
    lines.append(f"\nSKU Pricing (Wholesale):")
    if price_col:
        lines.append(f"Overall Average Wholesale Price: ${df[price_col].mean():.2f}/unit")
        lines.append(f"Price Range: ${df[price_col].min():.2f} - ${df[price_col].max():.2f}")
    if price_col:
        lines.append(f"\nTop 5 Most Expensive SKUs (Wholesale):")
        for idx, (sku, row) in enumerate(price_stats.head(5).iterrows(), 1):
            lines.append(f"{idx}. {sku}: ${row['mean']:.2f} avg (${row['min']:.2f}-${row['max']:.2f})")
    
    lines.append(f"\n💰 Pricing Insights:")
    if price_col:
        lines.append(f"   - Wholesale avg: ${df[price_col].mean():.2f}/unit")
        lines.append(f"   - Price variability: Std Dev = ${df[price_col].std():.2f}")
    lines.append(f"   ℹ️  Compare with Sales POS (retail) to validate margin structure")
    lines.append("")
    
    # === 7. TEMPORAL PATTERNS ===
    lines.append("=" * 80)
    lines.append("7. TEMPORAL ORDERING PATTERNS (B2B)")
    lines.append("=" * 80)
    
    # Daily patterns
    daily_stats = df.groupby('date').agg({
        'quantity_sold': 'sum',
        'depot_id': 'nunique'
    })
    
    lines.append(f"\n📅 Daily Metrics:")
    lines.append(f"   - Average daily volume: {daily_stats['quantity_sold'].mean():.0f} units")
    lines.append(f"   - Active depots per day: {daily_stats['depot_id'].mean():.1f}")
    lines.append(f"   - Peak day volume: {daily_stats['quantity_sold'].max():,.0f} units")
    lines.append(f"   - Lowest day volume: {daily_stats['quantity_sold'].min():,.0f} units")
    
    # Day of week patterns
    if 'day_of_week' in df.columns:
        dow_stats = df.groupby('day_of_week')['quantity_sold'].sum().sort_values(ascending=False)
        lines.append(f"\n📊 Day of Week Patterns:")
        lines.append(f"   - Highest volume day: {dow_stats.index[0]} ({dow_stats.iloc[0]:,.0f} units)")
        lines.append(f"   - Lowest volume day: {dow_stats.index[-1]} ({dow_stats.iloc[-1]:,.0f} units)")
    
    # Hourly patterns
    if 'hour' in df.columns:
        hourly_stats = df.groupby('hour')['quantity_sold'].sum().sort_values(ascending=False)
        lines.append(f"\n⏰ Hourly Ordering Patterns:")
        lines.append(f"   - Peak hour: {hourly_stats.index[0]:02d}:00 ({hourly_stats.iloc[0]:,.0f} units)")
        lines.append(f"   - Slowest hour: {hourly_stats.index[-1]:02d}:00 ({hourly_stats.iloc[-1]:,.0f} units)")
    lines.append("")
    
    # === 8. B2B vs B2C COMPARISON (conceptual) ===
    lines.append("=" * 80)
    lines.append("8. B2B CHANNEL CHARACTERISTICS")
    lines.append("=" * 80)
    
    avg_b2b_order = df['quantity_sold'].mean()
    lines.append(f"\n📦 B2B Order Profile:")
    lines.append(f"   - Average B2B order size: {avg_b2b_order:.1f} units")
    lines.append(f"   - Median B2B order size: {df['quantity_sold'].median():.0f} units")
    lines.append(f"   - 75th percentile: {df['quantity_sold'].quantile(0.75):.0f} units")
    lines.append(f"   - 95th percentile: {df['quantity_sold'].quantile(0.95):.0f} units")
    lines.append(f"   ℹ️  Compare with Sales POS avg order size (~31 units) for B2B vs B2C validation")
    lines.append(f"   ℹ️  B2B orders should be 3-5x larger (depot→store vs store→consumer)")
    
    lines.append(f"\n🔗 Network Structure:")
    lines.append(f"   - Active depots: {n_depots}")
    lines.append("")
    
    # === 9. DEPOT-SKU PREFERENCES ===
    lines.append("=" * 80)
    lines.append("9. DEPOT-SPECIFIC SKU DEMAND")
    lines.append("=" * 80)
    
    depot_sku = df.groupby(['depot_id', 'sku'])['quantity_sold'].sum().reset_index()
    depot_sku_pivot = depot_sku.pivot(index='sku', columns='depot_id', values='quantity_sold').fillna(0)
    
    lines.append(f"\nDepot-SKU Matrix Summary:")
    lines.append(f"   - Total depot-SKU combinations: {len(depot_sku)}")
    for depot in depot_sku_pivot.columns[:5]:  # Top 5 depots
        top_sku = depot_sku_pivot[depot].idxmax()
        top_units = depot_sku_pivot[depot].max()
        lines.append(f"   - {depot} top SKU: {top_sku} ({top_units:,.0f} units)")
    lines.append("")
    
    # === 10. KEY INSIGHTS & ACTIONS ===
    lines.append("=" * 80)
    lines.append("10. KEY INSIGHTS & ACTION ITEMS")
    lines.append("=" * 80)
    
    lines.append("\n🎯 Critical Findings:")
    
    # Finding 1: Depot concentration
    if top3_depots_pct > 60:
        lines.append(f"\n1. HIGH DEPOT CONCENTRATION ({top3_depots_pct:.1f}%)")
        lines.append(f"   → Risk: Over-reliance on top 3 depots creates capacity bottleneck")
        lines.append(f"   → Action: Expand secondary depot capacity, backup distribution plans")
    else:
        lines.append(f"\n1. BALANCED DEPOT NETWORK ({top3_depots_pct:.1f}%)")
        lines.append(f"   → Strength: No single-point-of-failure in depot network")
        lines.append(f"   → Action: Maintain balanced load distribution")
    
    # Finding 2: B2B order size
    if avg_b2b_order < 100:
        lines.append(f"\n2. SMALL B2B ORDER SIZES ({avg_b2b_order:.1f} units)")
        lines.append(f"   → Issue: Orders may be too frequent/small (inefficient logistics)")
        lines.append(f"   → Action: Encourage larger, less frequent orders (MOQ policies)")
    else:
        lines.append(f"\n2. HEALTHY B2B ORDER SIZES ({avg_b2b_order:.1f} units)")
        lines.append(f"   → Strength: Bulk ordering reduces distribution frequency/cost")
        lines.append(f"   → Action: Maintain MOQ policies, volume discounts")
    
    # Finding 3: SKU coverage across depots
    avg_sku_coverage = sku_stats['Depots_Stocking'].mean() / n_depots * 100
    lines.append(f"\n3. SKU COVERAGE ACROSS DEPOTS: {avg_sku_coverage:.1f}%")
    if avg_sku_coverage < 50:
        lines.append(f"   → Issue: Low SKU availability across depot network")
        lines.append(f"   → Action: Improve depot SKU stocking, demand forecasting")
    else:
        lines.append(f"   → Strength: Good SKU availability network-wide")
    
    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Write to file
    summary_path = REPORTS_DIR / 'sales_b2b_enhanced_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logging.info(f"Wrote {summary_path}")
    
    return '\n'.join(lines)


def grouped_summaries(df):
    """
    Generate grouped summary CSVs for pivot analysis.
    """
    # 1. By Depot
    depot_summary = df.groupby('depot_id').agg({
        'quantity_sold': ['sum', 'count', 'mean'],
        'revenue': 'sum',
        'sku': 'nunique'
    }).round(2)
    depot_summary.columns = ['Total_Units', 'Orders', 'Avg_Order_Size', 'Revenue', 'SKU_Variety']
    depot_summary = depot_summary.sort_values('Total_Units', ascending=False)
    depot_summary.to_csv(SUMMARIES_DIR / 'sales_b2b_by_depot.csv')
    logging.info("Wrote sales_b2b_by_depot.csv")
    
    # 2. By SKU
    
    # 4. By SKU
    price_col = 'bakers_inn_price' if 'bakers_inn_price' in df.columns else None
    sku_summary = df.groupby('sku').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum',
        **({price_col: 'mean'} if price_col else {}),
        'depot_id': 'nunique'
    }).round(2)
    if price_col:
        sku_summary.columns = ['Total_Units', 'Revenue', 'Avg_Wholesale_Price', 'Depots_Stocking']
    else:
        sku_summary.columns = ['Total_Units', 'Revenue', 'Depots_Stocking']
    sku_summary = sku_summary.sort_values('Total_Units', ascending=False)
    sku_summary.to_csv(SUMMARIES_DIR / 'sales_b2b_by_sku.csv')
    logging.info("Wrote sales_b2b_by_sku.csv")
    
    # 3. By Date
    date_summary = df.groupby('date').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum',
        'depot_id': 'nunique'
    }).round(2)
    date_summary.columns = ['Total_Units', 'Revenue', 'Depots_Active']
    date_summary.to_csv(SUMMARIES_DIR / 'sales_b2b_by_date.csv')
    logging.info("Wrote sales_b2b_by_date.csv")
    
    # 4. Depot-SKU Matrix
    depot_sku = df.groupby(['depot_id', 'sku'])['quantity_sold'].sum().reset_index()
    depot_sku_pivot = depot_sku.pivot(index='sku', columns='depot_id', values='quantity_sold').fillna(0)
    depot_sku_pivot.to_csv(SUMMARIES_DIR / 'sales_b2b_depot_sku_matrix.csv')
    logging.info("Wrote sales_b2b_depot_sku_matrix.csv")
    # Store/Route network summaries are not available in the aggregated schema


def visualizations(df):
    """
    Generate 10+ comprehensive visualizations for B2B Sales Dataset.
    """

    def add_caption(text):
        fig = plt.gcf()
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    # 1. Depot Performance Bar Chart
    plt.figure(figsize=(12, 6))
    depot_vol = df.groupby('depot_id')['quantity_sold'].sum().sort_values(ascending=True)
    colors = ['crimson' if x > depot_vol.quantile(0.66) else 'orange' if x > depot_vol.quantile(0.33) else 'gold' 
              for x in depot_vol]
    depot_vol.plot(kind='barh', color=colors)
    plt.xlabel('Total Units Distributed', fontsize=12, fontweight='bold')
    plt.ylabel('Depot ID', fontsize=12, fontweight='bold')
    plt.title('Depot Distribution Performance - Total Units Distributed', fontsize=14, fontweight='bold')
    add_caption('Highlights the depots moving the most volume for capacity planning.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_by_depot.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_by_depot.png")
    
    # Store and route visuals are not available in aggregated schema
    
    # 4. SKU Demand in B2B Channel
    plt.figure(figsize=(12, 8))
    sku_vol = df.groupby('sku')['quantity_sold'].sum().sort_values(ascending=True)
    colors = ['darkgreen' if x > sku_vol.quantile(0.75) else 'orange' if x > sku_vol.median() else 'gold' 
              for x in sku_vol]
    sku_vol.plot(kind='barh', color=colors)
    plt.xlabel('Total Units Distributed (B2B)', fontsize=12, fontweight='bold')
    plt.ylabel('SKU', fontsize=12, fontweight='bold')
    plt.title('SKU Distribution Volume in B2B Channel', fontsize=14, fontweight='bold')
    plt.axvline(sku_vol.median(), color='red', linestyle='--', linewidth=2, 
                label=f"Median: {sku_vol.median():,.0f}")
    plt.legend()
    add_caption('Shows which SKUs dominate depot-to-store distribution volume.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_by_sku.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_by_sku.png")
    
    # 5. Daily Sales Trend
    plt.figure(figsize=(14, 6))
    daily_sales = df.groupby('date')['quantity_sold'].sum().sort_index()
    plt.fill_between(daily_sales.index, daily_sales.values, alpha=0.3, color='steelblue')
    plt.plot(daily_sales.index, daily_sales.values, color='darkblue', linewidth=2, label='Daily Volume')
    
    # 7-day moving average
    ma7 = daily_sales.rolling(window=7, center=True).mean()
    plt.plot(ma7.index, ma7.values, color='red', linewidth=2, linestyle='--', label='7-Day Moving Avg')
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Units Distributed', fontsize=12, fontweight='bold')
    plt.title('Daily B2B Distribution Volume with Moving Average', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    add_caption('Tracks demand cycles and the smoothed trend for planning.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_daily_trend.png")
    
    # 6. Day of Week Pattern
    plt.figure(figsize=(10, 6))
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = df.groupby('day_of_week')['quantity_sold'].sum().reindex(dow_order)
    
    colors = ['steelblue' if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] 
              else 'coral' for day in dow_sales.index]
    bars = plt.bar(dow_sales.index, dow_sales.values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
    plt.ylabel('Total Units Distributed', fontsize=12, fontweight='bold')
    plt.title('B2B Distribution Volume by Day of Week', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    add_caption('Shows weekly ordering rhythm for distribution scheduling.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_day_of_week.png")
    
    # 7. Hourly Ordering Pattern (skip if hour missing)
    if 'hour' in df.columns:
        plt.figure(figsize=(12, 6))
        hourly_sales = df.groupby('hour')['quantity_sold'].sum()
        plt.bar(hourly_sales.index, hourly_sales.values, color='teal', alpha=0.7)
        plt.plot(hourly_sales.index, hourly_sales.values, color='red', marker='o', linewidth=2)
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
        plt.ylabel('Total Units Ordered', fontsize=12, fontweight='bold')
        plt.title('B2B Order Volume by Hour of Day', fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3, axis='y')
        add_caption('Reveals peak order windows for depot operations.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'sales_b2b_hourly_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved sales_b2b_hourly_pattern.png")
    
    # 8. Order Size Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['quantity_sold'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(df['quantity_sold'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {df['quantity_sold'].mean():.1f}")
    plt.axvline(df['quantity_sold'].median(), color='green', linestyle='--', linewidth=2, 
                label=f"Median: {df['quantity_sold'].median():.1f}")
    plt.xlabel('Order Size (Units)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('B2B Order Size Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    add_caption('Shows typical wholesale order sizes and extreme orders.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_order_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_order_size_distribution.png")
    
    # 9. Depot-SKU Heatmap
    plt.figure(figsize=(14, 10))
    depot_sku = df.groupby(['depot_id', 'sku'])['quantity_sold'].sum().unstack(fill_value=0)
    sns.heatmap(depot_sku.T, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Units Distributed'})
    plt.xlabel('Depot ID', fontsize=12, fontweight='bold')
    plt.ylabel('SKU', fontsize=12, fontweight='bold')
    plt.title('Depot-SKU Distribution Heatmap', fontsize=14, fontweight='bold')
    add_caption('Highlights which depots focus on which SKUs.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_depot_sku_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_depot_sku_heatmap.png")
    
    # 10. Wholesale Pricing by SKU
    if 'bakers_inn_price' in df.columns:
        plt.figure(figsize=(12, 8))
        price_by_sku = df.groupby('sku')['bakers_inn_price'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        plt.barh(range(len(price_by_sku)), price_by_sku['mean'], 
                 xerr=price_by_sku['std'], color='gold', edgecolor='black', alpha=0.7)
        plt.yticks(range(len(price_by_sku)), price_by_sku.index)
        plt.xlabel('Wholesale Price per Unit ($)', fontsize=12, fontweight='bold')
        plt.ylabel('SKU', fontsize=12, fontweight='bold')
        plt.title('Wholesale Pricing by SKU (Mean ± Std Dev)', fontsize=14, fontweight='bold')
        add_caption('Compares wholesale price levels and variability by SKU.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'sales_b2b_pricing_by_sku.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved sales_b2b_pricing_by_sku.png")
    
    # 11. Depot Market Share (Pie Chart)
    plt.figure(figsize=(10, 10))
    depot_share = df.groupby('depot_id')['quantity_sold'].sum().sort_values(ascending=False)
    colors_pie = plt.cm.Set3(range(len(depot_share)))
    
    plt.pie(depot_share.values, labels=depot_share.index, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90)
    plt.title('Depot Market Share (by Volume)', fontsize=14, fontweight='bold')
    add_caption('Shows each depot’s share of total distributed volume.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_depot_share_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_depot_share_pie.png")
    
    # 12. Revenue by Depot
    plt.figure(figsize=(12, 6))
    depot_revenue = df.groupby('depot_id')['revenue'].sum().sort_values(ascending=True)
    colors = ['darkgreen' if x > depot_revenue.quantile(0.66) else 'orange' 
              if x > depot_revenue.quantile(0.33) else 'gold' for x in depot_revenue]
    depot_revenue.plot(kind='barh', color=colors)
    plt.xlabel('Total Revenue ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Depot ID', fontsize=12, fontweight='bold')
    plt.title('Depot Performance by Revenue (Wholesale)', fontsize=14, fontweight='bold')
    add_caption('Highlights depots contributing most to wholesale revenue.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_b2b_depot_revenue.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_b2b_depot_revenue.png")

    # 13. B2B demand heatmap by day and month
    if 'day_of_week' in df.columns and 'month_name' in df.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        heat = df.pivot_table(index='day_of_week', columns='month_name', values='quantity_sold', aggfunc='sum').reindex(day_order)
        heat = heat.reindex(columns=[m for m in month_order if m in heat.columns])
        plt.figure(figsize=(12, 6))
        sns.heatmap(heat, cmap='YlGnBu')
        plt.xlabel('Month')
        plt.ylabel('Day of Week')
        plt.title('B2B Demand Heatmap (Day of Week vs Month)')
        add_caption('Shows seasonal day-of-week demand patterns in wholesale orders.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'sales_b2b_demand_heatmap_day_month.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved sales_b2b_demand_heatmap_day_month.png")

    # 14. Depot order size distribution (top depots)
    top_depots = df.groupby('depot_id')['quantity_sold'].sum().nlargest(5).index
    df_top = df[df['depot_id'].isin(top_depots)]
    if not df_top.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='depot_id', y='quantity_sold', data=df_top)
        plt.xlabel('Depot ID')
        plt.ylabel('Order Size (Units)')
        plt.title('Order Size Distribution (Top 5 Depots)')
        plt.xticks(rotation=45)
        add_caption('Compares order-size variability across high-volume depots.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'sales_b2b_order_size_top_depots.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved sales_b2b_order_size_top_depots.png")

    # 15. Revenue vs units scatter (price consistency)
    if 'revenue' in df.columns:
        df_scatter = df[['quantity_sold', 'revenue']].dropna()
        if len(df_scatter) > 0:
            if len(df_scatter) > 10000:
                df_scatter = df_scatter.sample(10000, random_state=42)
            plt.figure(figsize=(7, 5))
            plt.scatter(df_scatter['quantity_sold'], df_scatter['revenue'], alpha=0.3)
            plt.xlabel('Units Sold')
            plt.ylabel('Revenue ($)')
            plt.title('Revenue vs Units Sold (B2B Orders)')
            add_caption('Checks pricing consistency across order sizes.')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / 'sales_b2b_revenue_vs_units.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved sales_b2b_revenue_vs_units.png")


def main():
    """
    Main execution function.
    """
    logging.info("=" * 80)
    logging.info("Starting Sales Dataset (B2B Channel) Enhanced EDA")
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
    logging.info("✅ Sales B2B EDA complete!")
    logging.info(f"   - Summary: reports/sales_b2b_enhanced_summary.txt")
    logging.info(f"   - Figures: reports/figures/sales_b2b_*.png (12 visualizations)")
    logging.info(f"   - CSVs: reports/summaries/sales_b2b_*.csv (7 summary files)")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
