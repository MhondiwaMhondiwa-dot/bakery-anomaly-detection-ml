"""
Exploratory Data Analysis for Sales / Retail POS Dataset

This script analyzes the demand-side dataset that records actual customer purchases
at retail outlets. Sales data validates production and dispatch decisions and drives
demand forecasting.

Key Analyses:
- Demand patterns (daily/weekly/hourly)
- SKU performance (fast vs slow-moving products)
- Regional demand variations
- Promotion effectiveness and ROI
- Price elasticity and revenue analysis
- Holiday impact on sales
- Retailer performance ranking
- Sell-through analysis

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
    Load sales POS dataset and prepare derived fields.
    
    Returns:
        pd.DataFrame: Sales data with derived fields
    """
    df = pd.read_parquet(DATA_DIR / 'sales_pos_dataset.parquet')
    logging.info(f"Loaded {len(df):,} sales transactions")
    
    # ensure retailer_id is string in case it was mis-parsed as datetime
    if 'retailer_id' in df.columns:
        df['retailer_id'] = df['retailer_id'].astype(str)
    
    # Derive time-based features
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['month_name'] = df['timestamp'].dt.month_name()
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Compute total quantity_sold from multi-SKU columns
    sku_cols = [c for c in ['soft_white','high_energy_brown','whole_grain_loaf','low_gi_seed_loaf'] if c in df.columns]
    if sku_cols:
        df['quantity_sold'] = df[sku_cols].sum(axis=1)
    else:
        df['quantity_sold'] = df.get('quantity_sold', 0)

    # Derive business metrics using retail_price if available, else bakers_inn_price
    # choose a price column that is numeric (ignore malformed datetimes)
    price_col = None
    for cand in ['retail_price', 'bakers_inn_price']:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            price_col = cand
            break
    if price_col is not None:
        df['revenue'] = df['quantity_sold'] * df[price_col]
    else:
        df['revenue'] = df['quantity_sold']
    
    # Promotion categorization
    df['promotion_category'] = df['promotion_name'].fillna('No Promotion')
    
    return df, price_col

def summary_stats(df, price_col):
    """
    Generate comprehensive summary statistics for sales POS dataset.
    
    Args:
        df: Sales DataFrame
        price_col: Price column name ('retail_price' or 'bakers_inn_price' or None)
    """
    summary_path = REPORTS_DIR / 'sales_pos_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SALES / RETAIL POS DATASET - EXPLORATORY DATA ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset overview
        f.write("📊 DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Sales Transactions: {len(df):,}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Total Units Sold: {df['quantity_sold'].sum():,}\n")
        f.write(f"Total Revenue: ${df['revenue'].sum():,.2f}\n")
        f.write(f"Average Transaction Size: {df['quantity_sold'].mean():.1f} units\n")
        if price_col:
            f.write(f"Average Price: ${df[price_col].mean():.2f}\n")
        f.write(f"Unique Retailers: {df['retailer_id'].nunique()}\n")
        f.write(f"Regions: {df['region'].nunique()} - {', '.join(sorted(df['region'].unique()))}\n")
        f.write(f"Multi-SKU columns: soft_white, high_energy_brown, whole_grain_loaf, low_gi_seed_loaf\n\n")
        
        # Promotion overview
        promo_count = df['promotion_flag'].sum()
        promo_pct = (promo_count / len(df)) * 100
        f.write("🎯 PROMOTION OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Sales with Promotions: {promo_count:,} ({promo_pct:.1f}%)\n")
        f.write(f"Sales without Promotions: {len(df) - promo_count:,} ({100 - promo_pct:.1f}%)\n")
        f.write(f"Unique Promotions: {df[df['promotion_flag'] == 1]['promotion_name'].nunique()}\n\n")
        
        # Promotion effectiveness
        promo_sales = df.groupby('promotion_flag').agg({
            'quantity_sold': ['sum', 'mean'],
            'revenue': ['sum', 'mean'],
            price_col: 'mean' if price_col else 'sum'
        }).round(2)
        f.write("Promotion vs Non-Promotion Performance:\n")
        f.write(f"{promo_sales}\n\n")
        
        promo_uplift_qty = ((promo_sales.loc[1, ('quantity_sold', 'mean')] / 
                             promo_sales.loc[0, ('quantity_sold', 'mean')]) - 1) * 100
        promo_uplift_rev = ((promo_sales.loc[1, ('revenue', 'mean')] / 
                            promo_sales.loc[0, ('revenue', 'mean')]) - 1) * 100
        
        if promo_uplift_qty > 0:
            f.write(f"✅ Promotion Uplift: +{promo_uplift_qty:.1f}% quantity, +{promo_uplift_rev:.1f}% revenue per transaction\n\n")
        else:
            f.write(f"⚠️ WARNING: Promotions showing NEGATIVE uplift: {promo_uplift_qty:.1f}% quantity\n\n")
        
        # Top promotions by volume
        f.write("Top Promotions by Sales Volume:\n")
        top_promos = df[df['promotion_flag'] == 1].groupby('promotion_name').agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'sale_id': 'count'
        }).sort_values('quantity_sold', ascending=False).head(10)
        top_promos.columns = ['Units Sold', 'Revenue', 'Transactions']
        f.write(f"{top_promos}\n\n")
        
        # Regional performance
        f.write("🌍 REGIONAL DEMAND ANALYSIS\n")
        f.write("-" * 80 + "\n")
        regional = df.groupby('region').agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'sale_id': 'count',
            'retailer_id': 'nunique'
        }).sort_values('quantity_sold', ascending=False)
        regional.columns = ['Units Sold', 'Revenue', 'Transactions', 'Retailers']
        regional['Avg Units/Transaction'] = (regional['Units Sold'] / regional['Transactions']).round(1)
        regional['Avg Revenue/Transaction'] = (regional['Revenue'] / regional['Transactions']).round(2)
        f.write(f"{regional}\n\n")
        
        # SKU performance (from multi-SKU columns)
        f.write("🍞 SKU PERFORMANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Aggregate each SKU from its column
        sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
        sku_perf_data = []
        for sku_col in sku_cols:
            if sku_col in df.columns:
                total_qty = df[sku_col].sum()
                total_revenue = (df[sku_col] * df.get(price_col, 1)).sum() if price_col else total_qty
                transactions = (df[sku_col] > 0).sum()
                sku_perf_data.append({
                    'SKU': sku_col.replace('_', ' ').title(),
                    'Units Sold': int(total_qty),
                    'Revenue': total_revenue,
                    'Transactions': int(transactions),
                    '% of Total Units': (total_qty / df['quantity_sold'].sum() * 100) if df['quantity_sold'].sum() > 0 else 0
                })
        
        sku_perf_df = pd.DataFrame(sku_perf_data).sort_values('Units Sold', ascending=False)
        f.write(f"{sku_perf_df.to_string()}\n\n")
        
        # Temporal patterns
        f.write("⏰ TEMPORAL DEMAND PATTERNS\n")
        f.write("-" * 80 + "\n")
        
        # Day of week
        dow_sales = df.groupby('day_name').agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'sale_id': 'count'
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        dow_sales.columns = ['Units', 'Revenue', 'Transactions']
        f.write("Sales by Day of Week:\n")
        f.write(f"{dow_sales}\n\n")
        
        # Weekend vs weekday
        weekend_comp = df.groupby('is_weekend').agg({
            'quantity_sold': ['sum', 'mean'],
            'revenue': ['sum', 'mean']
        }).round(2)
        weekend_comp.index = ['Weekday', 'Weekend']
        f.write("Weekday vs Weekend:\n")
        f.write(f"{weekend_comp}\n\n")
        
        # Hourly patterns
        hourly = df.groupby('hour')['quantity_sold'].agg(['sum', 'mean', 'count']).round(1)
        hourly.columns = ['Total Units', 'Avg Units/Sale', 'Transactions']
        peak_hour = hourly['Total Units'].idxmax()
        lowest_hour = hourly['Total Units'].idxmin()
        f.write("Hourly Sales Summary:\n")
        f.write(f"{hourly}\n\n")
        f.write(f"Peak Sales Hour: {peak_hour}:00 ({hourly.loc[peak_hour, 'Total Units']:.0f} units)\n")
        f.write(f"Lowest Sales Hour: {lowest_hour}:00 ({hourly.loc[lowest_hour, 'Total Units']:.0f} units)\n\n")
        
        # Price analysis (for available price columns)
        if price_col:
            f.write("💲 PRICE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            price_stats = df.groupby('retailer_id')[price_col].agg(['min', 'mean', 'max', 'std']).round(2).head(10)
            price_stats.columns = ['Min Price', 'Avg Price', 'Max Price', 'Std Dev']
            f.write(f"Price Statistics by Retailer (Top 10):\n{price_stats}\n\n")
        
        # Retailer performance
        f.write("🏪 TOP RETAILERS BY SALES VOLUME\n")
        f.write("-" * 80 + "\n")
        retailer_perf = df.groupby('retailer_id').agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'sale_id': 'count'
        }).sort_values('quantity_sold', ascending=False).head(20)
        retailer_perf.columns = ['Units Sold', 'Revenue', 'Transactions']
        retailer_perf['Avg Units/Transaction'] = (retailer_perf['Units Sold'] / retailer_perf['Transactions']).round(1)
        f.write(f"{retailer_perf}\n\n")
        
        # ACTION ITEMS
        f.write("🎯 KEY INSIGHTS & ACTION ITEMS\n")
        f.write("=" * 80 + "\n")
        
        # Check for anomalies
        f.write(f"1. ⚠️ Retailers with very low sales\n")
        f.write("   Action: Investigate stock-outs, poor locations, or dispatch issues\n\n")
        
        # Zero sales check (retailers in dispatch but not in sales)
        f.write("2. Check for retailers receiving dispatch but showing zero/low sales\n")
        f.write("   Action: Cross-reference with dispatch_dataset to identify overstock situations\n\n")
        
        # Promotion effectiveness
        f.write("3. Promotion Impact: Review promotion strategy and effectiveness\n\n")
        
        # SKU performance
        f.write("4. Multi-SKU Performance: Track demand variations across soft_white, high_energy_brown, whole_grain_loaf, low_gi_seed_loaf\n")
        f.write("   Action: Focus on high-demand SKUs; consider discontinuing low-demand products\n\n")
        
        # Regional expansion
        f.write("5. Regional expansion opportunities\n")
        f.write("   Action: Identify high-demand regions and expand retailer network\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("✅ Sales POS summary complete!\n")
    
    logging.info(f"Wrote {summary_path}")

def grouped_summaries(df):
    """
    Generate grouped summary CSV files.
    
    Args:
        df: Sales DataFrame
    """
    # determine numeric price column for use in later SKU/regional loops
    price_col = None
    for cand in ['retail_price','bakers_inn_price']:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            price_col = cand
            break

    # 1. Sales by date
    daily_summary = df.groupby('date').agg({
        'quantity_sold': 'sum',
        'revenue': 'sum',
        'sale_id': 'count',
        'promotion_flag': 'sum'
    }).round(2)
    daily_summary.columns = ['total_units', 'total_revenue', 'transactions', 'promo_transactions']
    daily_summary.to_csv(SUMMARIES_DIR / 'sales_pos_by_date.csv')
    logging.info("Wrote sales_pos_by_date.csv")
    
    # 2. Sales by region
    region_summary = df.groupby('region').agg({
        'quantity_sold': ['sum', 'mean'],
        'revenue': ['sum', 'mean'],
        'sale_id': 'count',
        'retailer_id': 'nunique',
        'promotion_flag': 'sum'
    }).round(2)
    region_summary.columns = ['_'.join(col).strip() for col in region_summary.columns]
    region_summary = region_summary.sort_values('quantity_sold_sum', ascending=False)
    region_summary.to_csv(SUMMARIES_DIR / 'sales_pos_by_region.csv')
    logging.info("Wrote sales_pos_by_region.csv")
    
    # 3. Sales by retailer (top 50)
    retailer_summary = df.groupby('retailer_id').agg({
        'quantity_sold': ['sum', 'mean'],
        'revenue': ['sum', 'mean'],
        'sale_id': 'count',
        'promotion_flag': 'sum',
        'region': 'first'
    }).round(2)
    retailer_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in retailer_summary.columns]
    retailer_summary = retailer_summary.sort_values('quantity_sold_sum', ascending=False).head(50)
    retailer_summary.to_csv(SUMMARIES_DIR / 'sales_pos_by_retailer_top50.csv')
    logging.info("Wrote sales_pos_by_retailer_top50.csv")
    
    # 4. Sales by hour of day
    hourly_summary = df.groupby('hour').agg({
        'quantity_sold': ['sum', 'mean', 'count'],
        'revenue': ['sum', 'mean']
    }).round(2)
    hourly_summary.columns = ['_'.join(col).strip() for col in hourly_summary.columns]
    hourly_summary.to_csv(SUMMARIES_DIR / 'sales_pos_by_hour.csv')
    logging.info("Wrote sales_pos_by_hour.csv")
    
    # 6. Promotion performance
    promo_summary = df[df['promotion_flag'] == 1].groupby('promotion_name').agg({
        'quantity_sold': ['sum', 'mean', 'count'],
        'revenue': ['sum', 'mean'],
        'retailer_id': 'nunique'
    }).round(2)
    promo_summary.columns = ['_'.join(col).strip() for col in promo_summary.columns]
    promo_summary = promo_summary.sort_values('quantity_sold_sum', ascending=False)
    promo_summary.to_csv(SUMMARIES_DIR / 'sales_pos_by_promotion.csv')
    logging.info("Wrote sales_pos_by_promotion.csv")
    
    # 7. Regional multi-SKU preferences (from column sums)
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    region_sku_data = []
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        for sku_col in sku_cols:
            if sku_col in region_df.columns:
                total_qty = region_df[sku_col].sum()
                if price_col:
                    total_rev = (region_df[sku_col] * region_df[price_col]).sum()
                else:
                    total_rev = region_df[sku_col].sum()
                region_sku_data.append({
                    'Region': region,
                    'SKU': sku_col.replace('_', ' ').title(),
                    'Total_Units': int(total_qty),
                    'Total_Revenue': total_rev
                })
    
    if region_sku_data:
        regional_sku = pd.DataFrame(region_sku_data).sort_values('Total_Units', ascending=False)
        regional_sku.to_csv(SUMMARIES_DIR / 'sales_pos_regional_sku_preferences.csv', index=False)
        logging.info("Wrote sales_pos_regional_sku_preferences.csv")

def visualizations(df):
    """
    Generate comprehensive visualizations for sales POS dataset.
    
    Args:
        df: Sales DataFrame
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    def add_caption(fig, text):
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    # 1. Sales volume by multi-SKU columns
    fig, ax = plt.subplots(figsize=(12, 6))
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    sku_totals = {col.replace('_', ' ').title(): df[col].sum() for col in sku_cols if col in df.columns}
    sku_series = pd.Series(sku_totals).sort_values(ascending=True)
    colors = ['green' if x > sku_series.median() else 'orange' for x in sku_series]
    sku_series.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Total Sales Volume by SKU', fontsize=16, fontweight='bold')
    ax.set_xlabel('Units Sold', fontsize=12)
    ax.set_ylabel('SKU', fontsize=12)
    ax.axvline(sku_series.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {sku_series.median():.0f}')
    ax.legend()
    add_caption(fig, 'Shows which SKUs drive most unit sales for mix and planning decisions.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_volume_by_sku.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_volume_by_sku.png")
    
    # 2. Revenue by region
    fig, ax = plt.subplots(figsize=(12, 6))
    region_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
    region_rev.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Total Revenue by Region', fontsize=16, fontweight='bold')
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Revenue ($)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(region_rev.values):
        ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=10)
    add_caption(fig, 'Compares regional revenue contribution to prioritize distribution focus.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_revenue_by_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_revenue_by_region.png")
    
    # 3. Promotion effectiveness comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Volume comparison
    promo_comp = df.groupby('promotion_flag')['quantity_sold'].mean()
    promo_labels = ['No Promotion', 'With Promotion']
    colors_promo = ['lightcoral', 'lightgreen']
    axes[0].bar(promo_labels, promo_comp.values, color=colors_promo)
    axes[0].set_title('Average Units Sold per Transaction', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Avg Units', fontsize=12)
    for i, v in enumerate(promo_comp.values):
        axes[0].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Revenue comparison
    promo_rev = df.groupby('promotion_flag')['revenue'].mean()
    axes[1].bar(promo_labels, promo_rev.values, color=colors_promo)
    axes[1].set_title('Average Revenue per Transaction', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Avg Revenue ($)', fontsize=12)
    for i, v in enumerate(promo_rev.values):
        axes[1].text(i, v, f'${v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('Promotion Effectiveness Analysis', fontsize=16, fontweight='bold', y=1.02)
    add_caption(fig, 'Quantifies uplift from promotions in both units and revenue per sale.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_promotion_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_promotion_effectiveness.png")
    
    # 4. Daily sales trend
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_sales = df.groupby('date')['quantity_sold'].sum().reset_index()
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    
    ax.plot(daily_sales['date'], daily_sales['quantity_sold'], linewidth=2, color='darkblue', alpha=0.7)
    ax.fill_between(daily_sales['date'], daily_sales['quantity_sold'], alpha=0.3, color='skyblue')
    
    # Add 7-day moving average
    daily_sales['ma7'] = daily_sales['quantity_sold'].rolling(window=7, center=True).mean()
    ax.plot(daily_sales['date'], daily_sales['ma7'], linewidth=3, color='red', label='7-Day Moving Avg')
    
    ax.set_title('Daily Sales Trend (Units Sold)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Units Sold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_caption(fig, 'Highlights seasonality and longer-term demand shifts.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_daily_trend.png")
    
    # 5. Hourly sales pattern
    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_sales = df.groupby('hour')['quantity_sold'].sum()
    ax.bar(hourly_sales.index, hourly_sales.values, color='teal', alpha=0.7)
    ax.plot(hourly_sales.index, hourly_sales.values, color='darkred', marker='o', linewidth=2, markersize=8)
    ax.set_title('Sales Volume by Hour of Day', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Total Units Sold', fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)
    add_caption(fig, 'Shows intraday demand peaks to align bake and dispatch timing.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_hourly_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_hourly_pattern.png")
    
    # 6. Day of week comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = df.groupby('day_name')['quantity_sold'].sum().reindex(dow_order)
    colors_dow = ['lightblue' if day not in ['Saturday', 'Sunday'] else 'lightcoral' for day in dow_order]
    ax.bar(dow_order, dow_sales.values, color=colors_dow)
    ax.set_title('Sales Volume by Day of Week', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Total Units Sold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(dow_sales.values):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
    add_caption(fig, 'Compares weekday vs weekend demand for staffing and routing.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_day_of_week.png")
    
    # 7. Top promotions by volume
    fig, ax = plt.subplots(figsize=(12, 7))
    promo_sales = df[df['promotion_flag'] == 1].groupby('promotion_name')['quantity_sold'].sum().sort_values(ascending=True)
    if len(promo_sales) > 0:
        promo_sales.plot(kind='barh', ax=ax, color='gold')
        ax.set_title('Sales Volume by Promotion', fontsize=16, fontweight='bold')
        ax.set_xlabel('Units Sold', fontsize=12)
        ax.set_ylabel('Promotion', fontsize=12)
        add_caption(fig, 'Ranks promotions by total units sold to identify best performers.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'sales_pos_promotion_volume.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved sales_pos_promotion_volume.png")
    else:
        plt.close()
    
    # 8. Regional multi-SKU preferences heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    available_skus = [c for c in sku_cols if c in df.columns]
    
    if available_skus:
        regional_data = []
        for region in sorted(df['region'].unique()):
            region_df = df[df['region'] == region]
            row = {col.replace('_', ' ').title(): region_df[col].sum() for col in available_skus}
            row['Region'] = region
            regional_data.append(row)
        
        regional_sku_matrix = pd.DataFrame(regional_data).set_index('Region')
        sns.heatmap(regional_sku_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Units Sold'})
        ax.set_title('Regional SKU Preferences Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('SKU', fontsize=12)
        ax.set_ylabel('Region', fontsize=12)
        add_caption(fig, 'Shows which SKUs sell best in each region for localized planning.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'sales_pos_regional_sku_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved sales_pos_regional_sku_heatmap.png")
    else:
        plt.close()
    
    # 9. Price distribution — histogram with explicit numeric bins
    price_col_for_viz = None
    for _cand in ['retail_price', 'bakers_inn_price']:
        if _cand in df.columns and pd.api.types.is_numeric_dtype(df[_cand]) and df[_cand].std() > 0:
            price_col_for_viz = _cand
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: histogram of transaction-level price
    if price_col_for_viz:
        prices = df[price_col_for_viz].dropna()
        pmin, pmax = prices.min(), prices.max()
        bins = np.linspace(pmin, pmax, 15)
        counts, edges = np.histogram(prices, bins=bins)
        labels = [f"${edges[i]:.2f}–\n${edges[i+1]:.2f}" for i in range(len(edges) - 1)]
        axes[0].bar(range(len(counts)), counts, color='steelblue', edgecolor='white')
        axes[0].set_xticks(range(len(counts)))
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_title(f'Transaction Price Distribution\n({price_col_for_viz})',
                          fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Price per Unit ($)', fontsize=11)
        axes[0].set_ylabel('Transactions', fontsize=11)
    else:
        axes[0].text(0.5, 0.5, 'No price data available', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('Price Distribution', fontsize=13, fontweight='bold')

    # Right: per-SKU estimated revenue (quantity × SKU price)
    sku_price_map = {
        'soft_white': 1.05, 'high_energy_brown': 1.25,
        'whole_grain_loaf': 1.45, 'low_gi_seed_loaf': 1.70,
    }
    sku_rev = {}
    for sku_col, unit_price in sku_price_map.items():
        if sku_col in df.columns:
            sku_rev[sku_col.replace('_', ' ').title()] = (df[sku_col] * unit_price).sum()
    if sku_rev:
        sku_labels = list(sku_rev.keys())
        sku_vals   = list(sku_rev.values())
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
        axes[1].bar(sku_labels, sku_vals, color=colors[:len(sku_labels)], edgecolor='white')
        axes[1].set_title('Total Revenue by SKU (Period)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('SKU', fontsize=11)
        axes[1].set_ylabel('Total Revenue ($)', fontsize=11)
        axes[1].tick_params(axis='x', rotation=20)
        for i, v in enumerate(sku_vals):
            axes[1].text(i, v * 1.01, f'${v:,.0f}', ha='center', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No SKU data available', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=12)

    add_caption(fig, 'Left: distribution of weighted-average retail price per transaction. '
                     'Right: estimated total revenue contribution per SKU.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_price_distribution.png")
    
    # 10. Top 20 retailers by revenue
    fig, ax = plt.subplots(figsize=(12, 8))
    # Compute revenue using per-SKU prices for accuracy
    _sku_price_map = {
        'soft_white': 1.05, 'high_energy_brown': 1.25,
        'whole_grain_loaf': 1.45, 'low_gi_seed_loaf': 1.70,
    }
    _available_sku_cols = [c for c in _sku_price_map if c in df.columns]
    if _available_sku_cols:
        df['_sku_revenue'] = sum(df[c] * _sku_price_map[c] for c in _available_sku_cols)
        retailer_rev = df.groupby('retailer_id')['_sku_revenue'].sum().sort_values(ascending=True)
    else:
        retailer_rev = df.groupby('retailer_id')['revenue'].sum().sort_values(ascending=True)

    if not retailer_rev.empty and retailer_rev.dropna().sum() > 0:
        top20 = retailer_rev.tail(20)
        bars = ax.barh(range(len(top20)), top20.values, color='mediumseagreen')
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index, fontsize=9)
        # Add value labels
        for bar, val in zip(bars, top20.values):
            ax.text(val * 1.005, bar.get_y() + bar.get_height() / 2,
                    f'${val:,.0f}', va='center', fontsize=8)
    else:
        logging.info('No valid retailer revenue data; skipping retailer bar plot')
        ax.text(0.5, 0.5, 'No revenue data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    ax.set_title('Top 20 Retailers by Revenue', fontsize=16, fontweight='bold')
    ax.set_xlabel('Total Revenue ($)', fontsize=12)
    ax.set_ylabel('Retailer ID', fontsize=12)
    add_caption(fig, 'Identifies highest-value retailers for account prioritization.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_top_retailers.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_top_retailers.png")

    # 11. Demand heatmap by day of week and hour
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heat = df.pivot_table(index='day_name', columns='hour', values='quantity_sold', aggfunc='sum').reindex(days)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat, cmap='YlGnBu', ax=ax)
    ax.set_title('Demand Heatmap by Day and Hour', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    add_caption(fig, 'Reveals the busiest day-hour windows for demand planning.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sales_pos_demand_heatmap_day_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved sales_pos_demand_heatmap_day_hour.png")

    # 12. Promotion uplift by SKU (mean units per transaction)
    sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
    available_skus = [c for c in sku_cols if c in df.columns]
    if available_skus and 'promotion_flag' in df.columns:
        uplift_rows = []
        for sku in available_skus:
            promo_mean = df[df['promotion_flag'] == 1][sku].mean()
            nonpromo_mean = df[df['promotion_flag'] == 0][sku].mean()
            if nonpromo_mean and not np.isnan(nonpromo_mean):
                uplift = (promo_mean / nonpromo_mean - 1) * 100
            else:
                uplift = np.nan
            uplift_rows.append({'sku': sku.replace('_', ' ').title(), 'uplift_pct': uplift})
        uplift_df = pd.DataFrame(uplift_rows).dropna()
        if not uplift_df.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            uplift_df = uplift_df.sort_values('uplift_pct', ascending=True)
            ax.barh(uplift_df['sku'], uplift_df['uplift_pct'], color='slateblue')
            ax.set_title('Promotion Uplift by SKU (Mean Units per Sale)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Uplift (%)', fontsize=12)
            ax.axvline(0, color='black', linewidth=1)
            add_caption(fig, 'Shows which SKUs respond best to promotions by unit uplift.')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / 'sales_pos_promo_uplift_by_sku.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved sales_pos_promo_uplift_by_sku.png")

    # 13. Price vs units sold scatter (if price is available)
    if 'retail_price' in df.columns or 'bakers_inn_price' in df.columns:
        price_col_for_viz = 'retail_price' if 'retail_price' in df.columns else 'bakers_inn_price'
        df_scatter = df[[price_col_for_viz, 'quantity_sold']].dropna()
        if len(df_scatter) > 0:
            if len(df_scatter) > 10000:
                df_scatter = df_scatter.sample(10000, random_state=42)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(x=price_col_for_viz, y='quantity_sold', data=df_scatter, alpha=0.3, ax=ax)
            ax.set_title('Price vs Units Sold per Transaction', fontsize=16, fontweight='bold')
            ax.set_xlabel('Price', fontsize=12)
            ax.set_ylabel('Units Sold', fontsize=12)
            add_caption(fig, 'Explores price sensitivity at the transaction level.')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / 'sales_pos_price_vs_units_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved sales_pos_price_vs_units_scatter.png")

def main():
    """
    Main execution function for Sales POS EDA.
    """
    # Load and prepare data
    df, price_col = load_and_prepare()
    
    # Generate summary statistics
    summary_stats(df, price_col)
    
    # Generate grouped summaries
    grouped_summaries(df)
    
    # Generate visualizations
    visualizations(df)
    
    logging.info("✅ Sales POS EDA complete!")

if __name__ == '__main__':
    main()
