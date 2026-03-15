"""
Exploratory Data Analysis for Production Dataset

This script analyzes production operations, defect patterns, quality issues, 
line performance, and operator productivity at the Shepperton plant.

Key Analyses:
- Production volume by line, SKU, operator, hour
- Defect rates and failure type distribution  
- Quality issues analysis: stacked, squashed, torn, undersized, valleys, loose packs, pale/underbaked
- Temporal patterns (hourly, daily trends)
- Line and operator performance comparisons
- Root cause analysis for high-defect batches

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
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
SUMMARIES_DIR = REPORTS_DIR / 'summaries'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare():
    """Load and prepare production dataset with time features."""
    df = pd.read_parquet(DATA_DIR / 'production_dataset.parquet')
    logger.info(f"Loaded {len(df):,} production records")
    
    # Parse timestamps and derive features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    
    # Compute defect metrics
    defect_cols = ['stacked_before_robot', 'squashed', 'torn', 'undersized_small', 
                   'valleys', 'loose_packs', 'pale_underbaked']
    df['total_defects'] = df[defect_cols].sum(axis=1)
    df['defect_rate'] = (df['total_defects'] / df['quantity_produced'] * 100).fillna(0)
    
    # Filter to Shepperton plant
    shep_count = (df['plant'] == 'Shepperton_Plant').sum()
    if shep_count > 0:
        df = df[df['plant'] == 'Shepperton_Plant'].copy()
    
    logger.info(f"Final dataset: {len(df):,} records from {df['plant'].nunique()} plant(s)")
    return df

def summary_stats(df):
    """Generate comprehensive summary statistics."""
    defect_cols = ['stacked_before_robot', 'squashed', 'torn', 'undersized_small', 
                   'valleys', 'loose_packs', 'pale_underbaked']
    
    with open(REPORTS_DIR / 'production_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("PRODUCTION DATASET - EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("=" * 90 + "\n\n")
        
        # Overview
        f.write("📊 DATASET OVERVIEW\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total Production Batches: {len(df):,}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Total Units Produced: {df['quantity_produced'].sum():,}\n")
        f.write(f"Average Batch Size: {df['quantity_produced'].mean():.1f} units\n")
        f.write(f"Production Lines: {df['line_id'].nunique()}\n")
        f.write(f"SKUs: {df['sku'].nunique()}\n")
        f.write(f"Operators: {df['operator_id'].nunique()}\n")
        f.write(f"Depots: {df['depot_id'].nunique()}\n\n")
        
        # Defect overview
        total_defects = df[defect_cols].sum().sum()
        overall_defect_rate = (total_defects / df['quantity_produced'].sum() * 100)
        
        f.write("🚨 DEFECT OVERVIEW\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total Defects Detected: {int(total_defects):,}\n")
        f.write(f"Overall Defect Rate: {overall_defect_rate:.2f}%\n")
        f.write(f"Batches with Defects: {(df['total_defects'] > 0).sum():,} ({(df['total_defects'] > 0).sum()/len(df)*100:.1f}%)\n")
        f.write(f"Average Defects per Batch: {df['total_defects'].mean():.2f}\n")
        f.write(f"Maximum Defects in Single Batch: {df['total_defects'].max():.0f}\n\n")
        
        # Defect types
        f.write("🔴 Defect Types (Highest to Lowest):\n")
        defect_totals = df[defect_cols].sum().sort_values(ascending=False)
        for defect_type, count in defect_totals.items():
            pct = (count / total_defects * 100) if total_defects > 0 else 0
            f.write(f"  - {defect_type.replace('_', ' ').title()}: {int(count):,} ({pct:.1f}%)\n")
        f.write("\n")
        
        # Line performance
        f.write("🏭 PRODUCTION LINE PERFORMANCE\n")
        f.write("-" * 90 + "\n")
        line_stats = df.groupby('line_id').agg({
            'quantity_produced': ['sum', 'count', 'mean'],
            'total_defects': 'sum'
        }).round(2)
        line_stats.columns = ['Total_Units', 'Batches', 'Avg_Batch', 'Total_Defects']
        line_stats['Defect_Rate_%'] = (line_stats['Total_Defects'] / line_stats['Total_Units'] * 100).round(2)
        line_stats = line_stats.sort_values('Defect_Rate_%', ascending=False)
        
        for line, row in line_stats.head(10).iterrows():
            f.write(f"  {line}: {row['Total_Units']:,.0f} units, {row['Batches']:.0f} batches, {row['Defect_Rate_%']:.2f}% defect rate\n")
        f.write("\n")
        
        # SKU performance
        f.write("🍞 SKU PERFORMANCE (Top 10 by Defect Rate)\n")
        f.write("-" * 90 + "\n")
        sku_stats = df.groupby('sku').agg({
            'quantity_produced': ['sum', 'count'],
            'total_defects': 'sum'
        }).round(2)
        sku_stats.columns = ['Total_Units', 'Batches', 'Total_Defects']
        sku_stats['Defect_Rate_%'] = (sku_stats['Total_Defects'] / sku_stats['Total_Units'] * 100).round(2)
        sku_stats = sku_stats.sort_values('Defect_Rate_%', ascending=False)
        
        for sku, row in sku_stats.head(10).iterrows():
            f.write(f"  {sku}: {row['Total_Units']:,.0f} units, {row['Defect_Rate_%']:.2f}% defect rate\n")
        f.write("\n")
        
        # Operator performance
        f.write("👤 OPERATOR PERFORMANCE (Top 10 by Defect Rate)\n")
        f.write("-" * 90 + "\n")
        op_stats = df.groupby('operator_id').agg({
            'quantity_produced': ['sum', 'count'],
            'total_defects': 'sum'
        }).round(2)
        op_stats.columns = ['Total_Units', 'Batches', 'Total_Defects']
        op_stats['Defect_Rate_%'] = (op_stats['Total_Defects'] / op_stats['Total_Units'] * 100).round(2)
        op_stats = op_stats.sort_values('Defect_Rate_%', ascending=False)
        
        for op, row in op_stats.head(10).iterrows():
            f.write(f"  {op}: {row['Batches']:.0f} batches, {row['Defect_Rate_%']:.2f}% defect rate\n")
        f.write("\n")
        
        # Temporal patterns
        f.write("⏰ TEMPORAL PATTERNS\n")
        f.write("-" * 90 + "\n")
        hourly_stats = df.groupby('hour').agg({'quantity_produced': 'sum', 'total_defects': 'sum'})
        hourly_stats['defect_rate_%'] = (hourly_stats['total_defects'] / hourly_stats['quantity_produced'] * 100).round(2)
        
        peak_hour = hourly_stats['quantity_produced'].idxmax()
        worst_hour = hourly_stats['defect_rate_%'].idxmax()
        
        f.write(f"Peak Production Hour: {int(peak_hour):02d}:00 ({hourly_stats.loc[peak_hour, 'quantity_produced']:,.0f} units)\n")
        f.write(f"Worst Defect Hour: {int(worst_hour):02d}:00 ({hourly_stats.loc[worst_hour, 'defect_rate_%']:.2f}%)\n")
        
        daily_stats = df.groupby('dayofweek').agg({'quantity_produced': 'sum', 'total_defects': 'sum'})
        daily_stats['defect_rate_%'] = (daily_stats['total_defects'] / daily_stats['quantity_produced'] * 100).round(2)
        
        f.write(f"Highest Production Day: {daily_stats['quantity_produced'].idxmax()}\n")
        f.write(f"Highest Defect Rate Day: {daily_stats['defect_rate_%'].idxmax()} ({daily_stats['defect_rate_%'].max():.2f}%)\n\n")
        
        # Key findings
        f.write("🎯 CRITICAL FINDINGS & ACTION ITEMS\n")
        f.write("=" * 90 + "\n")
        
        high_defect_lines = line_stats[line_stats['Defect_Rate_%'] > 15]
        if len(high_defect_lines) > 0:
            f.write(f"\n1. ⚠️ HIGH DEFECT LINES ({len(high_defect_lines)} identified):\n")
            for line, rate in high_defect_lines['Defect_Rate_%'].items():
                f.write(f"   → {line}: {rate:.2f}% defect rate\n")
            f.write(f"   ACTION: Audit equipment, check calibration, inspect material quality\n")
        
        high_defect_skus = sku_stats[sku_stats['Defect_Rate_%'] > 15]
        if len(high_defect_skus) > 0:
            f.write(f"\n2. 🚨 PROBLEMATIC SKUs ({len(high_defect_skus)} identified):\n")
            for sku, rate in high_defect_skus['Defect_Rate_%'].items():
                f.write(f"   → {sku}: {rate:.2f}% defect rate\n")
            f.write(f"   ACTION: Review recipe/process parameters, increase QC sampling\n")
        
        high_defect_ops = op_stats[op_stats['Defect_Rate_%'] > 20]
        if len(high_defect_ops) > 0:
            f.write(f"\n3. 👤 OPERATORS WITH QUALITY ISSUES ({len(high_defect_ops)} identified):\n")
            for op, row in high_defect_ops.iterrows():
                f.write(f"   → {op}: {row['Defect_Rate_%']:.2f}% defect rate\n")
            f.write(f"   ACTION: Retraining program, pair with experienced operator, investigate work environment\n")
        
        f.write("\n" + "=" * 90 + "\nEND OF REPORT\n" + "=" * 90)
    
    logger.info(f"Wrote production_summary.txt")

def grouped_summaries(df):
    """Generate grouped summary CSV files."""
    defect_cols = ['stacked_before_robot', 'squashed', 'torn', 'undersized_small', 
                   'valleys', 'loose_packs', 'pale_underbaked']
    
    # By Line
    line_summary = df.groupby('line_id').agg({'quantity_produced': ['sum', 'count', 'mean'], 'total_defects': 'sum'})
    line_summary.columns = ['Total_Units', 'Batches', 'Avg_Batch_Size', 'Total_Defects']
    line_summary['Defect_Rate_%'] = (line_summary['Total_Defects'] / line_summary['Total_Units'] * 100).round(2)
    line_summary = line_summary.sort_values('Defect_Rate_%', ascending=False)
    line_summary.to_csv(SUMMARIES_DIR / 'production_by_line.csv')
    logger.info("Wrote production_by_line.csv")
    
    # By SKU
    sku_summary = df.groupby('sku').agg({'quantity_produced': ['sum', 'count'], 'total_defects': 'sum'})
    sku_summary.columns = ['Total_Units', 'Batches', 'Total_Defects']
    sku_summary['Defect_Rate_%'] = (sku_summary['Total_Defects'] / sku_summary['Total_Units'] * 100).round(2)
    sku_summary = sku_summary.sort_values('Defect_Rate_%', ascending=False)
    sku_summary.to_csv(SUMMARIES_DIR / 'production_by_sku.csv')
    logger.info("Wrote production_by_sku.csv")
    
    # By Operator (top 50)
    op_summary = df.groupby('operator_id').agg({'quantity_produced': ['sum', 'count'], 'total_defects': 'sum'})
    op_summary.columns = ['Total_Units', 'Batches', 'Total_Defects']
    op_summary['Defect_Rate_%'] = (op_summary['Total_Defects'] / op_summary['Total_Units'] * 100).round(2)
    op_summary = op_summary.sort_values('Defect_Rate_%', ascending=False).head(50)
    op_summary.to_csv(SUMMARIES_DIR / 'production_by_operator_top50.csv')
    logger.info("Wrote production_by_operator_top50.csv")
    
    # By Hour
    hourly = df.groupby('hour').agg({'quantity_produced': ['sum', 'count'], 'total_defects': 'sum'})
    hourly.columns = ['Total_Units', 'Batches', 'Total_Defects']
    hourly['Defect_Rate_%'] = (hourly['Total_Defects'] / hourly['Total_Units'] * 100).round(2)
    hourly.to_csv(SUMMARIES_DIR / 'production_by_hour.csv')
    logger.info("Wrote production_by_hour.csv")
    
    # By Date
    daily = df.groupby('date').agg({'quantity_produced': ['sum', 'count'], 'total_defects': 'sum'})
    daily.columns = ['Total_Units', 'Batches', 'Total_Defects']
    daily['Defect_Rate_%'] = (daily['Total_Defects'] / daily['Total_Units'] * 100).round(2)
    daily.to_csv(SUMMARIES_DIR / 'production_by_date.csv')
    logger.info("Wrote production_by_date.csv")
    
    # Defect breakdown
    defect_summary = df[defect_cols].sum().reset_index()
    defect_summary.columns = ['Defect_Type', 'Count']
    defect_summary['Percentage'] = (defect_summary['Count'] / defect_summary['Count'].sum() * 100).round(2)
    defect_summary = defect_summary.sort_values('Count', ascending=False)
    defect_summary.to_csv(SUMMARIES_DIR / 'production_defect_breakdown.csv', index=False)
    logger.info("Wrote production_defect_breakdown.csv")

def visualizations(df):
    """Generate comprehensive visualizations."""
    defect_cols = ['stacked_before_robot', 'squashed', 'torn', 'undersized_small', 
                   'valleys', 'loose_packs', 'pale_underbaked']

    def add_caption(fig, text):
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    sns.set_style("whitegrid")
    
    # 1. Production by line
    fig, ax = plt.subplots(figsize=(12, 6))
    line_prod = df.groupby('line_id')['quantity_produced'].sum().sort_values(ascending=True)
    line_prod.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Total Units Produced', fontweight='bold')
    ax.set_title('Total Production Volume by Line', fontweight='bold', fontsize=13)
    add_caption(fig, 'Shows which lines contribute most to total output for capacity focus.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_by_line.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_by_line.png")
    
    # 2. Defect rate by line
    fig, ax = plt.subplots(figsize=(12, 6))
    line_defects = df.groupby('line_id').apply(
        lambda x: (x['total_defects'].sum() / x['quantity_produced'].sum() * 100)
    ).sort_values(ascending=True)
    colors = ['red' if x > 15 else 'orange' if x > 10 else 'green' for x in line_defects]
    line_defects.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Defect Rate (%)', fontweight='bold')
    ax.set_title('Production Line Quality (Defect Rate %)', fontweight='bold', fontsize=13)
    ax.axvline(10, color='orange', linestyle='--', alpha=0.5)
    add_caption(fig, 'Highlights lines exceeding defect thresholds for targeted fixes.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_defect_rate_by_line.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_defect_rate_by_line.png")
    
    # 3. Defect types breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    defect_totals = df[defect_cols].sum().sort_values(ascending=True)
    defect_totals.plot(kind='barh', ax=ax, color='crimson')
    ax.set_xlabel('Count', fontweight='bold')
    ax.set_title('Total Defects by Type', fontweight='bold', fontsize=13)
    add_caption(fig, 'Reveals which defect types dominate overall quality losses.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_defect_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_defect_breakdown.png")
    
    # 4. Production by SKU
    fig, ax = plt.subplots(figsize=(12, 6))
    sku_prod = df.groupby('sku')['quantity_produced'].sum().sort_values(ascending=True)
    sku_prod.plot(kind='barh', ax=ax, color='seagreen')
    ax.set_xlabel('Total Units Produced', fontweight='bold')
    ax.set_title('Total Production Volume by SKU', fontweight='bold', fontsize=13)
    add_caption(fig, 'Shows production mix across SKUs to align capacity and demand.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_by_sku.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_by_sku.png")
    
    # 5. Hourly pattern
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_prod = df.groupby('hour')['quantity_produced'].sum()
    hourly_defect = df.groupby('hour')['total_defects'].sum()
    
    ax2 = ax.twinx()
    ax.bar(hourly_prod.index, hourly_prod.values, alpha=0.6, color='steelblue', label='Units')
    ax2.plot(hourly_defect.index, hourly_defect.values, color='red', marker='o', linewidth=2, label='Defects')
    
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Units Produced', fontweight='bold', color='steelblue')
    ax2.set_ylabel('Total Defects', fontweight='bold', color='red')
    ax.set_title('Hourly Production Volume & Defects', fontweight='bold', fontsize=13)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    add_caption(fig, 'Compares hourly output against defects to spot risky shifts.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_hourly_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_hourly_pattern.png")
    
    # 6. Daily trend
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_prod = df.groupby('date')['quantity_produced'].sum().sort_index()
    daily_defect_rate = df.groupby('date').apply(
        lambda x: (x['total_defects'].sum() / x['quantity_produced'].sum() * 100)
    ).sort_index()
    
    ax2 = ax.twinx()
    ax.fill_between(range(len(daily_prod)), daily_prod.values, alpha=0.3, color='steelblue')
    ax.plot(daily_prod.values, color='darkblue', linewidth=2, label='Production')
    ax2.plot(daily_defect_rate.values, color='red', linewidth=2, marker='o', label='Defect Rate %')
    
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Units Produced', fontweight='bold', color='darkblue')
    ax2.set_ylabel('Defect Rate (%)', fontweight='bold', color='red')
    ax.set_title('Daily Production Volume & Quality Trend', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    add_caption(fig, 'Tracks daily output and defect rate to detect sustained drift.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_daily_trend.png")
    
    # 7. Defect rate distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    df['defect_rate'].hist(bins=50, ax=ax, color='salmon', edgecolor='black', alpha=0.7)
    ax.axvline(df['defect_rate'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['defect_rate'].mean():.2f}%")
    ax.set_xlabel('Defect Rate (%)', fontweight='bold')
    ax.set_ylabel('Frequency (Batches)', fontweight='bold')
    ax.set_title('Distribution of Batch Defect Rates', fontweight='bold', fontsize=13)
    ax.legend()
    add_caption(fig, 'Shows spread of batch quality to identify outlier-heavy periods.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_defect_rate_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_defect_rate_distribution.png")

    # 8. Defect rate heatmap by day of week and hour
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heat = df.groupby(['dayofweek', 'hour']).agg({'total_defects': 'sum', 'quantity_produced': 'sum'}).reset_index()
    heat['defect_rate'] = (heat['total_defects'] / heat['quantity_produced'] * 100).replace([np.inf, -np.inf], np.nan)
    heat_pivot = heat.pivot(index='dayofweek', columns='hour', values='defect_rate').reindex(days)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat_pivot, cmap='coolwarm', center=heat_pivot.stack().median(skipna=True), ax=ax)
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Day of Week', fontweight='bold')
    ax.set_title('Defect Rate (%) by Day and Hour', fontweight='bold', fontsize=13)
    add_caption(fig, 'Pinpoints shift/day combinations with the worst defect rates.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_defect_rate_heatmap_day_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_defect_rate_heatmap_day_hour.png")

    # 9. Batch size vs defect rate scatter
    df_scatter = df[['quantity_produced', 'defect_rate']].dropna()
    if len(df_scatter) > 10000:
        df_scatter = df_scatter.sample(10000, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x='quantity_produced', y='defect_rate', data=df_scatter, alpha=0.3, ax=ax)
    ax.set_xlabel('Batch Size (Units)', fontweight='bold')
    ax.set_ylabel('Defect Rate (%)', fontweight='bold')
    ax.set_title('Batch Size vs Defect Rate', fontweight='bold', fontsize=13)
    add_caption(fig, 'Checks whether larger batches correlate with higher defect rates.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_batch_size_vs_defect_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_batch_size_vs_defect_rate.png")

    # 10. Defect Pareto with cumulative percentage
    defect_totals = df[defect_cols].sum().sort_values(ascending=False)
    pareto = defect_totals.reset_index()
    pareto.columns = ['defect_type', 'count']
    pareto['cum_pct'] = pareto['count'].cumsum() / pareto['count'].sum() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(pareto['defect_type'], pareto['count'], color='crimson', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(pareto['defect_type'], pareto['cum_pct'], color='black', marker='o')
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.6)
    ax.set_xlabel('Defect Type', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax2.set_ylabel('Cumulative %', fontweight='bold')
    ax.set_title('Defect Pareto (Counts and Cumulative %)', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=30)
    add_caption(fig, 'Identifies the few defect types causing most of the issues.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_defect_pareto.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_defect_pareto.png")

    # 11. Line x SKU defect rate heatmap (top lines and SKUs)
    top_lines = df.groupby('line_id')['quantity_produced'].sum().nlargest(8).index
    top_skus = df.groupby('sku')['quantity_produced'].sum().nlargest(8).index
    df_ls = df[df['line_id'].isin(top_lines) & df['sku'].isin(top_skus)]
    if not df_ls.empty:
        ls = df_ls.groupby(['line_id', 'sku']).agg({'total_defects': 'sum', 'quantity_produced': 'sum'})
        ls['defect_rate'] = (ls['total_defects'] / ls['quantity_produced'] * 100).replace([np.inf, -np.inf], np.nan)
        ls_pivot = ls['defect_rate'].unstack('sku')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(ls_pivot, cmap='coolwarm', center=ls_pivot.stack().median(skipna=True), annot=True, fmt='.1f', ax=ax)
        ax.set_xlabel('SKU', fontweight='bold')
        ax.set_ylabel('Line ID', fontweight='bold')
        ax.set_title('Defect Rate (%) by Line and SKU (Top Volumes)', fontweight='bold', fontsize=13)
        add_caption(fig, 'Shows line/SKU combinations with consistently high defect rates.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'production_defect_rate_line_sku_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved production_defect_rate_line_sku_heatmap.png")

    # 12. Rolling 7-day defect rate trend
    daily = df.groupby('date').agg({'total_defects': 'sum', 'quantity_produced': 'sum'}).sort_index()
    daily['defect_rate'] = daily['total_defects'] / daily['quantity_produced'] * 100
    daily['defect_rate_7d'] = daily['defect_rate'].rolling(7, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily['defect_rate'], color='lightcoral', alpha=0.6, label='Daily')
    ax.plot(daily.index, daily['defect_rate_7d'], color='darkred', linewidth=2, label='7-day rolling')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Defect Rate (%)', fontweight='bold')
    ax.set_title('Rolling Defect Rate Trend (7-day)', fontweight='bold', fontsize=13)
    ax.legend()
    add_caption(fig, 'Smooths daily noise to reveal sustained quality deterioration.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'production_defect_rate_rolling_7d.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved production_defect_rate_rolling_7d.png")

def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("Starting Production Dataset EDA")
    logger.info("=" * 80)
    
    df = load_and_prepare()
    summary_stats(df)
    grouped_summaries(df)
    visualizations(df)
    
    logger.info("=" * 80)
    logger.info("✅ Production EDA complete!")
    logger.info(f"   - Summary: reports/production_summary.txt")
    logger.info(f"   - Figures: reports/figures/production_*.png (7 visualizations)")
    logger.info(f"   - CSVs: reports/summaries/production_*.csv (6 summary files)")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
