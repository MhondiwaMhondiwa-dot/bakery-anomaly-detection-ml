"""
Exploratory Data Analysis for Quality Control Dataset

This script analyzes QC test results, parameter-specific pass/fail rates, batch quality,
and identifies systemic quality issues at the Shepperton plant.

Key Analyses:
- Overall pass/fail rate trends
- Parameter-specific QC failure analysis
- Batch-level quality metrics (defect concentration, serial failures)
- SKU quality consistency
- Temporal quality patterns (hourly, daily)
- Test value distributions for each parameter
- Root cause analysis for high-failure parameters

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
    """Load and prepare quality control dataset with time features."""
    df = pd.read_parquet(DATA_DIR / 'quality_control_dataset.parquet')
    logger.info(f"Loaded {len(df):,} QC test records")
    
    # Parse timestamps and derive features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    
    # Ensure pass_fail is standardized
    df['is_pass'] = df['pass_fail'].str.lower() == 'pass'
    
    logger.info(f"Final dataset: {len(df):,} QC tests")
    return df

def summary_stats(df):
    """Generate comprehensive QC summary statistics."""
    with open(REPORTS_DIR / 'quality_control_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("QUALITY CONTROL DATASET - EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("=" * 90 + "\n\n")
        
        # Overview
        total_tests = len(df)
        total_pass = df['is_pass'].sum()
        total_fail = (~df['is_pass']).sum()
        overall_pass_rate = (total_pass / total_tests * 100)
        
        f.write("📊 DATASET OVERVIEW\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total QC Tests: {total_tests:,}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Unique Batches: {df['batch_id'].nunique()}\n")
        f.write(f"Unique SKUs: {df['sku'].nunique()}\n")
        f.write(f"Unique Test Parameters: {df['parameter'].nunique()}\n\n")
        
        # Pass/Fail overview
        f.write("✅ PASS/FAIL SUMMARY\n")
        f.write("-" * 90 + "\n")
        f.write(f"Tests Passed: {total_pass:,} ({overall_pass_rate:.2f}%)\n")
        f.write(f"Tests Failed: {total_fail:,} ({100-overall_pass_rate:.2f}%)\n")
        f.write(f"Tests Passed / Tests Failed Ratio: {total_pass/total_fail:.2f}x\n\n")
        
        # Parameter analysis
        f.write("🔬 QUALITY TEST PARAMETERS ANALYSIS\n")
        f.write("-" * 90 + "\n")
        param_stats = df.groupby('parameter').agg({
            'pass_fail': 'count',
            'is_pass': 'sum'
        })
        param_stats.columns = ['Total_Tests', 'Passed_Tests']
        param_stats['Failed_Tests'] = param_stats['Total_Tests'] - param_stats['Passed_Tests']
        param_stats['Pass_Rate_%'] = (param_stats['Passed_Tests'] / param_stats['Total_Tests'] * 100).round(2)
        param_stats['Fail_Rate_%'] = 100 - param_stats['Pass_Rate_%']
        param_stats = param_stats.sort_values('Fail_Rate_%', ascending=False)
        
        f.write(f"Top 10 Failing Parameters:\n")
        for param, row in param_stats.head(10).iterrows():
            f.write(f"  {param}:\n")
            f.write(f"    - Total Tests: {row['Total_Tests']:,.0f}\n")
            f.write(f"    - Pass Rate: {row['Pass_Rate_%']:.2f}%\n")
            f.write(f"    - Failures: {row['Failed_Tests']:,.0f}\n")
        f.write("\n")
        
        # SKU quality
        f.write("🍞 SKU QUALITY METRICS (Top 10 by Fail Rate)\n")
        f.write("-" * 90 + "\n")
        sku_stats = df.groupby('sku').agg({
            'pass_fail': 'count',
            'is_pass': 'sum'
        })
        sku_stats.columns = ['Total_Tests', 'Passed_Tests']
        sku_stats['Failed_Tests'] = sku_stats['Total_Tests'] - sku_stats['Passed_Tests']
        sku_stats['Pass_Rate_%'] = (sku_stats['Passed_Tests'] / sku_stats['Total_Tests'] * 100).round(2)
        sku_stats['Fail_Rate_%'] = 100 - sku_stats['Pass_Rate_%']
        sku_stats = sku_stats.sort_values('Fail_Rate_%', ascending=False)
        
        for sku, row in sku_stats.head(10).iterrows():
            f.write(f"  {sku}: {row['Pass_Rate_%']:.2f}% pass rate ({row['Failed_Tests']:,.0f} failures)\n")
        f.write("\n")
        
        # Batch analysis
        f.write("📦 BATCH QUALITY ANALYSIS\n")
        f.write("-" * 90 + "\n")
        batch_stats = df.groupby('batch_id').agg({
            'pass_fail': 'count',
            'is_pass': 'sum',
            'sku': 'first'
        })
        batch_stats.columns = ['Total_Tests', 'Passed_Tests', 'SKU']
        batch_stats['Failed_Tests'] = batch_stats['Total_Tests'] - batch_stats['Passed_Tests']
        batch_stats['Pass_Rate_%'] = (batch_stats['Passed_Tests'] / batch_stats['Total_Tests'] * 100).round(2)
        
        failed_batches = batch_stats[batch_stats['Passed_Tests'] < batch_stats['Total_Tests']]
        perfect_batches = batch_stats[batch_stats['Passed_Tests'] == batch_stats['Total_Tests']]
        failed_only_batches = batch_stats[batch_stats['Passed_Tests'] == 0]
        
        f.write(f"Total Batches Tested: {len(batch_stats):,}\n")
        f.write(f"Perfect Batches (100% Pass): {len(perfect_batches):,} ({len(perfect_batches)/len(batch_stats)*100:.1f}%)\n")
        f.write(f"Batches with Failures: {len(failed_batches):,} ({len(failed_batches)/len(batch_stats)*100:.1f}%)\n")
        f.write(f"Complete Failures (0% Pass): {len(failed_only_batches):,}\n\n")
        
        # Temporal patterns
        f.write("⏰ TEMPORAL PATTERNS\n")
        f.write("-" * 90 + "\n")
        hourly_stats = df.groupby('hour')['is_pass'].agg(['sum', 'count'])
        hourly_stats.columns = ['Passed', 'Total']
        hourly_stats['Pass_Rate_%'] = (hourly_stats['Passed'] / hourly_stats['Total'] * 100).round(2)
        
        peak_fail_hour = hourly_stats['Pass_Rate_%'].idxmin()
        best_hour = hourly_stats['Pass_Rate_%'].idxmax()
        hour_diff = hourly_stats.loc[best_hour, 'Pass_Rate_%'] - hourly_stats.loc[peak_fail_hour, 'Pass_Rate_%']

        # Only report best/worst hour when there is a meaningful difference
        if hour_diff >= 1.0:
            f.write(f"Worst QC Hour: {peak_fail_hour:02d}:00 ({hourly_stats.loc[peak_fail_hour, 'Pass_Rate_%']:.2f}% pass rate)\n")
            f.write(f"Best QC Hour: {best_hour:02d}:00 ({hourly_stats.loc[best_hour, 'Pass_Rate_%']:.2f}% pass rate)\n")
            f.write(f"Hourly variation: {hour_diff:.2f}% spread between best and worst hour\n")
        else:
            f.write(f"Hourly QC variation: minimal ({hour_diff:.2f}% spread) — no significant hour-of-day quality pattern detected\n")
        
        daily_stats = df.groupby('dayofweek')['is_pass'].agg(['sum', 'count'])
        daily_stats.columns = ['Passed', 'Total']
        daily_stats['Pass_Rate_%'] = (daily_stats['Passed'] / daily_stats['Total'] * 100).round(2)
        
        worst_day = daily_stats['Pass_Rate_%'].idxmin()
        f.write(f"Day with Lowest Pass Rate: {worst_day} ({daily_stats.loc[worst_day, 'Pass_Rate_%']:.2f}%)\n\n")
        
        # Critical findings
        f.write("🎯 CRITICAL FINDINGS & ACTION ITEMS\n")
        f.write("=" * 90 + "\n")
        
        # Critical parameters
        critical_params = param_stats[param_stats['Fail_Rate_%'] > 10]
        if len(critical_params) > 0:
            f.write(f"\n1. 🚨 CRITICAL PARAMETER FAILURES ({len(critical_params)} identified):\n")
            for param, row in critical_params.head(5).iterrows():
                f.write(f"   → {param}: {row['Fail_Rate_%']:.2f}% failure rate ({row['Failed_Tests']:,.0f} failures)\n")
            f.write(f"   ACTION: Investigate test procedure, check calibration, review measurement standards\n")
        
        # SKU issues
        problem_skus = sku_stats[sku_stats['Fail_Rate_%'] > 10]
        if len(problem_skus) > 0:
            f.write(f"\n2. 📦 SKU QUALITY CONCERNS ({len(problem_skus)} identified):\n")
            for sku, row in problem_skus.head(5).iterrows():
                f.write(f"   → {sku}: {row['Fail_Rate_%']:.2f}% failure rate\n")
            f.write(f"   ACTION: Review production recipe, increase QC sampling, audit production line\n")
        
        # Shift quality issues
        bad_hour = hourly_stats['Pass_Rate_%'].idxmin()
        good_hour = hourly_stats['Pass_Rate_%'].idxmax()
        hour_diff = hourly_stats.loc[good_hour, 'Pass_Rate_%'] - hourly_stats.loc[bad_hour, 'Pass_Rate_%']
        
        if hour_diff > 10:
            f.write(f"\n3. ⏰ SHIFT-SPECIFIC QUALITY ISSUES:\n")
            f.write(f"   → Hour {bad_hour:02d}:00 shows {hourly_stats.loc[bad_hour, 'Pass_Rate_%']:.2f}% pass rate (vs {hourly_stats.loc[good_hour, 'Pass_Rate_%']:.2f}% best hour)\n")
            f.write(f"   → {hour_diff:.1f}% difference between worst and best hour\n")
            f.write(f"   ACTION: Check staffing/fatigue, equipment maintenance schedule, QC focus areas\n")
        
        f.write("\n" + "=" * 90 + "\nEND OF REPORT\n" + "=" * 90)
    
    logger.info(f"Wrote quality_control_summary.txt")

def grouped_summaries(df):
    """Generate grouped summary CSV files."""
    
    # By Parameter
    param_summary = df.groupby('parameter').agg({
        'pass_fail': 'count',
        'is_pass': 'sum',
        'value': ['mean', 'std', 'min', 'max']
    })
    param_summary.columns = ['Total_Tests', 'Passed_Tests', 'Avg_Value', 'Std_Value', 'Min_Value', 'Max_Value']
    param_summary['Failed_Tests'] = param_summary['Total_Tests'] - param_summary['Passed_Tests']
    param_summary['Pass_Rate_%'] = (param_summary['Passed_Tests'] / param_summary['Total_Tests'] * 100).round(2)
    param_summary = param_summary.sort_values('Pass_Rate_%', ascending=True)
    param_summary.to_csv(SUMMARIES_DIR / 'qc_by_parameter.csv')
    logger.info("Wrote qc_by_parameter.csv")
    
    # By SKU
    sku_summary = df.groupby('sku').agg({
        'pass_fail': 'count',
        'is_pass': 'sum'
    })
    sku_summary.columns = ['Total_Tests', 'Passed_Tests']
    sku_summary['Failed_Tests'] = sku_summary['Total_Tests'] - sku_summary['Passed_Tests']
    sku_summary['Pass_Rate_%'] = (sku_summary['Passed_Tests'] / sku_summary['Total_Tests'] * 100).round(2)
    sku_summary = sku_summary.sort_values('Pass_Rate_%', ascending=True)
    sku_summary.to_csv(SUMMARIES_DIR / 'qc_by_sku.csv')
    logger.info("Wrote qc_by_sku.csv")
    
    # By Batch
    batch_summary = df.groupby('batch_id').agg({
        'pass_fail': 'count',
        'is_pass': 'sum',
        'sku': 'first'
    })
    batch_summary.columns = ['Total_Tests', 'Passed_Tests', 'SKU']
    batch_summary['Failed_Tests'] = batch_summary['Total_Tests'] - batch_summary['Passed_Tests']
    batch_summary['Pass_Rate_%'] = (batch_summary['Passed_Tests'] / batch_summary['Total_Tests'] * 100).round(2)
    batch_summary = batch_summary.sort_values('Pass_Rate_%', ascending=True).head(100)
    batch_summary.to_csv(SUMMARIES_DIR / 'qc_by_batch_worst100.csv')
    logger.info("Wrote qc_by_batch_worst100.csv")
    
    # By Hour
    hourly = df.groupby('hour').agg({
        'pass_fail': 'count',
        'is_pass': 'sum'
    })
    hourly.columns = ['Total_Tests', 'Passed_Tests']
    hourly['Failed_Tests'] = hourly['Total_Tests'] - hourly['Passed_Tests']
    hourly['Pass_Rate_%'] = (hourly['Passed_Tests'] / hourly['Total_Tests'] * 100).round(2)
    hourly.to_csv(SUMMARIES_DIR / 'qc_by_hour.csv')
    logger.info("Wrote qc_by_hour.csv")
    
    # By Date
    daily = df.groupby('date').agg({
        'pass_fail': 'count',
        'is_pass': 'sum'
    })
    daily.columns = ['Total_Tests', 'Passed_Tests']
    daily['Failed_Tests'] = daily['Total_Tests'] - daily['Passed_Tests']
    daily['Pass_Rate_%'] = (daily['Passed_Tests'] / daily['Total_Tests'] * 100).round(2)
    daily.to_csv(SUMMARIES_DIR / 'qc_by_date.csv')
    logger.info("Wrote qc_by_date.csv")

def visualizations(df):
    """Generate comprehensive visualizations."""
    sns.set_style("whitegrid")

    def add_caption(fig, text):
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    # 1. Overall pass/fail pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    pass_counts = df['is_pass'].value_counts()
    labels = ['Pass', 'Fail']
    colors = ['green', 'red']
    explode = (0.05, 0.05)
    
    ax.pie(pass_counts, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors, explode=explode, textprops={'fontsize': 12, 'weight': 'bold'})
    ax.set_title('Overall QC Pass/Fail Rate', fontweight='bold', fontsize=14)
    add_caption(fig, 'Summarizes overall QC success rate for quick baseline tracking.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_overall_pass_fail.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_overall_pass_fail.png")
    
    # 2. Parameter-specific fail rates
    param_stats = df.groupby('parameter')['is_pass'].agg(['sum', 'count'])
    param_stats.columns = ['Passed', 'Total']
    param_stats['Fail_Rate_%'] = (1 - param_stats['Passed'] / param_stats['Total']) * 100
    param_stats = param_stats.sort_values('Fail_Rate_%', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_bar = ['green' if x < 5 else 'orange' if x < 10 else 'red' for x in param_stats['Fail_Rate_%']]
    param_stats['Fail_Rate_%'].plot(kind='barh', ax=ax, color=colors_bar)
    ax.set_xlabel('Failure Rate (%)', fontweight='bold')
    ax.set_title('QC Test Parameter Failure Rates', fontweight='bold', fontsize=13)
    ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='Critical: 10%')
    ax.legend()
    add_caption(fig, 'Highlights the parameters driving QC failures for focused action.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_parameter_fail_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_parameter_fail_rates.png")
    
    # 3. Hourly pass rate trend
    fig, ax = plt.subplots(figsize=(12, 6))
    hourly = df.groupby('hour')['is_pass'].agg(['sum', 'count'])
    hourly['rate'] = hourly['sum'] / hourly['count'] * 100
    
    ax.plot(hourly.index, hourly['rate'], marker='o', linewidth=2, markersize=8, color='darkblue')
    ax.fill_between(hourly.index, hourly['rate'], alpha=0.3, color='steelblue')
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontweight='bold')
    ax.set_title('Hourly QC Pass Rate Trend', fontweight='bold', fontsize=13)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    ax.axhline(df['is_pass'].mean() * 100, color='red', linestyle='--', label='Daily Average')
    ax.legend()
    add_caption(fig, 'Shows pass-rate swings by hour to isolate shift-level issues.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_hourly_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_hourly_trend.png")
    
    # 4. Daily pass rate
    fig, ax = plt.subplots(figsize=(14, 6))
    daily = df.groupby('date')['is_pass'].agg(['sum', 'count']).sort_index()
    daily['rate'] = daily['sum'] / daily['count'] * 100
    
    ax.plot(range(len(daily)), daily['rate'].values, marker='o', linewidth=2, color='darkgreen', markersize=4)
    ax.fill_between(range(len(daily)), daily['rate'].values, alpha=0.3, color='lightgreen')
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontweight='bold')
    ax.set_title('Daily QC Pass Rate Trend', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(df['is_pass'].mean() * 100, color='red', linestyle='--', label='Overall Average')
    ax.legend()
    add_caption(fig, 'Tracks daily QC stability and identifies sustained dips.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_daily_trend.png")
    
    # 5. SKU quality comparison
    sku_stats = df.groupby('sku')['is_pass'].agg(['sum', 'count'])
    sku_stats['rate'] = sku_stats['sum'] / sku_stats['count'] * 100
    sku_stats = sku_stats.sort_values('rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_sku = ['red' if x < 80 else 'orange' if x < 90 else 'green' for x in sku_stats['rate']]
    sku_stats['rate'].plot(kind='barh', ax=ax, color=colors_sku)
    ax.set_xlabel('Pass Rate (%)', fontweight='bold')
    ax.set_title('SKU Quality Metrics (Pass Rate)', fontweight='bold', fontsize=13)
    ax.axvline(90, color='orange', linestyle='--', alpha=0.5)
    add_caption(fig, 'Compares SKU-level quality to spot consistently weak products.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_by_sku.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_by_sku.png")
    
    # 6. Test value distributions (for numeric parameters)
    numeric_params = df[df['value'].notna()].groupby('parameter')['value'].nunique().nlargest(3).index
    
    if len(numeric_params) > 0:
        fig, axes = plt.subplots(1, min(3, len(numeric_params)), figsize=(15, 5))
        if len(numeric_params) == 1:
            axes = [axes]
        
        for i, param in enumerate(numeric_params):
            param_data = df[df['parameter'] == param]
            axes[i].hist(param_data['value'].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{param}', fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Test Value Distributions (Top 3 Parameters)', fontweight='bold', fontsize=13)
        add_caption(fig, 'Reveals drift or bimodal behavior in key QC measurements.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'qc_parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved qc_parameter_distributions.png")
    
    # 7. Batch quality composition (pass/fail/mixed)
    batch_stats = df.groupby('batch_id')['is_pass'].agg(['sum', 'count'])
    batch_stats['status'] = batch_stats.apply(
        lambda x: 'Perfect' if x['sum'] == x['count'] else 'Failed' if x['sum'] == 0 else 'Mixed',
        axis=1
    )
    
    status_counts = batch_stats['status'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_status = {'Perfect': 'green', 'Mixed': 'orange', 'Failed': 'red'}
    colors_list = [colors_status.get(s, 'gray') for s in status_counts.index]
    
    ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
           colors=colors_list, explode=(0.05, 0.05, 0.05), textprops={'fontsize': 11, 'weight': 'bold'})
    ax.set_title('Batch Quality Composition', fontweight='bold', fontsize=13)
    add_caption(fig, 'Shows how many batches are perfect, mixed, or failed.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_batch_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_batch_composition.png")

    # 8. QC checks per batch (histogram)
    checks_per_batch = df.groupby('batch_id')['parameter'].count()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(checks_per_batch.values, bins=range(1, int(checks_per_batch.max()) + 2),
        color='slateblue', edgecolor='black', alpha=0.8)
    ax.axvline(checks_per_batch.mean(), color='red', linestyle='--',
            label=f'Mean: {checks_per_batch.mean():.1f}')
    ax.axvline(checks_per_batch.median(), color='green', linestyle='--',
            label=f'Median: {checks_per_batch.median():.0f}')
    ax.set_xlabel('QC Checks per Batch', fontweight='bold')
    ax.set_ylabel('Number of Batches', fontweight='bold')
    ax.set_title('QC Checks Per Batch (Intensity Distribution)', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_caption(fig, 'Monitors QC intensity to ensure consistent inspection coverage.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_checks_per_batch_hist.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_checks_per_batch_hist.png")

    # 9. Pass rate heatmap by day of week and hour
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heat = df.pivot_table(index='dayofweek', columns='hour', values='is_pass', aggfunc='mean') * 100
    heat = heat.reindex(days)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat, cmap='coolwarm', center=heat.stack().median(skipna=True), ax=ax)
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_ylabel('Day of Week', fontweight='bold')
    ax.set_title('QC Pass Rate (%) by Day and Hour', fontweight='bold', fontsize=13)
    add_caption(fig, 'Pinpoints shift/day combinations with unusually low pass rates.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_pass_rate_heatmap_day_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_pass_rate_heatmap_day_hour.png")

    # 10. Daily pass rate trend for top failing parameters
    top_params = param_stats.sort_values('Fail_Rate_%', ascending=False).head(5).index
    param_daily = df[df['parameter'].isin(top_params)].groupby(['date', 'parameter'])['is_pass'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for param in top_params:
        series = param_daily[param_daily['parameter'] == param]
        ax.plot(series['date'], series['is_pass'] * 100, linewidth=1.8, label=param)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontweight='bold')
    ax.set_title('Daily Pass Rate for Top Failing Parameters', fontweight='bold', fontsize=13)
    ax.legend()
    add_caption(fig, 'Tracks whether key parameter failures are improving or worsening.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_top_params_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_top_params_daily_trend.png")

    # 11. SKU x parameter fail rate heatmap (top volumes)
    top_skus = df['sku'].value_counts().head(8).index
    top_params2 = df['parameter'].value_counts().head(8).index
    df_sp = df[df['sku'].isin(top_skus) & df['parameter'].isin(top_params2)]
    if not df_sp.empty:
        sp = df_sp.groupby(['sku', 'parameter'])['is_pass'].mean().unstack('parameter')
        sp = (1 - sp) * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(sp, cmap='coolwarm', center=sp.stack().median(skipna=True), annot=True, fmt='.1f', ax=ax)
        ax.set_xlabel('Parameter', fontweight='bold')
        ax.set_ylabel('SKU', fontweight='bold')
        ax.set_title('QC Fail Rate (%) by SKU and Parameter', fontweight='bold', fontsize=13)
        add_caption(fig, 'Shows SKU/parameter combinations driving failures.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'qc_fail_rate_sku_parameter_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved qc_fail_rate_sku_parameter_heatmap.png")

    # 12. Value distributions by pass/fail for top numeric parameters
    df_val = df.copy()
    df_val['value'] = pd.to_numeric(df_val['value'], errors='coerce')
    numeric_params2 = df_val[df_val['value'].notna()].groupby('parameter')['value'].nunique().nlargest(3).index
    if len(numeric_params2) > 0:
        subset = df_val[df_val['parameter'].isin(numeric_params2) & df_val['value'].notna()]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='parameter', y='value', hue='is_pass', data=subset, ax=ax)
        ax.set_xlabel('Parameter', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('QC Values by Pass/Fail (Top Numeric Parameters)', fontweight='bold', fontsize=13)
        ax.legend(title='Pass')
        add_caption(fig, 'Compares value distributions for pass vs fail to surface thresholds.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'qc_value_by_pass_fail_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved qc_value_by_pass_fail_boxplot.png")

def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("Starting Quality Control Dataset EDA")
    logger.info("=" * 80)
    
    df = load_and_prepare()
    summary_stats(df)
    grouped_summaries(df)
    visualizations(df)
    
    logger.info("=" * 80)
    logger.info("✅ Quality Control EDA complete!")
    logger.info(f"   - Summary: reports/quality_control_summary.txt")
    logger.info(f"   - Figures: reports/figures/qc_*.png (7 visualizations)")
    logger.info(f"   - CSVs: reports/summaries/qc_*.csv (5 summary files)")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
