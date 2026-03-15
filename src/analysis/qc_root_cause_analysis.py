"""QC Root Cause Analysis - Identify why 38.3% of QC tests are failing.

This script analyzes the Quality Control dataset to identify patterns in failures:
- Which parameters fail most often?
- Which batches/SKUs have highest fail rates?
- Are there operator/equipment/time patterns?
- What correlations exist with production defects?

Goal: Generate actionable insights to reduce fail rate from 38.3% to target <10%
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports')
FIGURES_DIR = REPORTS_DIR / 'figures'
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    """Load QC data and related datasets for correlation analysis."""
    qc = pd.read_parquet(DATA_DIR / 'quality_control_dataset.parquet')
    qc['timestamp'] = pd.to_datetime(qc['timestamp'], errors='coerce')
    
    # Load production for batch-level correlation
    prod = pd.read_parquet(DATA_DIR / 'production_dataset.parquet')
    prod['timestamp'] = pd.to_datetime(prod['timestamp'], errors='coerce')
    
    # Load waste to see QC failure → waste correlation
    waste = pd.read_parquet(DATA_DIR / 'waste_dataset.parquet')
    waste['timestamp'] = pd.to_datetime(waste.get('timestamp'), errors='coerce')
    
    return qc, prod, waste


def root_cause_analysis(qc, prod, waste):
    """Comprehensive root cause analysis for QC failures."""
    
    report_path = REPORTS_DIR / 'QC_ROOT_CAUSE_ANALYSIS.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("QUALITY CONTROL FAILURE ROOT CAUSE ANALYSIS\n")
        f.write("GOAL: Reduce fail rate from 38.3% to target <10%\n")
        f.write("=" * 100 + "\n\n")
        
        # Overall statistics
        total_tests = len(qc)
        failures = (qc['pass_fail'] == 'fail').sum()
        fail_rate = failures / total_tests * 100
        
        f.write("🎯 CURRENT STATE\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total QC Tests: {total_tests:,}\n")
        f.write(f"Failed Tests: {failures:,}\n")
        f.write(f"FAIL RATE: {fail_rate:.2f}% (TARGET: <10%)\n")
        f.write(f"GAP TO TARGET: {fail_rate - 10:.2f} percentage points\n")
        f.write(f"TESTS THAT SHOULD NOT FAIL: {int((fail_rate - 10) / 100 * total_tests):,} tests/year\n\n")
        
        f.write(f"⚠️ SEVERITY: {'CRITICAL' if fail_rate > 30 else 'HIGH' if fail_rate > 20 else 'MEDIUM'}\n")
        f.write(f"   A {fail_rate:.1f}% fail rate means nearly 4 out of 10 products fail quality standards.\n")
        f.write(f"   This impacts customer satisfaction, waste, and returns.\n\n")
        
        # 1. PARAMETER-SPECIFIC ANALYSIS
        f.write("=" * 100 + "\n")
        f.write("1. FAILURE BREAKDOWN BY PARAMETER\n")
        f.write("=" * 100 + "\n\n")
        
        param_analysis = qc.groupby('parameter').agg({
            'qc_id': 'count',
            'pass_fail': lambda x: (x == 'fail').sum()
        }).rename(columns={'qc_id': 'total_tests', 'pass_fail': 'failures'})
        param_analysis['fail_rate_%'] = (param_analysis['failures'] / param_analysis['total_tests'] * 100).round(2)
        param_analysis['gap_to_target_%'] = param_analysis['fail_rate_%'] - 10
        param_analysis = param_analysis.sort_values('fail_rate_%', ascending=False)
        
        f.write("Parameter Performance:\n")
        f.write(f"{'Parameter':<20} {'Tests':>8} {'Failures':>10} {'Fail Rate':>12} {'Gap to 10%':>15}\n")
        f.write("-" * 100 + "\n")
        for param, row in param_analysis.iterrows():
            status = "🔴 CRITICAL" if row['fail_rate_%'] > 30 else "🟡 HIGH" if row['fail_rate_%'] > 20 else "🟢 WARNING"
            f.write(f"{param:<20} {row['total_tests']:>8} {row['failures']:>10} {row['fail_rate_%']:>11.1f}% {row['gap_to_target_%']:>14.1f}% {status}\n")
        
        f.write("\n🎯 PARAMETER PRIORITIES:\n")
        worst_param = param_analysis.index[0]
        worst_rate = param_analysis.iloc[0]['fail_rate_%']
        f.write(f"1. **{worst_param}**: {worst_rate:.1f}% fail rate - HIGHEST PRIORITY\n")
        f.write(f"   Action: Investigate equipment calibration, measurement methodology, spec limits\n\n")
        
        for i, (param, row) in enumerate(param_analysis.head(3).iloc[1:].iterrows(), 2):
            f.write(f"{i}. **{param}**: {row['fail_rate_%']:.1f}% fail rate\n")
            f.write(f"   Action: Review process controls, raw material quality, environmental factors\n\n")
        
        # 2. TIME-BASED PATTERNS
        f.write("=" * 100 + "\n")
        f.write("2. TEMPORAL FAILURE PATTERNS\n")
        f.write("=" * 100 + "\n\n")
        
        qc['hour'] = qc['timestamp'].dt.hour
        qc['day_of_week'] = qc['timestamp'].dt.day_name()
        qc['date'] = qc['timestamp'].dt.date
        
        # Hourly pattern
        hourly = qc.groupby('hour').agg({
            'qc_id': 'count',
            'pass_fail': lambda x: (x == 'fail').sum()
        })
        hourly['fail_rate'] = (hourly['pass_fail'] / hourly['qc_id'] * 100).round(2)
        
        worst_hours = hourly.nlargest(3, 'fail_rate')
        f.write("Worst Performing Hours:\n")
        for hour, row in worst_hours.iterrows():
            f.write(f"  Hour {hour:02d}:00 - {row['fail_rate']:.1f}% fail rate ({row['pass_fail']} failures)\n")
        
        f.write("\n💡 INSIGHT: ")
        if worst_hours.index[0] in range(0, 6):
            f.write("Night shift (00:00-06:00) has highest failures. Review shift staffing/fatigue.\n")
        elif worst_hours.index[0] in range(6, 12):
            f.write("Morning shift has highest failures. Check equipment warm-up procedures.\n")
        else:
            f.write("Afternoon/evening shift has highest failures. Review end-of-shift procedures.\n")
        
        # Day of week pattern
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = qc.groupby('day_of_week').agg({
            'qc_id': 'count',
            'pass_fail': lambda x: (x == 'fail').sum()
        })
        daily['fail_rate'] = (daily['pass_fail'] / daily['qc_id'] * 100).round(2)
        daily = daily.reindex(day_order, fill_value=0)
        
        worst_day = daily['fail_rate'].idxmax()
        worst_day_rate = daily['fail_rate'].max()
        f.write(f"\nWorst Day: {worst_day} ({worst_day_rate:.1f}% fail rate)\n")
        f.write(f"💡 INSIGHT: Review {worst_day} operations - staffing, equipment maintenance schedule\n\n")
        
        # 3. BATCH-LEVEL ANALYSIS
        f.write("=" * 100 + "\n")
        f.write("3. BATCH-LEVEL FAILURE ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        batch_qc = qc.groupby('batch_id').agg({
            'qc_id': 'count',
            'pass_fail': lambda x: (x == 'fail').sum()
        })
        batch_qc['fail_rate'] = (batch_qc['pass_fail'] / batch_qc['qc_id'] * 100).round(1)
        
        # Categorize batches
        perfect_batches = (batch_qc['fail_rate'] == 0).sum()
        good_batches = ((batch_qc['fail_rate'] > 0) & (batch_qc['fail_rate'] <= 10)).sum()
        concerning_batches = ((batch_qc['fail_rate'] > 10) & (batch_qc['fail_rate'] <= 50)).sum()
        failed_batches = (batch_qc['fail_rate'] > 50).sum()
        
        total_batches = len(batch_qc)
        f.write(f"Batch Quality Distribution:\n")
        f.write(f"  🟢 Perfect (0% fail):        {perfect_batches:>5} batches ({perfect_batches/total_batches*100:.1f}%)\n")
        f.write(f"  🟢 Good (≤10% fail):         {good_batches:>5} batches ({good_batches/total_batches*100:.1f}%)\n")
        f.write(f"  🟡 Concerning (10-50% fail): {concerning_batches:>5} batches ({concerning_batches/total_batches*100:.1f}%)\n")
        f.write(f"  🔴 Failed (>50% fail):       {failed_batches:>5} batches ({failed_batches/total_batches*100:.1f}%)\n\n")
        
        f.write(f"⚠️ CRITICAL: {failed_batches} batches ({failed_batches/total_batches*100:.1f}%) failed majority of QC tests\n")
        f.write(f"   These batches should NOT have been dispatched - risk of customer complaints.\n\n")
        
        # Worst batches
        worst_batches = batch_qc.nlargest(10, 'fail_rate')
        f.write("Top 10 Worst Performing Batches:\n")
        f.write(f"{'Batch ID':<15} {'Tests':>7} {'Failures':>10} {'Fail Rate':>12}\n")
        f.write("-" * 50 + "\n")
        for batch_id, row in worst_batches.iterrows():
            f.write(f"{batch_id:<15} {row['qc_id']:>7} {row['pass_fail']:>10} {row['fail_rate']:>11.1f}%\n")
        
        f.write("\n💡 ACTION: Investigate these batches in production dataset for common patterns\n")
        f.write("   (operator, line, shift, raw materials, equipment used)\n\n")
        
        # 4. SKU-LEVEL ANALYSIS
        f.write("=" * 100 + "\n")
        f.write("4. SKU-LEVEL FAILURE PATTERNS\n")
        f.write("=" * 100 + "\n\n")
        
        sku_qc = qc.groupby('sku').agg({
            'qc_id': 'count',
            'pass_fail': lambda x: (x == 'fail').sum()
        })
        sku_qc['fail_rate'] = (sku_qc['pass_fail'] / sku_qc['qc_id'] * 100).round(2)
        sku_qc = sku_qc.sort_values('fail_rate', ascending=False)
        
        f.write("SKU Performance:\n")
        f.write(f"{'SKU':<30} {'Tests':>8} {'Failures':>10} {'Fail Rate':>12}\n")
        f.write("-" * 70 + "\n")
        for sku, row in sku_qc.iterrows():
            status = "🔴" if row['fail_rate'] > 40 else "🟡" if row['fail_rate'] > 30 else "🟢"
            f.write(f"{sku:<30} {row['qc_id']:>8} {row['pass_fail']:>10} {row['fail_rate']:>11.1f}% {status}\n")
        
        worst_sku = sku_qc.index[0]
        worst_sku_rate = sku_qc.iloc[0]['fail_rate']
        f.write(f"\n⚠️ WORST SKU: {worst_sku} ({worst_sku_rate:.1f}% fail rate)\n")
        f.write(f"   Action: Review {worst_sku} production process, recipe, specifications\n\n")
        
        # 5. PRODUCTION CORRELATION
        f.write("=" * 100 + "\n")
        f.write("5. CORRELATION WITH PRODUCTION DEFECTS\n")
        f.write("=" * 100 + "\n\n")
        
        # Merge QC with production on batch_id
        qc_prod = qc.merge(prod[['batch_id', 'quantity_produced', 'operator_id', 'line_id', 
                                  'stacked_before_robot', 'squashed', 'torn', 'undersized_small',
                                  'valleys', 'loose_packs', 'pale_underbaked']], 
                          on='batch_id', how='left')
        
        # Calculate total production defects
        defect_cols = ['stacked_before_robot', 'squashed', 'torn', 'undersized_small',
                       'valleys', 'loose_packs', 'pale_underbaked']
        qc_prod['total_defects'] = qc_prod[defect_cols].sum(axis=1)
        qc_prod['defect_rate'] = qc_prod['total_defects'] / qc_prod['quantity_produced'].replace(0, np.nan)
        
        # Compare QC pass vs fail
        passed_qc = qc_prod[qc_prod['pass_fail'] == 'pass']
        failed_qc = qc_prod[qc_prod['pass_fail'] == 'fail']
        
        avg_defect_passed = passed_qc['defect_rate'].mean() * 100
        avg_defect_failed = failed_qc['defect_rate'].mean() * 100
        
        f.write(f"Average Production Defect Rate:\n")
        f.write(f"  Batches that PASSED QC:  {avg_defect_passed:.2f}%\n")
        f.write(f"  Batches that FAILED QC:  {avg_defect_failed:.2f}%\n\n")
        
        if avg_defect_failed > avg_defect_passed * 1.5:
            f.write(f"✅ STRONG CORRELATION: QC failures correlate with production defects\n")
            f.write(f"   QC is correctly identifying problematic batches.\n")
            f.write(f"   Focus: Reduce production defects at source.\n\n")
        else:
            f.write(f"⚠️ WEAK CORRELATION: QC failures may not align with production defects\n")
            f.write(f"   Possible issues:\n")
            f.write(f"   1. QC spec limits too tight (failing good product)\n")
            f.write(f"   2. QC equipment needs calibration\n")
            f.write(f"   3. Different quality dimensions measured\n\n")
        
        # Operator analysis
        if 'operator_id' in qc_prod.columns:
            operator_qc = qc_prod.groupby('operator_id').agg({
                'qc_id': 'count',
                'pass_fail': lambda x: (x == 'fail').sum()
            })
            operator_qc['fail_rate'] = (operator_qc['pass_fail'] / operator_qc['qc_id'] * 100).round(2)
            operator_qc = operator_qc[operator_qc['qc_id'] >= 10]  # Min 10 tests
            
            if len(operator_qc) > 0:
                worst_operators = operator_qc.nlargest(5, 'fail_rate')
                f.write("Top 5 Operators with Highest QC Fail Rates (min 10 tests):\n")
                f.write(f"{'Operator':<15} {'Tests':>8} {'Failures':>10} {'Fail Rate':>12}\n")
                f.write("-" * 50 + "\n")
                for op, row in worst_operators.iterrows():
                    f.write(f"{op:<15} {row['qc_id']:>8} {row['pass_fail']:>10} {row['fail_rate']:>11.1f}%\n")
                f.write("\n💡 ACTION: Provide additional training or review workload for these operators\n\n")
        
        # 6. WASTE CORRELATION
        f.write("=" * 100 + "\n")
        f.write("6. QC FAILURE → WASTE CORRELATION\n")
        f.write("=" * 100 + "\n\n")
        
        # Find QC-failed batches that resulted in waste
        failed_batches = qc[qc['pass_fail'] == 'fail']['batch_id'].unique()
        waste_batches = waste['batch_id'].dropna().unique()
        
        failed_to_waste = len(set(failed_batches) & set(waste_batches))
        f.write(f"QC-Failed Batches: {len(failed_batches):,}\n")
        f.write(f"QC-Failed → Wasted: {failed_to_waste:,} ({failed_to_waste/len(failed_batches)*100:.1f}%)\n\n")
        
        if failed_to_waste / len(failed_batches) < 0.5:
            f.write(f"⚠️ CONCERN: Only {failed_to_waste/len(failed_batches)*100:.1f}% of QC-failed batches resulted in waste.\n")
            f.write(f"   This suggests many failed batches were still dispatched to customers!\n")
            f.write(f"   Action: Implement HARD STOP for batches failing QC - do not dispatch.\n\n")
        else:
            f.write(f"✅ GOOD: {failed_to_waste/len(failed_batches)*100:.1f}% of QC-failed batches properly discarded as waste.\n\n")
        
        # 7. ACTION PLAN
        f.write("=" * 100 + "\n")
        f.write("7. COMPREHENSIVE ACTION PLAN TO ACHIEVE <10% FAIL RATE\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SHORT-TERM ACTIONS (Week 1-4):\n")
        f.write("-" * 100 + "\n")
        f.write(f"1. **IMMEDIATE: Calibrate {worst_param} equipment**\n")
        f.write(f"   - Current fail rate: {worst_rate:.1f}%\n")
        f.write(f"   - Target: Reduce to <20% in 2 weeks\n")
        f.write(f"   - Actions: Equipment calibration, measurement SOP review, spec limit validation\n\n")
        
        f.write(f"2. **Investigate worst {len(worst_batches)} batches**\n")
        f.write(f"   - Review production logs for common patterns\n")
        f.write(f"   - Identify: operator, line, shift, raw material lot, equipment used\n")
        f.write(f"   - Document findings and corrective actions\n\n")
        
        f.write(f"3. **Implement HARD STOP for QC failures**\n")
        f.write(f"   - DO NOT dispatch batches failing >50% of QC tests\n")
        f.write(f"   - Quarantine failed batches for investigation\n")
        f.write(f"   - Target: Prevent customer complaints from known bad batches\n\n")
        
        f.write(f"4. **Shift-specific interventions**\n")
        f.write(f"   - Worst shift: Review identified in time analysis above\n")
        f.write(f"   - Actions: Staffing review, fatigue management, equipment checks\n\n")
        
        f.write("\nMID-TERM ACTIONS (Month 2-3):\n")
        f.write("-" * 100 + "\n")
        f.write(f"5. **Process improvements for top 3 failing parameters**\n")
        for i, (param, row) in enumerate(param_analysis.head(3).iterrows(), 1):
            f.write(f"   {i}. {param}: Review process controls, raw materials, environmental factors\n")
        f.write("\n")
        
        f.write(f"6. **Operator training program**\n")
        f.write(f"   - Focus on worst-performing operators identified above\n")
        f.write(f"   - Standard operating procedures review\n")
        f.write(f"   - Quality awareness training\n\n")
        
        f.write(f"7. **SKU-specific interventions**\n")
        f.write(f"   - Focus on {worst_sku} (worst SKU)\n")
        f.write(f"   - Review recipe, process, specifications\n")
        f.write(f"   - Consider pilot runs with improved process\n\n")
        
        f.write("\nLONG-TERM ACTIONS (Month 4-6):\n")
        f.write("-" * 100 + "\n")
        f.write(f"8. **Statistical Process Control (SPC)**\n")
        f.write(f"   - Implement control charts for all QC parameters\n")
        f.write(f"   - Real-time monitoring and alerts\n")
        f.write(f"   - Target: Detect issues before batches complete\n\n")
        
        f.write(f"9. **Predictive quality model**\n")
        f.write(f"   - Use production data to predict QC failures\n")
        f.write(f"   - Early intervention before QC stage\n")
        f.write(f"   - Target: Prevent failures at source\n\n")
        
        f.write(f"10. **Continuous improvement program**\n")
        f.write(f"    - Monthly QC fail rate tracking\n")
        f.write(f"    - Root cause analysis for all batches >50% fail\n")
        f.write(f"    - Quarterly spec limit review\n\n")
        
        # 8. SUCCESS METRICS
        f.write("=" * 100 + "\n")
        f.write("8. SUCCESS METRICS & TARGETS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Target Milestones:\n")
        f.write(f"  Month 1: Reduce fail rate to <30% (from current {fail_rate:.1f}%)\n")
        f.write(f"  Month 2: Reduce fail rate to <20%\n")
        f.write(f"  Month 3: Reduce fail rate to <15%\n")
        f.write(f"  Month 6: Achieve target <10%\n\n")
        
        f.write("Weekly Monitoring:\n")
        f.write(f"  - Overall QC fail rate\n")
        f.write(f"  - Parameter-specific fail rates\n")
        f.write(f"  - Failed batch count\n")
        f.write(f"  - QC-failed batches dispatched (should be 0)\n\n")
        
        f.write("Cost-Benefit Estimate:\n")
        current_waste_from_qc_fail = failed_to_waste
        f.write(f"  Current waste from QC failures: {current_waste_from_qc_fail:,} batches\n")
        f.write(f"  Target waste reduction: {int(current_waste_from_qc_fail * 0.6):,} batches (60% reduction)\n")
        f.write(f"  Estimated cost savings: Significant reduction in waste + customer returns\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF ROOT CAUSE ANALYSIS\n")
        f.write("=" * 100 + "\n")
    
    logger.info(f"✅ Root cause analysis written to {report_path}")
    return param_analysis, worst_batches


def visualize_analysis(qc, param_analysis):
    """Create visualizations for QC failure analysis."""
    
    # 1. Parameter fail rate comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    param_plot = param_analysis.sort_values('fail_rate_%', ascending=True)
    colors = ['red' if x > 30 else 'orange' if x > 20 else 'yellow' for x in param_plot['fail_rate_%']]
    param_plot['fail_rate_%'].plot(kind='barh', ax=ax, color=colors)
    ax.axvline(x=10, color='green', linestyle='--', linewidth=2, label='Target (10%)')
    ax.set_xlabel('Fail Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('QC Fail Rate by Parameter - Target: <10%', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_root_cause_parameter_failrates.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_root_cause_parameter_failrates.png")
    
    # 2. Hourly pattern heatmap
    qc['hour'] = qc['timestamp'].dt.hour
    qc['day_name'] = qc['timestamp'].dt.day_name()
    qc['is_fail'] = (qc['pass_fail'] == 'fail').astype(int)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_heatmap = qc.pivot_table(values='is_fail', index='day_name', columns='hour', aggfunc='mean') * 100
    hourly_heatmap = hourly_heatmap.reindex(day_order)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(hourly_heatmap, annot=True, fmt='.1f', cmap='RdYlGn_r', center=38.3,
                vmin=0, vmax=60, ax=ax, cbar_kws={'label': 'Fail Rate (%)'})
    ax.set_title('QC Fail Rate Heatmap: Day of Week vs Hour', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_root_cause_temporal_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_root_cause_temporal_heatmap.png")
    
    # 3. Batch quality distribution
    batch_qc = qc.groupby('batch_id').agg({
        'qc_id': 'count',
        'pass_fail': lambda x: (x == 'fail').sum()
    })
    batch_qc['fail_rate'] = (batch_qc['pass_fail'] / batch_qc['qc_id'] * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(batch_qc['fail_rate'], bins=50, color='darkred', alpha=0.7, edgecolor='black')
    ax.axvline(x=10, color='green', linestyle='--', linewidth=2, label='Target (<10%)')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Critical (>50%)')
    ax.set_xlabel('Batch QC Fail Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Batches', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Batch-Level QC Fail Rates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_root_cause_batch_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_root_cause_batch_distribution.png")
    
    # 4. Fail rate trend over time
    qc['date'] = qc['timestamp'].dt.date
    daily_fails = qc.groupby('date').agg({
        'qc_id': 'count',
        'is_fail': 'sum'
    })
    daily_fails['fail_rate'] = (daily_fails['is_fail'] / daily_fails['qc_id'] * 100)
    daily_fails['rolling_7d'] = daily_fails['fail_rate'].rolling(7, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_fails.index, daily_fails['fail_rate'], alpha=0.3, color='gray', label='Daily')
    ax.plot(daily_fails.index, daily_fails['rolling_7d'], color='red', linewidth=2, label='7-day Average')
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (10%)')
    ax.axhline(y=38.3, color='orange', linestyle='--', linewidth=2, label='Current Average')
    ax.fill_between(daily_fails.index, 0, 10, alpha=0.2, color='green', label='Target Zone')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fail Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('QC Fail Rate Trend Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_root_cause_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved qc_root_cause_trend.png")


def main():
    """Run comprehensive QC root cause analysis."""
    logger.info("=" * 80)
    logger.info("QC ROOT CAUSE ANALYSIS - Addressing 38.3% Fail Rate")
    logger.info("=" * 80)
    
    qc, prod, waste = load_data()
    logger.info(f"Loaded {len(qc):,} QC tests, {len(prod):,} production batches, {len(waste):,} waste records")
    
    param_analysis, worst_batches = root_cause_analysis(qc, prod, waste)
    visualize_analysis(qc, param_analysis)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ QC ROOT CAUSE ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📄 Report: reports/QC_ROOT_CAUSE_ANALYSIS.txt")
    logger.info(f"📊 Visualizations: reports/figures/qc_root_cause_*.png (4 files)")
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("   1. Review QC_ROOT_CAUSE_ANALYSIS.txt for detailed action plan")
    logger.info("   2. Share findings with operations team")
    logger.info("   3. Implement short-term actions (Week 1-4)")
    logger.info("   4. Monitor weekly fail rate progress toward <10% target")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
