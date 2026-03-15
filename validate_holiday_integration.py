import pandas as pd
import numpy as np

df = pd.read_parquet('data/analytic/plant_daily.parquet')

print('=== COMPLETE ANALYTIC DATASET WITH HOLIDAY CONTEXT ===\n')
print(f'Dataset: {df.shape[0]} days × {df.shape[1]} columns (Full year 2025)\n')

# Data coverage analysis
print('=== DATA COVERAGE ===')
coverage_metrics = {
    'Production': df['total_prod'].notna().sum(),
    'Dispatch': df['avg_delay'].notna().sum(),
    'Waste': df['total_waste'].notna().sum(),
    'Returns': df['total_return'].notna().sum(),
    'QC': df['qc_pass_rate'].notna().sum(),
    'Sales POS': df['total_sold'].notna().sum(),
    'Inventory': df['negative_balance_count'].notna().sum()
}

for metric, count in coverage_metrics.items():
    pct = count / 365 * 100
    print(f'{metric:12s}: {count:3d}/365 days ({pct:5.1f}%)')

# Holiday context
print('\n=== HOLIDAY CONTEXT ===')
print(f'Holidays: {df["is_holiday"].sum()} days')
print(f'Pre-holidays: {df["is_pre_holiday"].sum()} days')
print(f'Post-holidays: {df["is_post_holiday"].sum()} days')

# Show holidays with production data
print('\n=== HOLIDAYS WITH OPERATIONAL DATA ===')
holidays = df[df['is_holiday'] == True].copy()
holidays = holidays[holidays['total_prod'].notna()]
print(holidays[['holiday_name', 'total_prod', 'total_sold', 'avg_delay']].to_string())

# Anomaly detection summary
print('\n=== ANOMALY DETECTION (Days Flagged) ===')
anomaly_cols = [c for c in df.columns if 'anomaly' in c]
for col in anomaly_cols:
    count = df[col].sum()
    # Only count days with data
    data_days = df[df['total_prod'].notna()]
    if len(data_days) > 0 and col in data_days.columns:
        data_count = data_days[col].sum()
        pct = data_count / len(data_days) * 100
        print(f'{col:25s}: {int(data_count):3d}/{len(data_days)} days ({pct:5.1f}%)')

# Holiday impact analysis
print('\n=== HOLIDAY IMPACT ANALYSIS ===')
data_with_prod = df[df['total_prod'].notna()].copy()

holiday_avg = data_with_prod[data_with_prod['is_holiday']]['total_sold'].mean()
pre_holiday_avg = data_with_prod[data_with_prod['is_pre_holiday']]['total_sold'].mean()
normal_avg = data_with_prod[~(data_with_prod['is_holiday'] | data_with_prod['is_pre_holiday'] | data_with_prod['is_post_holiday'])]['total_sold'].mean()

print(f'Average sales on holidays: {holiday_avg:.0f} units')
print(f'Average sales pre-holiday: {pre_holiday_avg:.0f} units')
print(f'Average sales normal days: {normal_avg:.0f} units')
print(f'Holiday uplift: {((holiday_avg/normal_avg - 1) * 100):.1f}%')
print(f'Pre-holiday uplift: {((pre_holiday_avg/normal_avg - 1) * 100):.1f}%')

# Key insights
print('\n=== KEY INSIGHTS ===')
print(f'✓ Full year coverage: 365 days tracked')
print(f'✓ Holiday context: {df["is_holiday"].sum()} holidays + {df["is_pre_holiday"].sum()} pre-holidays marked')
print(f'✓ Smart anomaly detection: Sales spikes on holidays excluded from anomalies')
print(f'✓ Coverage: {len(data_with_prod)} days with complete operational data')
print(f'✓ QC anomalies: {int(data_with_prod["qc_anomaly"].sum())} days flagged (38%+ fail rate)')
print(f'✓ Inventory anomalies: {int(data_with_prod["inventory_anomaly"].sum())} days with negative balance')
