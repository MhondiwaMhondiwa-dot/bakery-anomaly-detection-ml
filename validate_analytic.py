import pandas as pd

df = pd.read_parquet('data/analytic/plant_daily.parquet')

print('=== COMPLETE ANALYTIC DATASET ===\n')
print(f'Shape: {df.shape[0]} days × {df.shape[1]} columns\n')

feature_groups = {
    'Production': [c for c in df.columns if 'prod' in c or 'defect' in c],
    'Dispatch': [c for c in df.columns if 'delay' in c or 'late' in c or 'early' in c],
    'Waste': [c for c in df.columns if 'waste' in c],
    'Returns': [c for c in df.columns if 'return' in c],
    'QC': [c for c in df.columns if 'qc_' in c],
    'Sales POS': [c for c in df.columns if 'sold' in c or 'demand' in c or 'promotion' in c or 'retailer' in c],
    'Inventory': [c for c in df.columns if 'balance' in c or 'stock' in c or 'expiry' in c],
    'Anomalies': [c for c in df.columns if 'anomaly' in c]
}

for group, cols in feature_groups.items():
    if cols:
        coverage = df[cols].notna().any(axis=1).sum()
        print(f'{group}: {len(cols)} features, {coverage}/{len(df)} days coverage')

print('\n=== KEY INSIGHTS ===')
print(f'QC anomalies: {df["qc_anomaly"].sum()} days (38% fail rate threshold)')
print(f'Inventory anomalies: {df["inventory_anomaly"].sum()} days (negative balance detected)')
print(f'Return anomalies: {df["return_anomaly"].sum()} days (spike detection)')
print(f'Total negative balance events: {df["negative_balance_count"].sum():.0f}')
print(f'Items nearing expiry: {df["nearing_expiry_count"].sum():.0f}')
print(f'Average daily sales: {df["total_sold"].mean():.0f} units')
print(f'Promotion days: {df["promotion_days"].sum():.0f} total')
