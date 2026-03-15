"""EDA for Waste dataset.

Analyzes:
- Waste quantities by plant, location, SKU, reason
- Temporal patterns (peak waste times)
- Cost implications if available
- Correlation with other factors

Outputs:
- reports/summaries/waste_by_{plant,sku,reason,location}.csv
- reports/figures/waste_*.png
- reports/waste_summary.txt
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


def load_and_prepare(path: Path) -> pd.DataFrame:
    logger.info(f'Loading waste data')
    # Try parquet first, fallback to cleaned CSV (written when parquet had duplicate columns)
    try:
        if path.exists() and path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.with_suffix('.cleaned.csv').exists():
            df = pd.read_csv(path.with_suffix('.cleaned.csv'))
        elif path.with_suffix('.csv').exists():
            df = pd.read_csv(path.with_suffix('.csv'))
        else:
            raise FileNotFoundError(f"No waste data file found")
    except Exception as e:
        logger.error(f'Error loading {path}: {e}')
        # Try CSV fallback
        csv_path = path.with_suffix('.cleaned.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise
    for col in ['timestamp', 'waste_date', 'expiry_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean up duplicate qty columns and identify the correct quantity column
    qty_cols = [c for c in df.columns if 'qty' in c.lower() or 'waste' in c.lower() and 'quantity' not in c.lower()]
    
    # Use qty_waste.1 if it's numeric, else qty_waste (the ID column will be dropped)
    numeric_qty_col = None
    for col in ['qty_waste.1', 'quantity_wasted', 'qty_waste', 'soft_white']:
        if col in df.columns:
            try:
                # Try to convert to numeric
                temp = pd.to_numeric(df[col], errors='coerce')
                if temp.notna().sum() > len(df) * 0.5:  # At least 50% non-null
                    numeric_qty_col = col
                    break
            except:
                pass
    
    # If no good numeric column found, sum multi-SKU columns
    if numeric_qty_col is None:
        sku_cols = ['soft_white', 'high_energy_brown', 'whole_grain_loaf', 'low_gi_seed_loaf']
        available_sku = [c for c in sku_cols if c in df.columns]
        if available_sku:
            df['qty_waste'] = df[available_sku].sum(axis=1)
            numeric_qty_col = 'qty_waste'
    
    # Store for use in later functions
    df._qty_col = numeric_qty_col
    
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    
    # Drop duplicate/problematic columns
    cols_to_drop = ['qty_waste.1', 'qty_waste.2']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df


def summary_stats(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    qty_col = getattr(df, '_qty_col', 'qty_waste')
    
    summary = []
    summary.append(f"Waste Dataset Summary")
    summary.append(f"="*60)
    summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary.append(f"\nColumns: {', '.join(df.columns.tolist())}")
    summary.append(f"\nData types:\n{df.dtypes.value_counts()}")
    summary.append(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if qty_col and qty_col in df.columns:
        summary.append(f"\nWaste Quantity Stats ({qty_col}):")
        total_waste = pd.to_numeric(df[qty_col], errors='coerce').sum()
        summary.append(f"  Total waste: {total_waste:,.0f} units")
        summary.append(f"  Mean: {pd.to_numeric(df[qty_col], errors='coerce').mean():.2f}")
        summary.append(f"  Median: {pd.to_numeric(df[qty_col], errors='coerce').median():.2f}")
        summary.append(f"  Max: {pd.to_numeric(df[qty_col], errors='coerce').max():.0f}")
    
    text = '\n'.join(summary)
    (out_dir / 'waste_summary.txt').write_text(text, encoding='utf-8')
    logger.info(f'Wrote summary to {out_dir / "waste_summary.txt"}')
    return text


def grouped_summaries(df: pd.DataFrame, out_dir: Path):
    summaries_dir = out_dir / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    qty_col = getattr(df, '_qty_col', 'qty_waste')
    
    # Ensure qty_col is numeric
    if qty_col and qty_col in df.columns:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
    
    if 'plant_id' in df.columns and qty_col and qty_col in df.columns:
        by_plant = df.groupby('plant_id')[qty_col].agg(['count', 'sum', 'mean']).reset_index()
        by_plant.columns = ['plant_id', 'waste_count', 'total_waste_qty', 'mean_qty']
        by_plant = by_plant.sort_values('total_waste_qty', ascending=False)
        by_plant.to_csv(summaries_dir / 'waste_by_plant.csv', index=False)
        logger.info('Wrote waste_by_plant.csv')
    
    # Handle SKU columns (either single 'sku' or multi-SKU columns)
    if 'sku' in df.columns and qty_col and qty_col in df.columns:
        by_sku = df.groupby('sku')[qty_col].agg(['count', 'sum', 'mean']).reset_index()
        by_sku.columns = ['sku_code', 'waste_count', 'total_waste_qty', 'mean_qty']
        by_sku = by_sku.sort_values('total_waste_qty', ascending=False)
        by_sku.to_csv(summaries_dir / 'waste_by_sku.csv', index=False)
        logger.info('Wrote waste_by_sku.csv')
    else:
        sku_cols = [c for c in ['soft_white','high_energy_brown','whole_grain_loaf','low_gi_seed_loaf'] if c in df.columns]
        if sku_cols:
            totals = df[sku_cols].sum().reset_index()
            totals.columns = ['sku','total_waste_qty']
            totals.to_csv(summaries_dir / 'waste_by_sku.csv', index=False)
            logger.info('Wrote waste_by_sku.csv (multi-SKU columns)')
    
    if 'handling_condition' in df.columns and qty_col and qty_col in df.columns:
        by_reason = df.groupby('handling_condition')[qty_col].agg(['count', 'sum', 'mean']).reset_index()
        by_reason.columns = ['waste_reason', 'waste_count', 'total_waste_qty', 'mean_qty']
        by_reason = by_reason.sort_values('total_waste_qty', ascending=False)
        by_reason.to_csv(summaries_dir / 'waste_by_reason.csv', index=False)
        logger.info('Wrote waste_by_reason.csv')
    
    if 'depot_id' in df.columns and qty_col and qty_col in df.columns:
        by_loc = df.groupby('depot_id')[qty_col].agg(['count', 'sum', 'mean']).reset_index()
        by_loc.columns = ['location', 'waste_count', 'total_waste_qty', 'mean_qty']
        by_loc = by_loc.sort_values('total_waste_qty', ascending=False)
        by_loc.to_csv(summaries_dir / 'waste_by_location.csv', index=False)
        logger.info('Wrote waste_by_location.csv')


def visualizations(df: pd.DataFrame, out_dir: Path):
    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    qty_col = getattr(df, '_qty_col', 'qty_waste')
    
    if qty_col and qty_col in df.columns:
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 5))
        df[qty_col].dropna().hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Waste Quantity')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Waste Quantities')
        plt.tight_layout()
        plt.savefig(figs_dir / 'waste_qty_hist.png', dpi=150)
        plt.close()
        logger.info('Saved waste_qty_hist.png')
    
    if 'waste_reason' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_reasons = df['waste_reason'].value_counts().head(15)
        top_reasons.plot(kind='barh', ax=ax)
        ax.set_xlabel('Count')
        ax.set_title('Top 15 Waste Reasons')
        plt.tight_layout()
        plt.savefig(figs_dir / 'waste_by_reason_bar.png', dpi=150)
        plt.close()
        logger.info('Saved waste_by_reason_bar.png')
    
    if 'timestamp' in df.columns and qty_col:
        daily = df.groupby('date')[qty_col].sum().reset_index()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily['date'], daily[qty_col], marker='o', markersize=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Waste Quantity')
        ax.set_title('Waste Over Time (Daily)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figs_dir / 'waste_timeseries.png', dpi=150)
        plt.close()
        logger.info('Saved waste_timeseries.png')
    
    if 'location' in df.columns and qty_col:
        fig, ax = plt.subplots(figsize=(12, 6))
        by_loc = df.groupby('location')[qty_col].sum().sort_values(ascending=False).head(20)
        by_loc.plot(kind='barh', ax=ax, color='salmon')
        ax.set_xlabel('Total Waste Quantity')
        ax.set_title('Top 20 Locations by Waste')
        plt.tight_layout()
        plt.savefig(figs_dir / 'waste_by_location.png', dpi=150)
        plt.close()
        logger.info('Saved waste_by_location.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/waste_dataset.parquet')
    parser.add_argument('--out_dir', default='reports')
    args = parser.parse_args()
    
    p = Path(args.input)
    # Check for parquet, cleaned CSV, or CSV
    if not (p.exists() or p.with_suffix('.cleaned.csv').exists() or p.with_suffix('.csv').exists()):
        logger.error(f'Input not found: {p} (or .cleaned.csv / .csv variants)')
        return
    
    df = load_and_prepare(p)
    out = Path(args.out_dir)
    
    summary_stats(df, out)
    grouped_summaries(df, out)
    visualizations(df, out)
    
    logger.info('Waste EDA complete')


if __name__ == '__main__':
    main()
