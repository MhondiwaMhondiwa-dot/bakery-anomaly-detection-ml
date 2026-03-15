"""EDA for Inventory (Stock Movements) dataset.

Analyzes:
- Stock movements (in/out) by plant, SKU, store
- Balance trends over time
- Movement types and reasons

Outputs:
- reports/summaries/inventory_by_{plant,sku,movement_type}.csv
- reports/figures/inventory_*.png
- reports/inventory_summary.txt
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
    logger.info(f'Loading {path}')
    df = pd.read_parquet(path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.day_name()
    
    return df


def summary_stats(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    summary.append(f"Inventory Dataset Summary")
    summary.append(f"="*60)
    summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary.append(f"\nColumns: {', '.join(df.columns.tolist())}")
    summary.append(f"\nData types:\n{df.dtypes.value_counts()}")
    summary.append(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if 'qty_in' in df.columns:
        summary.append(f"\nQuantity In Stats:")
        summary.append(f"  Total: {df['qty_in'].sum():,.0f}")
        summary.append(f"  Mean: {df['qty_in'].mean():.2f}")
    
    if 'qty_out' in df.columns:
        summary.append(f"\nQuantity Out Stats:")
        summary.append(f"  Total: {df['qty_out'].sum():,.0f}")
        summary.append(f"  Mean: {df['qty_out'].mean():.2f}")
    
    if 'movement_type' in df.columns:
        summary.append(f"\nMovement Types:")
        for mt, count in df['movement_type'].value_counts().items():
            summary.append(f"  {mt}: {count:,}")
    
    text = '\n'.join(summary)
    (out_dir / 'inventory_summary.txt').write_text(text, encoding='utf-8')
    logger.info(f'Wrote summary to {out_dir / "inventory_summary.txt"}')


def grouped_summaries(df: pd.DataFrame, out_dir: Path):
    summaries_dir = out_dir / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    # by plant
    if 'plant_id' in df.columns:
        agg_cols = []
        if 'qty_in' in df.columns:
            agg_cols.append('qty_in')
        if 'qty_out' in df.columns:
            agg_cols.append('qty_out')
        if 'balance_after' in df.columns:
            agg_cols.append('balance_after')
        
        by_plant = df.groupby('plant_id')[agg_cols].agg(['sum', 'mean']).reset_index()
        by_plant.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in by_plant.columns]
        by_plant.to_csv(summaries_dir / 'inventory_by_plant.csv', index=False)
        logger.info('Wrote inventory_by_plant.csv')
    
    # by SKU
    if 'sku' in df.columns:
        agg_cols = []
        if 'qty_in' in df.columns:
            agg_cols.append('qty_in')
        if 'qty_out' in df.columns:
            agg_cols.append('qty_out')
        
        by_sku = df.groupby('sku')[agg_cols].sum().reset_index()
        by_sku = by_sku.sort_values(by_sku.columns[1], ascending=False).head(50)
        by_sku.to_csv(summaries_dir / 'inventory_by_sku.csv', index=False)
        logger.info('Wrote inventory_by_sku.csv')
    
    # by movement type
    if 'movement_type' in df.columns:
        agg_cols = []
        if 'qty_in' in df.columns:
            agg_cols.append('qty_in')
        if 'qty_out' in df.columns:
            agg_cols.append('qty_out')
        
        by_movement = df.groupby('movement_type')[agg_cols].sum().reset_index()
        by_movement.to_csv(summaries_dir / 'inventory_by_movement_type.csv', index=False)
        logger.info('Wrote inventory_by_movement_type.csv')


def visualizations(df: pd.DataFrame, out_dir: Path):
    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Balance histogram
    if 'balance_after' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df['balance_after'].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Balance After Movement')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Inventory Balance')
        plt.tight_layout()
        plt.savefig(figs_dir / 'inventory_balance_hist.png', dpi=150)
        plt.close()
        logger.info('Saved inventory_balance_hist.png')
    
    # 2. In/Out by plant
    if 'plant_id' in df.columns and 'qty_in' in df.columns and 'qty_out' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        plant_data = df.groupby('plant_id')[['qty_in', 'qty_out']].sum()
        plant_data.plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_xlabel('Plant ID')
        ax.set_ylabel('Total Quantity')
        ax.set_title('Inventory In/Out by Plant')
        ax.legend(['Qty In', 'Qty Out'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figs_dir / 'inventory_by_plant_bar.png', dpi=150)
        plt.close()
        logger.info('Saved inventory_by_plant_bar.png')
    
    # 3. Balance timeseries
    if 'timestamp' in df.columns and 'balance_after' in df.columns:
        df_ts = df[df['timestamp'].notna()].copy()
        if len(df_ts) > 0:
            daily = df_ts.groupby('date')['balance_after'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(daily['date'], daily['balance_after'])
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Balance')
            ax.set_title('Inventory Balance Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(figs_dir / 'inventory_timeseries.png', dpi=150)
            plt.close()
            logger.info('Saved inventory_timeseries.png')
    
    # 4. Movement types
    if 'movement_type' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df['movement_type'].value_counts().plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel('Movement Type')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Movement Types')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figs_dir / 'inventory_movement_types.png', dpi=150)
        plt.close()
        logger.info('Saved inventory_movement_types.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/inventory_stock_movements_dataset.parquet')
    parser.add_argument('--out_dir', default='reports')
    args = parser.parse_args()
    
    p = Path(args.input)
    if not p.exists():
        logger.error(f'Input not found: {p}')
        return
    
    df = load_and_prepare(p)
    out = Path(args.out_dir)
    
    summary_stats(df, out)
    grouped_summaries(df, out)
    visualizations(df, out)
    
    logger.info('Inventory EDA complete')


if __name__ == '__main__':
    main()
