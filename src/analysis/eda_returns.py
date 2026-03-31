"""EDA for Returns dataset.

Analyzes:
- Return frequency by route, retailer, SKU, reason
- Return quantities and patterns over time
- Correlation between returns and dispatch/delivery
- Peak return periods

Outputs:
- reports/summaries/returns_by_{route,retailer,sku,reason}.csv
- reports/figures/returns_*.png
- reports/returns_summary.txt
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
    # parse timestamps
    for col in ['timestamp', 'return_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # derive time features if timestamp present
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    
    return df


def summary_stats(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    summary.append(f"Returns Dataset Summary")
    summary.append(f"="*60)
    summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary.append(f"\nColumns: {', '.join(df.columns.tolist())}")
    summary.append(f"\nData types:\n{df.dtypes.value_counts()}")
    summary.append(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Compute total returned quantity from multi-SKU columns when present
    sku_cols = [c for c in ['soft_white','high_energy_brown','whole_grain_loaf','low_gi_seed_loaf'] if c in df.columns]
    if sku_cols:
        df['qty_returned'] = df[sku_cols].sum(axis=1)
    if 'qty_returned' in df.columns:
        summary.append(f"\nReturn Quantity Stats:")
        summary.append(f"  Total returned: {df['qty_returned'].sum():,.0f}")
        summary.append(f"  Mean: {df['qty_returned'].mean():.2f}")
        summary.append(f"  Median: {df['qty_returned'].median():.2f}")
        summary.append(f"  Max: {df['qty_returned'].max():.0f}")
    
    reason_col = 'return_reason' if 'return_reason' in df.columns else ('reason_code' if 'reason_code' in df.columns else None)
    if reason_col:
        summary.append(f"\nTop Return Reasons:")
        top_reasons = df[reason_col].value_counts().head(10)
        for reason, count in top_reasons.items():
            summary.append(f"  {reason}: {count:,} ({count/len(df)*100:.1f}%)")
    
    text = '\n'.join(summary)
    (out_dir / 'returns_summary.txt').write_text(text, encoding='utf-8')
    logger.info(f'Wrote summary to {out_dir / "returns_summary.txt"}')
    return text


def grouped_summaries(df: pd.DataFrame, out_dir: Path):
    summaries_dir = out_dir / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine qty column
    qty_col = 'qty_returned' if 'qty_returned' in df.columns else None
    
    # Determine reason column
    reason_col = 'return_reason' if 'return_reason' in df.columns else ('reason_code' if 'reason_code' in df.columns else None)
    
    # by route
    if 'route_id' in df.columns and qty_col:
        by_route = df.groupby('route_id')[qty_col].agg(['count', 'sum', 'mean', 'median']).reset_index()
        by_route.columns = ['route_id', 'return_count', 'total_qty_returned', 'mean_qty', 'median_qty']
        by_route = by_route.sort_values('total_qty_returned', ascending=False)
        by_route.to_csv(summaries_dir / 'returns_by_route.csv', index=False)
        logger.info(f'Wrote returns_by_route.csv')
    
    # by retailer
    if 'retailer_id' in df.columns and qty_col:
        retailer_series = df['retailer_id']
        if pd.api.types.is_string_dtype(retailer_series) or retailer_series.dtype == object:
            valid = retailer_series.notna() & (retailer_series.astype(str).str.strip() != '')
        else:
            valid = retailer_series.notna()

        retailer_df = df.loc[valid, ['retailer_id', qty_col]]
        if retailer_df.empty:
            logger.warning('Skipping returns_by_retailer.csv: retailer_id has no valid values')
        else:
            by_retailer = retailer_df.groupby('retailer_id')[qty_col].agg(['count', 'sum', 'mean']).reset_index()
            by_retailer.columns = ['retailer_id', 'return_count', 'total_qty_returned', 'mean_qty']
            by_retailer = by_retailer.sort_values('total_qty_returned', ascending=False)
            by_retailer.to_csv(summaries_dir / 'returns_by_retailer.csv', index=False)
            logger.info(f'Wrote returns_by_retailer.csv')
    
    # by SKU
    if 'sku' in df.columns and qty_col:
        by_sku = df.groupby('sku')[qty_col].agg(['count', 'sum', 'mean']).reset_index()
        by_sku.columns = ['sku', 'return_count', 'total_qty_returned', 'mean_qty']
        by_sku = by_sku.sort_values('total_qty_returned', ascending=False)
        by_sku.to_csv(summaries_dir / 'returns_by_sku.csv', index=False)
        logger.info(f'Wrote returns_by_sku.csv')
    else:
        sku_cols = [c for c in ['soft_white','high_energy_brown','whole_grain_loaf','low_gi_seed_loaf'] if c in df.columns]
        if sku_cols:
            totals = df[sku_cols].sum().reset_index()
            totals.columns = ['sku','total_qty_returned']
            totals.to_csv(summaries_dir / 'returns_by_sku.csv', index=False)
            logger.info('Wrote returns_by_sku.csv (multi-SKU columns)')
    
    # by reason
    if reason_col and qty_col:
        by_reason = df.groupby(reason_col)[qty_col].agg(['count', 'sum', 'mean']).reset_index()
        by_reason.columns = ['return_reason', 'return_count', 'total_qty_returned', 'mean_qty']
        by_reason = by_reason.sort_values('total_qty_returned', ascending=False)
        by_reason.to_csv(summaries_dir / 'returns_by_reason.csv', index=False)
        logger.info(f'Wrote returns_by_reason.csv')


def visualizations(df: pd.DataFrame, out_dir: Path):
    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    def add_caption(fig, text):
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    qty_col = 'qty_returned' if 'qty_returned' in df.columns else None

    def non_empty_group_sum(frame: pd.DataFrame, key_col: str, value_col: str) -> pd.Series:
        """Return grouped sum excluding null/blank keys; empty Series if no valid keys."""
        if key_col not in frame.columns or value_col not in frame.columns:
            return pd.Series(dtype=float)

        key_series = frame[key_col]
        # Handle both numeric and string-like identifiers robustly.
        if pd.api.types.is_string_dtype(key_series) or key_series.dtype == object:
            valid = key_series.notna() & (key_series.astype(str).str.strip() != '')
        else:
            valid = key_series.notna()

        filtered = frame.loc[valid, [key_col, value_col]]
        if filtered.empty:
            return pd.Series(dtype=float)

        return filtered.groupby(key_col)[value_col].sum()
    
    # 1. Return quantity distribution
    if qty_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[qty_col].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Quantity Returned')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Return Quantities')
        add_caption(fig, 'Shows typical return sizes and highlights large return events.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'returns_qty_hist.png', dpi=150)
        plt.close()
        logger.info('Saved returns_qty_hist.png')
    
    # 2. Returns by reason (bar chart)
    if 'return_reason' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_reasons = df['return_reason'].value_counts().head(15)
        top_reasons.plot(kind='barh', ax=ax)
        ax.set_xlabel('Count')
        ax.set_title('Top 15 Return Reasons')
        add_caption(fig, 'Ranks the most common return reasons by count.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'returns_by_reason_bar.png', dpi=150)
        plt.close()
        logger.info('Saved returns_by_reason_bar.png')
    
    # 3. Returns over time
    if 'timestamp' in df.columns and qty_col:
        daily = df.groupby('date')[qty_col].sum().reset_index()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily['date'], daily[qty_col], marker='o', markersize=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Quantity Returned')
        ax.set_title('Returns Over Time (Daily)')
        plt.xticks(rotation=45)
        add_caption(fig, 'Tracks return volume trends and spikes over time.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'returns_timeseries.png', dpi=150)
        plt.close()
        logger.info('Saved returns_timeseries.png')
    
    # 4. Returns by day of week
    if 'dayofweek' in df.columns and qty_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        by_day = df.groupby('dayofweek')[qty_col].sum().reindex(day_order)
        by_day.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Total Quantity Returned')
        ax.set_title('Returns by Day of Week')
        plt.xticks(rotation=45)
        add_caption(fig, 'Shows weekly patterns in returns for staffing and logistics planning.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'returns_by_dayofweek.png', dpi=150)
        plt.close()
        logger.info('Saved returns_by_dayofweek.png')

    # 5. Returns by hour (heatmap)
    if 'dayofweek' in df.columns and 'hour' in df.columns and qty_col:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heat = df.pivot_table(index='dayofweek', columns='hour', values=qty_col, aggfunc='sum').reindex(day_order)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heat, cmap='YlOrRd', ax=ax)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        ax.set_title('Return Volume Heatmap (Day vs Hour)')
        add_caption(fig, 'Pinpoints the busiest return windows by day and hour.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'returns_heatmap_day_hour.png', dpi=150)
        plt.close()
        logger.info('Saved returns_heatmap_day_hour.png')

    # 6. Returns by route (top 15)
    if 'route_id' in df.columns and qty_col:
        by_route = non_empty_group_sum(df, 'route_id', qty_col).sort_values(ascending=True).tail(15)
        if by_route.empty:
            logger.warning('Skipping returns_by_route_top15.png: route_id has no valid values')
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            by_route.plot(kind='barh', ax=ax, color='slateblue')
            ax.set_xlabel('Total Quantity Returned')
            ax.set_title('Top 15 Routes by Return Volume')
            add_caption(fig, 'Highlights routes with the highest return volumes for investigation.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'returns_by_route_top15.png', dpi=150)
            plt.close()
            logger.info('Saved returns_by_route_top15.png')

    # 7. Returns by retailer (top 15)
    if 'retailer_id' in df.columns and qty_col:
        by_retailer = non_empty_group_sum(df, 'retailer_id', qty_col).sort_values(ascending=True).tail(15)
        if by_retailer.empty:
            logger.warning('Skipping returns_by_retailer_top15.png: retailer_id has no valid values')
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            by_retailer.plot(kind='barh', ax=ax, color='teal')
            ax.set_xlabel('Total Quantity Returned')
            ax.set_title('Top 15 Retailers by Return Volume')
            add_caption(fig, 'Shows retailers with unusually high return volumes.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'returns_by_retailer_top15.png', dpi=150)
            plt.close()
            logger.info('Saved returns_by_retailer_top15.png')

    # 8. Returns by reason (quantity)
    reason_col = 'return_reason' if 'return_reason' in df.columns else ('reason_code' if 'reason_code' in df.columns else None)
    if reason_col and qty_col:
        by_reason_qty = non_empty_group_sum(df, reason_col, qty_col).sort_values(ascending=True).tail(15)
        if by_reason_qty.empty:
            logger.warning('Skipping returns_by_reason_qty.png: reason column has no valid values')
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            by_reason_qty.plot(kind='barh', ax=ax, color='darkorange')
            ax.set_xlabel('Total Quantity Returned')
            ax.set_title('Top Return Reasons by Quantity')
            add_caption(fig, 'Ranks reasons by total units returned, not just count of cases.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'returns_by_reason_qty.png', dpi=150)
            plt.close()
            logger.info('Saved returns_by_reason_qty.png')

    # 9. Return quantity by reason (boxplot)
    if reason_col and qty_col:
        top_reasons = df[reason_col].value_counts().head(8).index
        df_reason = df[df[reason_col].isin(top_reasons)]
        if not df_reason.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=reason_col, y=qty_col, data=df_reason, ax=ax)
            ax.set_xlabel('Return Reason')
            ax.set_ylabel('Quantity Returned')
            ax.set_title('Return Quantity Distribution by Reason (Top 8)')
            plt.xticks(rotation=30, ha='right')
            add_caption(fig, 'Compares return size by reason to identify severe issues.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'returns_qty_by_reason_box.png', dpi=150)
            plt.close()
            logger.info('Saved returns_qty_by_reason_box.png')

    # 10. Returns by SKU (quantity)
    if 'sku' in df.columns and qty_col:
        by_sku = non_empty_group_sum(df, 'sku', qty_col).sort_values(ascending=True).tail(15)
        if by_sku.empty:
            logger.warning('Skipping returns_by_sku_top15.png: sku has no valid values')
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            by_sku.plot(kind='barh', ax=ax, color='mediumseagreen')
            ax.set_xlabel('Total Quantity Returned')
            ax.set_title('Top 15 SKUs by Return Volume')
            add_caption(fig, 'Highlights SKUs with the greatest return impact.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'returns_by_sku_top15.png', dpi=150)
            plt.close()
            logger.info('Saved returns_by_sku_top15.png')

    # 11. Rolling 7-day return trend
    if 'timestamp' in df.columns and qty_col:
        daily = df.groupby('date')[qty_col].sum().sort_index()
        roll = daily.rolling(7, min_periods=3).mean()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily.index, daily.values, color='lightcoral', alpha=0.6, label='Daily')
        ax.plot(roll.index, roll.values, color='darkred', linewidth=2, label='7-day rolling')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Quantity Returned')
        ax.set_title('Rolling Return Trend (7-day)')
        ax.legend()
        add_caption(fig, 'Smooths daily noise to show sustained return patterns.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'returns_rolling_7d.png', dpi=150)
        plt.close()
        logger.info('Saved returns_rolling_7d.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/returns_dataset.parquet')
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
    
    logger.info('Returns EDA complete')


if __name__ == '__main__':
    main()
