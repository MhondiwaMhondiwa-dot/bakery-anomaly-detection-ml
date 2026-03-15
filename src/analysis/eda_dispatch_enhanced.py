"""Enhanced EDA for Dispatch Dataset (depot-based, multi-SKU loads).

Designed for the latest Shepperton-only dispatch extract with one-truck-per-day
structure and depot IDs (no legacy route IDs). Outputs depot- and vehicle-focused
summaries plus visuals for Streamlit.
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
plt.rcParams['figure.figsize'] = (12, 6)

SKU_COLS = [
    'soft_white',
    'high_energy_brown',
    'whole_grain_loaf',
    'low_gi_seed_loaf'
]


def load_and_prepare(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load dispatch parquet, derive delay metrics, and add time features."""
    logger.info(f'Loading {path}')
    df = pd.read_parquet(path)

    # Harmonize timestamps
    for col in ['departure_time', 'expected_arrival', 'actual_arrival']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'timestamp' not in df.columns and 'departure_time' in df.columns:
        df['timestamp'] = df['departure_time']

    # Compute delay + categories
    if {'expected_arrival', 'actual_arrival'} <= set(df.columns):
        df['dispatch_delay_minutes'] = (
            (df['actual_arrival'] - df['expected_arrival']).dt.total_seconds() / 60.0
        )
        df['delay_category'] = pd.cut(
            df['dispatch_delay_minutes'],
            bins=[-np.inf, -30, 0, 30, 60, np.inf],
            labels=['Very Early (>30min)', 'Early (<30min)', 'On-Time (±30min)', 'Late (30-60min)', 'Very Late (>60min)']
        )
        df['on_time'] = df['dispatch_delay_minutes'].between(-30, 30).astype(int)

    # Time parts
    if 'timestamp' in df.columns:
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.day_name()

    sku_cols = [c for c in SKU_COLS if c in df.columns]
    if 'total_quantity' in df.columns:
        df['total_qty'] = df['total_quantity']
    elif sku_cols:
        df['total_qty'] = df[sku_cols].sum(axis=1)
    else:
        df['total_qty'] = np.nan

    logger.info(f'Loaded {len(df):,} dispatch events')
    return df, sku_cols


def write_summary(df: pd.DataFrame, sku_cols: list[str], output_dir: Path) -> None:
    """Write depot-based dispatch summary text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'dispatch_summary.txt'

    lines: list[str] = []
    lines.append('=' * 70)
    lines.append('DISPATCH DATASET SUMMARY')
    lines.append('=' * 70)
    lines.append(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if 'timestamp' in df.columns:
        lines.append(f"Date Range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # Volume overview
    lines.append('\nDISPATCH VOLUME OVERVIEW')
    lines.append('-' * 70)
    total_qty = df['total_qty'].sum() if 'total_qty' in df.columns else np.nan
    lines.append(f"Total Dispatch Events: {len(df):,}")
    if not np.isnan(total_qty):
        lines.append(f"Total Units Dispatched: {total_qty:,.0f}")
        lines.append(f"Mean Units per Dispatch: {df['total_qty'].mean():.1f}")

    # Plant distribution
    if 'plant_id' in df.columns:
        lines.append(f"Plants Dispatching: {df['plant_id'].nunique()}")
        for plant, count in df['plant_id'].value_counts().items():
            lines.append(f"  - {plant}: {count:,} dispatches ({count/len(df)*100:.1f}%)")

    # Depot coverage
    if 'depot_id' in df.columns:
        lines.append(f"Depots Served: {df['depot_id'].nunique()}")
        top_depots = df['depot_id'].value_counts().head(5)
        lines.append('Top Depots by Dispatch Count:')
        for depot, count in top_depots.items():
            lines.append(f"  - {depot}: {count:,} dispatches ({count/len(df)*100:.1f}%)")

    if 'vehicle_id' in df.columns:
        lines.append(f"Vehicles Used: {df['vehicle_id'].nunique()}")

    # SKU totals (multi-column)
    if sku_cols:
        lines.append('\nTop SKUs by Quantity:')
        sku_totals = df[sku_cols].sum().sort_values(ascending=False)
        for sku, qty in sku_totals.items():
            lines.append(f"  - {sku}: {qty:,.0f} units")

    # On-time performance
    lines.append('\nON-TIME DELIVERY PERFORMANCE')
    lines.append('-' * 70)
    if 'dispatch_delay_minutes' in df.columns:
        mean_delay = df['dispatch_delay_minutes'].mean()
        median_delay = df['dispatch_delay_minutes'].median()
        p95_delay = df['dispatch_delay_minutes'].quantile(0.95)
        lines.append(f"Mean Delay: {mean_delay:.1f} min")
        lines.append(f"Median Delay: {median_delay:.1f} min")
        lines.append(f"95th Percentile Delay: {p95_delay:.1f} min")

    if 'on_time' in df.columns:
        on_time_pct = df['on_time'].mean() * 100
        lines.append(f"On-Time Rate (±30 min): {on_time_pct:.2f}%")
        if on_time_pct < 80:
            lines.append('⚠️  Critical: below 80% target')
        elif on_time_pct < 90:
            lines.append('⚠️  Warning: below 90% target')
        else:
            lines.append('✅ Good: meets 90% target')

    # Time patterns
    lines.append('\nTIME PATTERNS')
    lines.append('-' * 70)
    if {'hour', 'dispatch_delay_minutes'} <= set(df.columns):
        hourly = df.groupby('hour')['dispatch_delay_minutes'].mean()
        lines.append('Mean Delay by Hour:')
        for hour, delay in hourly.items():
            lines.append(f"  - {hour:02d}: {delay:.1f} min")
    if {'dayofweek', 'dispatch_delay_minutes'} <= set(df.columns):
        daily = df.groupby('dayofweek')['dispatch_delay_minutes'].mean()
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        for day in order:
            if day in daily.index:
                lines.append(f"  - {day}: {daily.loc[day]:.1f} min")

    lines.append('\n' + '=' * 70)

    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'Wrote {summary_path}')


def grouped_summaries(df: pd.DataFrame, sku_cols: list[str], summaries_dir: Path) -> None:
    """Generate depot-focused summary CSVs."""
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if 'plant_id' in df.columns:
        by_plant = df.groupby('plant_id').agg(
            total_dispatches=('dispatch_id', 'count'),
            total_qty=('total_qty', 'sum'),
            mean_qty_per_dispatch=('total_qty', 'mean'),
            mean_delay_min=('dispatch_delay_minutes', 'mean'),
            median_delay_min=('dispatch_delay_minutes', 'median'),
            on_time_rate=('on_time', 'mean'),
            unique_depots=('depot_id', 'nunique'),
            unique_vehicles=('vehicle_id', 'nunique')
        ).round(2)
        by_plant['on_time_pct'] = (by_plant['on_time_rate'] * 100).round(2)
        by_plant.to_csv(summaries_dir / 'dispatch_by_plant.csv')
        logger.info('Wrote dispatch_by_plant.csv')

    if 'depot_id' in df.columns:
        by_depot = df.groupby('depot_id').agg(
            total_dispatches=('dispatch_id', 'count'),
            total_qty=('total_qty', 'sum'),
            mean_qty_per_dispatch=('total_qty', 'mean'),
            mean_delay_min=('dispatch_delay_minutes', 'mean'),
            median_delay_min=('dispatch_delay_minutes', 'median'),
            on_time_rate=('on_time', 'mean'),
            unique_plants=('plant_id', 'nunique'),
            unique_vehicles=('vehicle_id', 'nunique')
        ).round(2)
        by_depot['on_time_pct'] = (by_depot['on_time_rate'] * 100).round(2)
        by_depot = by_depot.sort_values('mean_delay_min', ascending=False)
        by_depot.to_csv(summaries_dir / 'dispatch_by_depot.csv')
        logger.info('Wrote dispatch_by_depot.csv')

    if 'vehicle_id' in df.columns:
        by_vehicle = df.groupby('vehicle_id').agg(
            total_trips=('dispatch_id', 'count'),
            total_qty=('total_qty', 'sum'),
            mean_delay_min=('dispatch_delay_minutes', 'mean'),
            median_delay_min=('dispatch_delay_minutes', 'median'),
            on_time_rate=('on_time', 'mean'),
            unique_depots=('depot_id', 'nunique')
        ).round(2)
        by_vehicle['on_time_pct'] = (by_vehicle['on_time_rate'] * 100).round(2)
        by_vehicle = by_vehicle.sort_values('mean_delay_min', ascending=False)
        by_vehicle.to_csv(summaries_dir / 'dispatch_by_vehicle.csv')
        logger.info('Wrote dispatch_by_vehicle.csv')

    if sku_cols:
        sku_df = pd.DataFrame({
            'total_qty': df[sku_cols].sum(),
            'dispatch_count': (df[sku_cols] > 0).sum()
        })
        sku_df['mean_qty_per_dispatch'] = sku_df['total_qty'] / sku_df['dispatch_count'].replace(0, np.nan)
        sku_df = sku_df.sort_values('total_qty', ascending=False).round(2)
        sku_df.to_csv(summaries_dir / 'dispatch_by_sku.csv')
        logger.info('Wrote dispatch_by_sku.csv')

    if 'hour' in df.columns:
        by_hour = df.groupby('hour').agg(
            dispatch_count=('dispatch_id', 'count'),
            mean_delay_min=('dispatch_delay_minutes', 'mean'),
            on_time_rate=('on_time', 'mean')
        ).round(2)
        by_hour['on_time_pct'] = (by_hour['on_time_rate'] * 100).round(2)
        by_hour.to_csv(summaries_dir / 'dispatch_by_hour.csv')
        logger.info('Wrote dispatch_by_hour.csv')


def visualizations(df: pd.DataFrame, sku_cols: list[str], figures_dir: Path) -> None:
    """Generate depot-first visualizations for Streamlit."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    if 'dispatch_delay_minutes' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        delays = df['dispatch_delay_minutes'].dropna()
        ax.hist(delays, bins=80, color='steelblue', alpha=0.75, edgecolor='black')
        ax.axvline(delays.mean(), color='red', linestyle='--', linewidth=2, label=f"Mean {delays.mean():.1f}m")
        ax.axvline(delays.median(), color='green', linestyle='--', linewidth=2, label=f"Median {delays.median():.1f}m")
        ax.axvline(0, color='blue', linestyle='-', linewidth=2, label='On-Time (0m)')
        ax.set_xlabel('Dispatch Delay (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Dispatch Delays', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_delay_hist.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_delay_hist.png')

    if {'depot_id', 'dispatch_delay_minutes'} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(14, 7))
        top_depots = df['depot_id'].value_counts().nlargest(20).index
        df_top = df[df['depot_id'].isin(top_depots)]
        order = df_top.groupby('depot_id')['dispatch_delay_minutes'].median().sort_values().index
        sns.boxplot(data=df_top, x='depot_id', y='dispatch_delay_minutes', order=order, palette='Set2', ax=ax)
        ax.axhline(0, color='blue', linestyle='--', linewidth=2, label='On-Time')
        ax.set_xlabel('Depot ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dispatch Delay (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Dispatch Delay by Depot (Top 20)', fontsize=14, fontweight='bold')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_delay_by_depot_box.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_delay_by_depot_box.png')

    if {'hour', 'dayofweek', 'dispatch_delay_minutes'} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot = df.pivot_table(index='dayofweek', columns='hour', values='dispatch_delay_minutes', aggfunc='mean')
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = pivot.reindex([d for d in order if d in pivot.index])
        sns.heatmap(pivot, cmap='RdYlGn_r', center=0, cbar_kws={'label': 'Mean Delay (minutes)'}, ax=ax)
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_title('Dispatch Delay Pattern: Hour × Day of Week', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(figures_dir / 'delay_hour_day_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved delay_hour_day_heatmap.png')

    if {'depot_id', 'on_time'} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(12, 8))
        depot_perf = df.groupby('depot_id').agg(on_time_rate=('on_time', 'mean'), trips=('dispatch_id', 'count'))
        depot_perf = depot_perf[depot_perf['trips'] >= 5]
        depot_perf['on_time_pct'] = depot_perf['on_time_rate'] * 100
        depot_perf = depot_perf.sort_values('on_time_pct')
        colors = ['red' if x < 80 else 'orange' if x < 90 else 'green' for x in depot_perf['on_time_pct']]
        ax.barh(depot_perf.index, depot_perf['on_time_pct'], color=colors)
        ax.axvline(90, color='blue', linestyle='--', linewidth=2, label='Target 90%')
        ax.set_xlabel('On-Time Delivery Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('On-Time Delivery Rate by Depot (min 5 trips)', fontsize=14, fontweight='bold')
        ax.legend()
        for idx, val in enumerate(depot_perf['on_time_pct']):
            ax.text(val + 1, idx, f"{val:.1f}%", va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_ontime_by_depot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_ontime_by_depot.png')

    if sku_cols:
        fig, ax = plt.subplots(figsize=(12, 8))
        sku_totals = df[sku_cols].sum().sort_values(ascending=True)
        ax.barh(sku_totals.index, sku_totals.values, color='steelblue')
        ax.set_xlabel('Total Quantity Dispatched', fontsize=12, fontweight='bold')
        ax.set_title('Dispatch Volume by SKU', fontsize=14, fontweight='bold')
        for i, val in enumerate(sku_totals.values):
            ax.text(val + max(sku_totals)*0.01, i, f"{val:,.0f}", va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_volume_by_sku.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_volume_by_sku.png')

    if 'delay_category' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        delay_counts = df['delay_category'].value_counts()
        wedges, texts, autotexts = ax.pie(delay_counts, labels=delay_counts.index, autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title(f'Dispatch Delay Categories (n={len(df):,})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_delay_category_pie.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_delay_category_pie.png')

    if 'date' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        daily = df.groupby('date').agg(dispatches=('dispatch_id', 'count'), total_qty=('total_qty', 'sum'))
        ax.plot(daily.index, daily['dispatches'], marker='o', linewidth=2, color='steelblue', label='Dispatch Count')
        ax.axhline(daily['dispatches'].mean(), color='red', linestyle='--', linewidth=2, label=f"Avg {daily['dispatches'].mean():.1f}")
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dispatches', fontsize=12, fontweight='bold')
        ax.set_title('Daily Dispatch Volume', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_volume_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_volume_timeseries.png')

    if {'vehicle_id', 'dispatch_delay_minutes'} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(12, 8))
        vehicle_perf = df.groupby('vehicle_id').agg(mean_delay=('dispatch_delay_minutes', 'mean'), trips=('dispatch_id', 'count'))
        vehicle_perf = vehicle_perf[vehicle_perf['trips'] >= 5]
        vehicle_perf = vehicle_perf.sort_values('mean_delay').tail(15)
        colors = ['red' if x > 60 else 'orange' if x > 30 else 'yellow' for x in vehicle_perf['mean_delay']]
        ax.barh(vehicle_perf.index, vehicle_perf['mean_delay'], color=colors)
        ax.axvline(30, color='blue', linestyle='--', linewidth=2, label='Target <30m')
        ax.set_xlabel('Mean Delay (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Top Vehicles by Mean Delay (min 5 trips)', fontsize=14, fontweight='bold')
        ax.legend()
        for idx, val in enumerate(vehicle_perf['mean_delay']):
            ax.text(val + 2, idx, f"{val:.1f}", va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'dispatch_delay_by_vehicle.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('Saved dispatch_delay_by_vehicle.png')


def main():
    data_path = Path('data/processed/dispatch_dataset.parquet')
    reports_dir = Path('reports')
    summaries_dir = reports_dir / 'summaries'
    figures_dir = reports_dir / 'figures'

    df, sku_cols = load_and_prepare(data_path)
    write_summary(df, sku_cols, reports_dir)
    grouped_summaries(df, sku_cols, summaries_dir)
    visualizations(df, sku_cols, figures_dir)

    logger.info('✅ Dispatch EDA complete (depot-based)')


if __name__ == '__main__':
    main()
