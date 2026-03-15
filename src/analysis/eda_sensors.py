"""EDA for Equipment/IoT Sensors dataset.

Analyzes:
- Sensor readings by plant, equipment, metric name
- Metric value distributions and trends
- Equipment monitoring patterns

Outputs:
- reports/summaries/sensors_by_{plant,metric_name,equipment}.csv
- reports/figures/sensors_*.png
- reports/sensors_summary.txt
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
    summary.append(f"Sensors/IoT Dataset Summary")
    summary.append(f"="*60)
    summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary.append(f"\nColumns: {', '.join(df.columns.tolist())}")
    summary.append(f"\nData types:\n{df.dtypes.value_counts()}")
    summary.append(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    if 'metric_value' in df.columns:
        summary.append(f"\nMetric Value Stats:")
        summary.append(f"  Mean: {df['metric_value'].mean():.2f}")
        summary.append(f"  Median: {df['metric_value'].median():.2f}")
        summary.append(f"  Std: {df['metric_value'].std():.2f}")
        summary.append(f"  Min: {df['metric_value'].min():.2f}")
        summary.append(f"  Max: {df['metric_value'].max():.2f}")
    
    if 'metric_name' in df.columns:
        summary.append(f"\nTop Metric Names:")
        for name, count in df['metric_name'].value_counts().head(10).items():
            summary.append(f"  {name}: {count:,}")
    
    if 'equipment_id' in df.columns:
        summary.append(f"\nUnique Equipment: {df['equipment_id'].nunique()}")
    
    text = '\n'.join(summary)
    (out_dir / 'sensors_summary.txt').write_text(text, encoding='utf-8')
    logger.info(f'Wrote summary to {out_dir / "sensors_summary.txt"}')


def grouped_summaries(df: pd.DataFrame, out_dir: Path):
    summaries_dir = out_dir / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    # by plant
    if 'plant_id' in df.columns and 'metric_value' in df.columns:
        by_plant = df.groupby('plant_id')['metric_value'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        by_plant.to_csv(summaries_dir / 'sensors_by_plant.csv', index=False)
        logger.info('Wrote sensors_by_plant.csv')
    
    # by metric name
    if 'metric_name' in df.columns and 'metric_value' in df.columns:
        by_metric = df.groupby('metric_name')['metric_value'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        by_metric = by_metric.sort_values('count', ascending=False)
        by_metric.to_csv(summaries_dir / 'sensors_by_metric_name.csv', index=False)
        logger.info('Wrote sensors_by_metric_name.csv')
    
    # by equipment
    if 'equipment_id' in df.columns and 'metric_value' in df.columns:
        by_equip = df.groupby('equipment_id')['metric_value'].agg(['count', 'mean']).reset_index()
        by_equip = by_equip.sort_values('count', ascending=False).head(50)
        by_equip.to_csv(summaries_dir / 'sensors_by_equipment.csv', index=False)
        logger.info('Wrote sensors_by_equipment.csv')


def visualizations(df: pd.DataFrame, out_dir: Path):
    figs_dir = out_dir / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)

    def add_caption(fig, text):
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    
    # 1. Metric value histogram
    if 'metric_value' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df['metric_value'].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_xlabel('Metric Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Sensor Metric Values')
        add_caption(fig, 'Shows the overall spread of sensor values to spot skew or extreme ranges.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'sensors_value_hist.png', dpi=150)
        plt.close()
        logger.info('Saved sensors_value_hist.png')
    
    # 2. Metrics by name boxplot
    if 'metric_name' in df.columns and 'metric_value' in df.columns:
        top_metrics = df['metric_name'].value_counts().head(10).index
        df_subset = df[df['metric_name'].isin(top_metrics)]
        if len(df_subset) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_subset.boxplot(column='metric_value', by='metric_name', ax=ax)
            ax.set_xlabel('Metric Name')
            ax.set_ylabel('Metric Value')
            ax.set_title('Sensor Metrics by Name (Top 10)')
            plt.suptitle('')
            plt.xticks(rotation=45)
            add_caption(fig, 'Compares distributions for high-volume metrics to find unstable signals.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'sensors_by_metric_box.png', dpi=150)
            plt.close()
            logger.info('Saved sensors_by_metric_box.png')
    
    # 3. Timeseries
    if 'timestamp' in df.columns and 'metric_value' in df.columns:
        df_ts = df[df['timestamp'].notna()].copy()
        if len(df_ts) > 0:
            # floor uses lowercase 'h' to avoid pandas frequency parsing issue
            hourly = df_ts.groupby(df_ts['timestamp'].dt.floor('h'))['metric_value'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(hourly['timestamp'], hourly['metric_value'])
            ax.set_xlabel('Time')
            ax.set_ylabel('Average Metric Value')
            ax.set_title('Sensor Readings Over Time (Hourly Average)')
            plt.xticks(rotation=45)
            add_caption(fig, 'Tracks overall sensor signal drift or step changes over time.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'sensors_timeseries.png', dpi=150)
            plt.close()
            logger.info('Saved sensors_timeseries.png')
    
    # 4. Equipment bar chart
    if 'equipment_id' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df['equipment_id'].value_counts().head(15).plot(kind='bar', ax=ax, color='orange')
        ax.set_xlabel('Equipment ID')
        ax.set_ylabel('Number of Readings')
        ax.set_title('Top 15 Equipment by Number of Sensor Readings')
        plt.xticks(rotation=45)
        add_caption(fig, 'Highlights equipment generating the most telemetry for monitoring focus.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'sensors_by_equipment_bar.png', dpi=150)
        plt.close()
        logger.info('Saved sensors_by_equipment_bar.png')

    # 5. Reading volume heatmap by day and hour
    if 'dayofweek' in df.columns and 'hour' in df.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heat = df.pivot_table(index='dayofweek', columns='hour', values='metric_value', aggfunc='size').reindex(day_order)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(heat, cmap='Blues', ax=ax)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        ax.set_title('Sensor Reading Volume (Day vs Hour)')
        add_caption(fig, 'Shows when sensors are most active to detect monitoring gaps.')
        plt.tight_layout()
        plt.savefig(figs_dir / 'sensors_volume_heatmap_day_hour.png', dpi=150)
        plt.close()
        logger.info('Saved sensors_volume_heatmap_day_hour.png')

    # 6. Hourly trends for top 3 metrics
    if 'metric_name' in df.columns and 'metric_value' in df.columns and 'timestamp' in df.columns:
        top_metrics = df['metric_name'].value_counts().head(3).index
        df_top = df[df['metric_name'].isin(top_metrics)].copy()
        if not df_top.empty:
            df_top['hourly_ts'] = df_top['timestamp'].dt.floor('h')
            hourly = df_top.groupby(['hourly_ts', 'metric_name'])['metric_value'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(14, 5))
            for metric in top_metrics:
                series = hourly[hourly['metric_name'] == metric]
                ax.plot(series['hourly_ts'], series['metric_value'], label=str(metric))
            ax.set_xlabel('Time')
            ax.set_ylabel('Average Metric Value')
            ax.set_title('Hourly Trends for Top Metrics')
            ax.legend()
            plt.xticks(rotation=45)
            add_caption(fig, 'Compares time behavior across the most common metrics.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'sensors_top_metrics_hourly_trend.png', dpi=150)
            plt.close()
            logger.info('Saved sensors_top_metrics_hourly_trend.png')

    # 7. Metric value histogram by top 3 metrics
    if 'metric_name' in df.columns and 'metric_value' in df.columns:
        top_metrics = df['metric_name'].value_counts().head(3).index
        df_top = df[df['metric_name'].isin(top_metrics)]
        if not df_top.empty:
            fig, axes = plt.subplots(1, len(top_metrics), figsize=(15, 4), sharey=True)
            if len(top_metrics) == 1:
                axes = [axes]
            for i, metric in enumerate(top_metrics):
                subset = df_top[df_top['metric_name'] == metric]
                axes[i].hist(subset['metric_value'].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                axes[i].set_title(str(metric))
                axes[i].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            plt.suptitle('Value Distributions for Top Metrics', fontweight='bold')
            add_caption(fig, 'Shows whether key metrics have stable or multi‑modal behavior.')
            plt.tight_layout()
            plt.savefig(figs_dir / 'sensors_top_metrics_hist.png', dpi=150)
            plt.close()
            logger.info('Saved sensors_top_metrics_hist.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/equipment_iot_sensor_dataset.parquet')
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
    
    logger.info('Sensors/IoT EDA complete')


if __name__ == '__main__':
    main()
