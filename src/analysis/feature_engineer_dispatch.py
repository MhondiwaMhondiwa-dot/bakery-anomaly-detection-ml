"""Feature engineering for dispatch dataset.

Produces per-dispatch features useful for anomaly detection:
- dispatch_delay_minutes, abs_delay, is_delayed (threshold 15 mins)
- hour, dayofweek, is_weekend
- rolling statistics (7-day, 30-day) by route_id and plant_id: mean, median, std, count
- route-level z-score of delay (based on historical mean/std)

Saves features to `data/features/dispatch_features.parquet` and a CSV sample.

Usage:
    python src/analysis/feature_engineer_dispatch.py --input data/processed/dispatch_dataset.parquet --out_dir data/features
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare(path: Path) -> pd.DataFrame:
    logger.info(f"Loading {path}")
    df = pd.read_parquet(path)
    # parse times
    for c in ['timestamp', 'expected_arrival', 'actual_arrival']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    # compute delay
    if 'expected_arrival' in df.columns and 'actual_arrival' in df.columns:
        df['dispatch_delay_minutes'] = (df['actual_arrival'] - df['expected_arrival']).dt.total_seconds() / 60.0
    else:
        df['dispatch_delay_minutes'] = np.nan
    # time features
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    else:
        df['hour'] = np.nan
        df['dayofweek'] = None
        df['is_weekend'] = False
    df['abs_delay'] = df['dispatch_delay_minutes'].abs()
    df['is_delayed_15'] = df['dispatch_delay_minutes'] > 15
    return df


def rolling_group_features(df: pd.DataFrame, group_col: str, windows=['7D','30D']):
    """Compute rolling window stats per group using time-based windows.
    Returns a DataFrame with columns named <group>_<window>_<stat>.
    """
    if 'timestamp' not in df.columns:
        logger.warning('No timestamp column found; skipping rolling features')
        return pd.DataFrame(index=df.index)
    # compute on subset without NaT timestamps, then map results back to original index
    valid_mask = df['timestamp'].notna()
    if valid_mask.sum() == 0:
        logger.warning('No valid timestamps found; skipping rolling features')
        return pd.DataFrame(index=df.index)
    df_valid = df.loc[valid_mask]
    # prepare output frame indexed by original df index
    out = pd.DataFrame(index=df.index)
    # iterate groups to avoid complex MultiIndex alignment issues
    groups = df_valid.groupby(group_col).groups
    for w in windows:
        name_mean = f'{group_col}_mean_{w}'
        name_median = f'{group_col}_median_{w}'
        name_std = f'{group_col}_std_{w}'
        name_count = f'{group_col}_count_{w}'
        out[name_mean] = np.nan
        out[name_median] = np.nan
        out[name_std] = np.nan
        out[name_count] = 0
        for grp, idxs in groups.items():
            gdf = df_valid.loc[idxs].sort_values('timestamp')
            if gdf.empty:
                continue
            # set timestamp index for rolling
            gdf_ts = gdf.set_index('timestamp')
            rolled = gdf_ts['dispatch_delay_minutes'].rolling(w, min_periods=1).agg(['mean','median','std','count'])
            # map rolled values back to original rows by timestamp order
            timestamps = df.loc[idxs, 'timestamp']
            out.loc[idxs, name_mean] = rolled['mean'].reindex(timestamps).values
            out.loc[idxs, name_median] = rolled['median'].reindex(timestamps).values
            out.loc[idxs, name_std] = rolled['std'].fillna(0).reindex(timestamps).values
            out.loc[idxs, name_count] = rolled['count'].reindex(timestamps).fillna(0).astype(int).values
    return out


def add_route_zscore(df: pd.DataFrame, group_col='route_id'):
    # compute group mean/std
    grp = df.groupby(group_col)['dispatch_delay_minutes'].agg(['mean','std']).rename(columns={'mean':f'{group_col}_mean','std':f'{group_col}_std'})
    df = df.merge(grp, how='left', left_on=group_col, right_index=True)
    df[f'{group_col}_zscore'] = (df['dispatch_delay_minutes'] - df[f'{group_col}_mean']) / df[f'{group_col}_std'].replace(0, np.nan)
    return df


def main(input_path: str, out_dir: str):
    p = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare(p)
    df_sorted = df.sort_values('timestamp') if 'timestamp' in df.columns else df

    # rolling features per route and per plant
    route_rolled = rolling_group_features(df_sorted, 'route_id', windows=['7D','30D'])
    plant_rolled = rolling_group_features(df_sorted, 'plant_id', windows=['7D','30D'])

    # merge rolled features
    features = df_sorted.reset_index(drop=True).copy()
    if not route_rolled.empty:
        features = pd.concat([features, route_rolled], axis=1)
    if not plant_rolled.empty:
        features = pd.concat([features, plant_rolled], axis=1)

    # add zscore per route
    features = add_route_zscore(features, 'route_id')

    # drop large/unnecessary columns
    drop_cols = [c for c in ['expected_arrival','actual_arrival'] if c in features.columns]
    features = features.drop(columns=drop_cols)

    # write out
    features_path = out / 'dispatch_features.parquet'
    features.to_parquet(features_path, index=False)
    logger.info(f'Wrote features to {features_path}')
    # write sample CSV
    sample_path = out / 'dispatch_features_sample.csv'
    features.sample(min(1000, len(features))).to_csv(sample_path, index=False)
    logger.info(f'Wrote sample to {sample_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed/dispatch_dataset.parquet')
    parser.add_argument('--out_dir', default='data/features')
    args = parser.parse_args()
    main(args.input, args.out_dir)
