"""Train and evaluate simple anomaly detection baselines.

This script implements:
- Time-based cross-validation splitting (by date segments)
- IsolationForest baseline (scikit-learn)
- Z-score baseline on `dispatch_delay_minutes` (if present)
- Evaluation (precision/recall/f1/roc_auc when labels exist)

Outputs:
- `reports/models/baseline_report.json` with per-fold and aggregate metrics
- `reports/models/flagged_anomalies.csv` with top anomalies across dataset

Usage:
    python src/models/train_baseline.py --input data/features/dispatch_features.parquet
"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_splits(df, n_splits=5, time_col='timestamp'):
    """Yield (train_idx, test_idx) index arrays using date-based segmentation.
    Splits unique dates into n_splits contiguous segments and uses earlier segments as train and the next segment as test.
    """
    if time_col not in df.columns:
        # fallback to simple row splits
        idx = np.arange(len(df))
        sizes = np.array_split(idx, n_splits)
        for i in range(len(sizes)-1):
            train_idx = np.concatenate(sizes[:i+1])
            test_idx = sizes[i+1]
            yield train_idx, test_idx
        return

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(time_col)
    dates = df[time_col].dt.date.unique()
    if len(dates) < n_splits:
        n_splits = max(2, len(dates))
    date_segments = np.array_split(dates, n_splits)
    # build masks
    for i in range(len(date_segments)-1):
        train_dates = np.concatenate(date_segments[:i+1])
        test_dates = date_segments[i+1]
        train_idx = df[df[time_col].dt.date.isin(train_dates)].index.values
        test_idx = df[df[time_col].dt.date.isin(test_dates)].index.values
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def eval_if_labels(y_true, scores, higher_score_more_anomalous=True):
    out = {}
    if y_true is None:
        return out
    # y_true expected boolean (True for anomaly)
    y = (y_true == 1) | (y_true == True)
    # use scores to compute ROC AUC if possible
    try:
        auc = roc_auc_score(y.astype(int), scores)
        out['roc_auc'] = float(auc)
    except Exception:
        out['roc_auc'] = None
    # pick a threshold: top contamination fraction will be handled externally; skip thresholding here
    return out


def run_baselines(df, out_dir: Path, contamination=0.02, n_splits=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {'folds': []}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = None
    for candidate in ['anomaly', 'label', 'is_anomaly']:
        if candidate in df.columns:
            label_col = candidate
            break

    logger.info(f"Numeric cols used: {numeric_cols[:20]}")
    # run time-based CV
    fold_i = 0
    for train_idx, test_idx in time_splits(df, n_splits=n_splits, time_col='timestamp'):
        fold_i += 1
        logger.info(f"Fold {fold_i}: train={len(train_idx)} test={len(test_idx)}")
        # use positional indexing since train_idx/test_idx may be integer positions
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train = train[numeric_cols].fillna(0).astype(float)
        X_test = test[numeric_cols].fillna(0).astype(float)

        # IsolationForest
        clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        clf.fit(X_train)
        scores_if = -clf.decision_function(X_test)  # higher = more anomalous
        preds_if = clf.predict(X_test)  # -1 anomaly, 1 normal
        anomalies_if = preds_if == -1

        # z-score baseline on dispatch_delay_minutes if present
        z_metrics = None
        if 'dispatch_delay_minutes' in numeric_cols:
            mu = train['dispatch_delay_minutes'].mean()
            sd = train['dispatch_delay_minutes'].std() if train['dispatch_delay_minutes'].std() > 0 else 1.0
            z = (test['dispatch_delay_minutes'] - mu) / sd
            z_scores = z.abs()
            z_thresh = 3.0
            anomalies_z = z_scores > z_thresh
        else:
            z_scores = None
            anomalies_z = None

        fold_report = {
            'fold': fold_i,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'if_top_fraction': float(contamination),
        }

        # evaluation if labels present
        if label_col is not None:
            y_test = test[label_col]
            # compute precision/recall/f1 for IF and z
            p_if, r_if, f_if, _ = precision_recall_fscore_support(y_test, anomalies_if, average='binary', zero_division=0)
            fold_report.update({'if_precision': float(p_if), 'if_recall': float(r_if), 'if_f1': float(f_if)})
            if anomalies_z is not None:
                p_z, r_z, f_z, _ = precision_recall_fscore_support(y_test, anomalies_z, average='binary', zero_division=0)
                fold_report.update({'z_precision': float(p_z), 'z_recall': float(r_z), 'z_f1': float(f_z)})
            # ROC AUCs
            try:
                fold_report['if_auc'] = float(roc_auc_score(y_test.astype(int), scores_if))
            except Exception:
                fold_report['if_auc'] = None
        else:
            # no labels; report simple anomaly counts and score stats
            fold_report.update({
                'if_anomalies': int(anomalies_if.sum()),
                'if_score_mean': float(np.nanmean(scores_if)),
                'if_score_std': float(np.nanstd(scores_if)),
            })
            if anomalies_z is not None:
                fold_report.update({'z_anomalies': int(anomalies_z.sum())})

        report['folds'].append(fold_report)

    # aggregate
    report['n_folds'] = len(report['folds'])
    # write report
    rep_path = out_dir / 'baseline_report.json'
    rep_path.write_text(json.dumps(report, indent=2))
    logger.info(f'Wrote report to {rep_path}')

    # Full dataset anomaly scoring (train on all data)
    if numeric_cols:
        X_all = df[numeric_cols].fillna(0).astype(float)
        clf_all = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        clf_all.fit(X_all)
        scores_all = -clf_all.decision_function(X_all)
        preds_all = clf_all.predict(X_all)
        df_out = df.copy()
        df_out['if_score'] = scores_all
        df_out['if_anomaly'] = preds_all == -1
    else:
        df_out = df.copy()

    # zscore across dispatch_delay_minutes
    if 'dispatch_delay_minutes' in df_out.columns:
        m = df_out['dispatch_delay_minutes'].mean()
        s = df_out['dispatch_delay_minutes'].std() if df_out['dispatch_delay_minutes'].std() > 0 else 1.0
        df_out['delay_zscore'] = (df_out['dispatch_delay_minutes'] - m) / s
        df_out['z_anomaly'] = df_out['delay_zscore'].abs() > 3.0

    flagged_path = out_dir / 'flagged_anomalies.csv'
    # save top anomalies by IF score
    if 'if_score' in df_out.columns:
        top = df_out.sort_values('if_score', ascending=False).head(1000)
        top.to_csv(flagged_path, index=False)
        logger.info(f'Wrote flagged anomalies sample to {flagged_path}')
    else:
        df_out.to_csv(flagged_path, index=False)
        logger.info(f'Wrote dataset to {flagged_path}')

    return report, rep_path, flagged_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/features/dispatch_features.parquet')
    parser.add_argument('--out_dir', default='reports/models')
    parser.add_argument('--contamination', type=float, default=0.02)
    parser.add_argument('--n_splits', type=int, default=5)
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        logger.error(f'Input file not found: {p}')
        return
    df = pd.read_parquet(p)
    report, rep_path, flagged_path = run_baselines(df, Path(args.out_dir), contamination=args.contamination, n_splits=args.n_splits)
    print('Report written to', rep_path)
    print('Flagged anomalies CSV:', flagged_path)


if __name__ == '__main__':
    main()
