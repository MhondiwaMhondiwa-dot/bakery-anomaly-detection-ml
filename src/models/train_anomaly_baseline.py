"""Phase 4: ML Model Development - Anomaly Detection Baseline Models

This module implements multiple baseline anomaly detection models aligned with project objectives:
1. Identify and quantify waste/anomalies in bakery operations
2. Detect anomalies in production, dispatch, quality, inventory, and sales
3. Recommend operational changes based on insights

Models Implemented:
- Isolation Forest (unsupervised anomaly detection)
- Local Outlier Factor (density-based anomaly detection)
- One-Class SVM (boundary-based anomaly detection)
- Statistical Baselines (Z-score, IQR)

Data Source: data/analytic/plant_daily.parquet (365 days, 52 features)
Plant Scope: Shepperton Plant Only

Usage:
    python src/models/train_anomaly_baseline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys
import io
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path('data/analytic')
MODELS_DIR = Path('reports/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class AnomalyDetectionPipeline:
    """Complete anomaly detection pipeline for bakery operations."""
    
    def __init__(self, contamination=0.05):
        """
        Initialize anomaly detection pipeline.
        
        Args:
            contamination (float): Expected proportion of anomalies (5% default)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load feature-engineered analytic dataset."""
        logger.info("Loading analytic dataset...")
        filepath = DATA_DIR / 'plant_daily.parquet'
        
        if not filepath.exists():
            raise FileNotFoundError(f"Analytic dataset not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} days of data with {df.shape[1]} features")
        
        # Verify Shepperton plant scope
        if 'plant_id' in df.columns:
            plants = df['plant_id'].unique()
            logger.info(f"Plant scope: {plants}")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for anomaly detection.
        
        Focuses on operational metrics aligned with project objectives:
        - Production metrics (volume, defects)
        - Dispatch metrics (delays, efficiency)
        - Quality metrics (QC pass rates, failures)
        - Inventory metrics (balances, stock movements)
        - Waste & Returns metrics
        - Sales metrics (demand patterns)
        """
        logger.info("Preparing features for anomaly detection...")
        
        # Define feature groups aligned with project objectives
        production_features = [
            'total_prod', 'avg_defect', 'high_defect_count'
        ]
        
        dispatch_features = [
            'avg_delay', 'late_pct', 'early_pct', 'total_dispatches'
        ]
        
        quality_features = [
            'qc_pass_rate', 'qc_fail_count', 'qc_moisture_fail', 
            'qc_seal_fail', 'qc_temp_fail', 'qc_weight_fail'
        ]
        
        waste_returns_features = [
            'total_waste', 'waste_rate', 'total_returns', 'return_rate'
        ]
        
        inventory_features = [
            'negative_balance_count', 'stock_movements', 'nearing_expiry_count'
        ]
        
        sales_features = [
            'total_sold', 'demand_collapse_pct', 'promotion_active'
        ]
        
        # Combine all feature groups
        all_features = (production_features + dispatch_features + quality_features + 
                       waste_returns_features + inventory_features + sales_features)
        
        # Select available features
        available_features = [f for f in all_features if f in df.columns]
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # Extract feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store metadata
        self.feature_names = available_features
        self.feature_groups = {
            'production': [f for f in production_features if f in available_features],
            'dispatch': [f for f in dispatch_features if f in available_features],
            'quality': [f for f in quality_features if f in available_features],
            'waste_returns': [f for f in waste_returns_features if f in available_features],
            'inventory': [f for f in inventory_features if f in available_features],
            'sales': [f for f in sales_features if f in available_features]
        }
        
        return X, df
    
    def get_ground_truth_labels(self, df):
        """
        Extract ground truth anomaly labels from existing anomaly flags.
        
        Uses pre-computed anomaly flags from feature engineering:
        - production_anomaly, dispatch_anomaly, qc_anomaly
        - waste_anomaly, return_anomaly, sales_anomaly, inventory_anomaly
        """
        anomaly_cols = [
            'production_anomaly', 'dispatch_anomaly', 'qc_anomaly',
            'waste_anomaly', 'return_anomaly', 'sales_anomaly', 'inventory_anomaly'
        ]
        
        available_anomaly_cols = [col for col in anomaly_cols if col in df.columns]
        
        if not available_anomaly_cols:
            logger.warning("No ground truth anomaly labels found")
            return None
        
        # A day is anomalous if ANY category flags it as anomalous
        y_true = df[available_anomaly_cols].any(axis=1).astype(int)
        
        anomaly_count = y_true.sum()
        anomaly_pct = anomaly_count / len(y_true) * 100
        logger.info(f"Ground truth: {anomaly_count}/{len(y_true)} ({anomaly_pct:.1f}%) days are anomalous")
        
        return y_true.values
    
    def train_isolation_forest(self, X_train, X_test):
        """Train Isolation Forest model."""
        logger.info("Training Isolation Forest...")
        
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Fit on training data
        model.fit(X_train)
        
        # Predict on test data (-1 = anomaly, 1 = normal)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Get anomaly scores (lower = more anomalous)
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        
        # Convert predictions to binary (1 = anomaly, 0 = normal)
        y_pred_train_binary = (y_pred_train == -1).astype(int)
        y_pred_test_binary = (y_pred_test == -1).astype(int)
        
        self.models['isolation_forest'] = model
        
        return {
            'train_predictions': y_pred_train_binary,
            'test_predictions': y_pred_test_binary,
            'train_scores': scores_train,
            'test_scores': scores_test
        }
    
    def train_lof(self, X_train, X_test):
        """Train Local Outlier Factor model."""
        logger.info("Training Local Outlier Factor...")
        
        # LOF for training data
        model_train = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=False,
            n_neighbors=20
        )
        y_pred_train = model_train.fit_predict(X_train)
        scores_train = model_train.negative_outlier_factor_
        
        # LOF for test data (novelty detection)
        model_test = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_neighbors=20
        )
        model_test.fit(X_train)
        y_pred_test = model_test.predict(X_test)
        scores_test = model_test.score_samples(X_test)
        
        # Convert to binary
        y_pred_train_binary = (y_pred_train == -1).astype(int)
        y_pred_test_binary = (y_pred_test == -1).astype(int)
        
        self.models['lof'] = model_test
        
        return {
            'train_predictions': y_pred_train_binary,
            'test_predictions': y_pred_test_binary,
            'train_scores': scores_train,
            'test_scores': scores_test
        }
    
    def train_ocsvm(self, X_train, X_test):
        """Train One-Class SVM model."""
        logger.info("Training One-Class SVM...")
        
        model = OneClassSVM(
            kernel='rbf',
            gamma='scale',   # 1/(n_features * X.var()) — better calibrated than 'auto'
            nu=self.contamination
        )
        
        model.fit(X_train)
        
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        
        # Calibrate predictions at the contamination percentile of training scores
        # This guarantees ~contamination% are flagged rather than using the raw boundary
        threshold = np.percentile(scores_train, self.contamination * 100)
        y_pred_train_binary = (scores_train < threshold).astype(int)
        y_pred_test_binary = (scores_test < threshold).astype(int)
        
        self.models['ocsvm'] = model
        
        return {
            'train_predictions': y_pred_train_binary,
            'test_predictions': y_pred_test_binary,
            'train_scores': scores_train,
            'test_scores': scores_test
        }
    
    def train_statistical_baseline(self, X_train, X_test):
        """Train statistical baseline (Z-score based)."""
        logger.info("Training Statistical Baseline (Z-score)...")
        
        # Calculate mean and std from training data
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        
        # Calculate Z-scores
        z_scores_train = np.abs((X_train - means) / (stds + 1e-10))
        z_scores_test = np.abs((X_test - means) / (stds + 1e-10))
        
        # Max Z-score across features
        max_z_train = z_scores_train.max(axis=1)
        max_z_test = z_scores_test.max(axis=1)
        
        # Threshold at 3 standard deviations
        threshold = 3.0
        y_pred_train = (max_z_train > threshold).astype(int)
        y_pred_test = (max_z_test > threshold).astype(int)
        
        return {
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_scores': -max_z_train,  # Negative for consistency
            'test_scores': -max_z_test
        }
    
    def evaluate_model(self, y_true, y_pred, scores, model_name):
        """Evaluate model performance."""
        if y_true is None:
            logger.warning(f"Skipping evaluation for {model_name} (no ground truth)")
            return {}
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate ROC-AUC if possible
        try:
            roc_auc = roc_auc_score(y_true, -scores)  # Negative scores (lower = more anomalous)
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'anomalies_detected': int(y_pred.sum()),
            'total_samples': int(len(y_pred))
        }
        
        logger.info(f"{model_name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return metrics
    
    def run_cross_validation(self, X, y_true=None, n_splits=5):
        """Run time-series cross-validation."""
        logger.info(f"\nRunning {n_splits}-fold time-series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\n--- Fold {fold}/{n_splits} ---")
            logger.info(f"Train: {len(train_idx)} days, Test: {len(test_idx)} days")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame to preserve column names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
            
            # Get ground truth for test set
            y_test_true = y_true[test_idx] if y_true is not None else None
            
            fold_results = {'fold': fold, 'models': {}}
            
            # Train all models
            models_to_train = {
                'isolation_forest': self.train_isolation_forest,
                'lof': self.train_lof,
                'ocsvm': self.train_ocsvm,
                'statistical': self.train_statistical_baseline
            }
            
            for model_name, train_func in models_to_train.items():
                try:
                    predictions = train_func(X_train_scaled, X_test_scaled)
                    metrics = self.evaluate_model(
                        y_test_true, 
                        predictions['test_predictions'],
                        predictions['test_scores'],
                        model_name
                    )
                    fold_results['models'][model_name] = metrics
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    fold_results['models'][model_name] = {'error': str(e)}
            
            cv_results.append(fold_results)
        
        return cv_results
    
    def train_final_models(self, X, y_true=None):
        """Train final models on full dataset."""
        logger.info("\n" + "=" * 80)
        logger.info("Training final models on full dataset...")
        logger.info("=" * 80)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        final_predictions = {}
        
        # Isolation Forest
        if_results = self.train_isolation_forest(X_scaled, X_scaled)
        final_predictions['isolation_forest'] = if_results['train_predictions']
        
        # LOF — fit on full dataset with novelty=True so we get consistent predictions
        lof_model = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_neighbors=20
        )
        lof_model.fit(X_scaled)
        lof_scores = lof_model.score_samples(X_scaled)
        # Calibrate at contamination percentile (guarantees expected flagging rate)
        lof_threshold = np.percentile(lof_scores, self.contamination * 100)
        final_predictions['lof'] = (lof_scores < lof_threshold).astype(int)
        self.models['lof'] = lof_model
        
        # One-Class SVM
        ocsvm_results = self.train_ocsvm(X_scaled, X_scaled)
        final_predictions['ocsvm'] = ocsvm_results['train_predictions']
        
        # Statistical baseline
        stat_results = self.train_statistical_baseline(X_scaled, X_scaled)
        final_predictions['statistical'] = stat_results['train_predictions']
        
        # Ensemble: majority vote across the 4 core models (≥2 agree)
        ensemble_votes = np.column_stack([
            final_predictions['isolation_forest'],
            final_predictions['lof'],
            final_predictions['ocsvm'],
            final_predictions['statistical']
        ])
        final_predictions['ensemble'] = (ensemble_votes.sum(axis=1) >= 2).astype(int)
        
        # Store anomaly scores
        self.anomaly_scores = {
            'isolation_forest': if_results['train_scores'],
            'lof': lof_scores,
            'ocsvm': ocsvm_results['train_scores'],
            'statistical': stat_results['train_scores']
        }
        
        logger.info("Final models trained successfully")
        
        return final_predictions
    
    def save_results(self, cv_results, final_predictions, df):
        """Save model results and flagged anomalies."""
        logger.info("\nSaving results...")
        
        # 1. Save cross-validation results
        cv_report = {
            'timestamp': datetime.now().isoformat(),
            'contamination': self.contamination,
            'features_used': self.feature_names,
            'feature_groups': self.feature_groups,
            'cross_validation_folds': cv_results
        }
        
        with open(MODELS_DIR / 'baseline_cv_report.json', 'w') as f:
            json.dump(cv_report, f, indent=2)
        logger.info(f"✅ Saved: {MODELS_DIR / 'baseline_cv_report.json'}")
        
        # 2. Save flagged anomalies with details
        anomalies_df = df.copy()
        
        # Add model predictions
        for model_name, predictions in final_predictions.items():
            anomalies_df[f'{model_name}_anomaly'] = predictions
        
        # Add anomaly scores
        for model_name, scores in self.anomaly_scores.items():
            anomalies_df[f'{model_name}_score'] = scores
        
        # Filter to anomalous days (any model flagged)
        anomaly_mask = (anomalies_df[[f'{m}_anomaly' for m in final_predictions.keys()]].sum(axis=1) > 0)
        anomalous_days = anomalies_df[anomaly_mask].copy()
        
        # Sort by ensemble agreement (most models agreeing)
        model_cols = [f'{m}_anomaly' for m in final_predictions.keys()]
        anomalous_days['model_agreement'] = anomalous_days[model_cols].sum(axis=1)
        anomalous_days = anomalous_days.sort_values('model_agreement', ascending=False)
        
        # Save
        output_file = MODELS_DIR / 'flagged_anomalies_baseline.csv'
        anomalous_days.to_csv(output_file, index=True)
        logger.info(f"✅ Saved {len(anomalous_days)} anomalous days: {output_file}")
        
        # 3. Save summary statistics
        summary = {
            'total_days': len(df),
            'models_trained': list(final_predictions.keys()),
            'anomalies_by_model': {
                model: int(preds.sum()) for model, preds in final_predictions.items()
            },
            'top_10_anomalous_dates': [str(d) for d in anomalous_days.head(10).index.tolist()]
        }
        
        with open(MODELS_DIR / 'model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Saved: {MODELS_DIR / 'model_summary.json'}")
        
        return anomalous_days


def main():
    """Main execution pipeline."""
    print("\n" + "=" * 80)
    print("PHASE 4: ANOMALY DETECTION BASELINE MODELS")
    print("Bakery Operations - Shepperton Plant")
    print("=" * 80 + "\n")
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(contamination=0.05)
    
    # Load data
    df = pipeline.load_data()
    
    # Prepare features
    X, df_full = pipeline.prepare_features(df)
    
    # Get ground truth labels
    y_true = pipeline.get_ground_truth_labels(df_full)
    
    # Run cross-validation
    cv_results = pipeline.run_cross_validation(X, y_true=y_true, n_splits=5)
    
    # Train final models on full dataset
    final_predictions = pipeline.train_final_models(X, y_true=y_true)
    
    # Save results
    anomalous_days = pipeline.save_results(cv_results, final_predictions, df_full)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] PHASE 4 COMPLETE - BASELINE MODELS TRAINED")
    print("=" * 80)
    print(f"\nResults saved to: {MODELS_DIR}/")
    print(f"  - baseline_cv_report.json (cross-validation results)")
    print(f"  - flagged_anomalies_baseline.csv ({len(anomalous_days)} anomalies)")
    print(f"  - model_summary.json (summary statistics)")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
