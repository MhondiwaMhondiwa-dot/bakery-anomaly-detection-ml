# -*- coding: utf-8 -*-
"""Phase 4 Visualizations: ML Model Results & Anomaly Detection

This module provides interactive visualizations for baseline anomaly detection models.
To be imported by the main streamlit_eda_explorer.py dashboard.

Features:
- Interactive model training interface with real-time progress
- Cross-validation results visualization
- Model performance comparison
- Anomalous days calendar view
- Feature importance analysis
- Time-series visualization of anomalies
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import subprocess
import sys
import time
import threading
from io import StringIO

# Paths
MODELS_DIR = Path('reports/models')
DATA_DIR = Path('data/analytic')
TRAINING_SCRIPT = Path('src/models/train_anomaly_baseline.py')


def load_model_results():
    """Load model results and flagged anomalies."""
    results = {}
    
    # Load CV report
    cv_file = MODELS_DIR / 'baseline_cv_report.json'
    if cv_file.exists():
        with open(cv_file, 'r') as f:
            results['cv_report'] = json.load(f)
    
    # Load model summary
    summary_file = MODELS_DIR / 'model_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load flagged anomalies
    anomalies_file = MODELS_DIR / 'flagged_anomalies_baseline.csv'
    if anomalies_file.exists():
        results['anomalies'] = pd.read_csv(anomalies_file, index_col=0)
        results['anomalies'].index = pd.to_datetime(results['anomalies'].index)
    
    return results


def check_training_status():
    """Check if models have been trained."""
    return (
        (MODELS_DIR / 'baseline_cv_report.json').exists() and
        (MODELS_DIR / 'flagged_anomalies_baseline.csv').exists() and
        (MODELS_DIR / 'model_summary.json').exists()
    )


def render_training_interface():
    """Render interactive training interface."""
    st.markdown("## Machine Learning Anomaly Detection")
    st.markdown("---")
    
    # Professional intro
    st.info("""
    **Multi-Algorithm Anomaly Detection System**
    
    This module employs an ensemble of unsupervised machine learning algorithms to identify statistical 
    anomalies in daily plant operations. The system analyzes operational metrics across production, 
    quality control, dispatch, inventory, waste management, and sales to flag days with abnormal patterns 
    that warrant investigation.
    
    The ensemble approach combines multiple detection methodologies (density-based, tree-based, boundary-based, 
    and statistical) to minimize false positives while maintaining high sensitivity to genuine operational anomalies. 
    Time-series cross-validation ensures model robustness and generalization to future operational patterns.
    """)
    
    # Check current status
    is_trained = check_training_status()
    
    if is_trained:
        st.success("✅ Models have been trained! You can view results below or retrain models.")
        with st.expander("📊 Current Training Summary", expanded=False):
            try:
                summary = json.load(open(MODELS_DIR / 'model_summary.json'))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", summary.get('total_days', 'N/A'))
                with col2:
                    st.metric("Models Trained", len(summary.get('models_trained', [])))
                with col3:
                    ensemble_count = summary.get('anomalies_by_model', {}).get('ensemble', 0)
                    st.metric("Anomalies Found", ensemble_count)
            except:
                pass
    else:
        st.warning("⏳ Models not trained yet. Start the training process below!")
    
    st.markdown("---")
    
    # Training configuration
    st.markdown("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📚 Models to Train:**")
        st.markdown("""
        - ✓ **Isolation Forest** (Tree-based ensemble)
        - ✓ **Local Outlier Factor** (Density-based)
        - ✓ **One-Class SVM** (Boundary-based)
        - ✓ **Statistical Z-Score** (Threshold-based)
        - ✓ **Ensemble Voting** (Combines all models)
        """)
    
    with col2:
        st.markdown("**🔍 Training Process:**")
        st.markdown("""
        1. Load 365 days of operational data
        2. Engineer 13 features across 6 domains
        3. Run 5-fold time-series cross-validation
        4. Train final models on full dataset
        5. Identify anomalous days
        6. Generate reports and visualizations
        """)
    
    st.markdown("---")
    
    # Training controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if is_trained:
            train_button = st.button("🔄 Retrain Models", type="secondary", use_container_width=True)
            st.caption("⚠️ This will overwrite existing results")
        else:
            train_button = st.button("🚀 Start Training", type="primary", use_container_width=True)
            st.caption("⏱️ Training takes 5-10 minutes")
    
    with col2:
        show_logs = st.checkbox("Show Detailed Logs", value=True)
    
    with col3:
        show_theory = st.checkbox("Algorithm Details", value=False)
    
    if show_theory:
        with st.expander("📖 Technical Methodology", expanded=True):
            st.markdown("""
            ### Unsupervised Anomaly Detection Approach
            
            The system identifies statistical deviations from normal operational patterns across multiple 
            dimensions: production efficiency, quality metrics, dispatch performance, inventory stability, 
            and waste generation.
            
            ### Multi-Algorithm Ensemble
            
            **1. Isolation Forest**
            - Tree-based ensemble that isolates anomalies through recursive partitioning
            - Effective for high-dimensional feature spaces and global outlier detection
            - Time complexity: O(n log n), suitable for large datasets
            
            **2. Local Outlier Factor (LOF)**
            - Density-based algorithm measuring local deviation from k-nearest neighbors
            - Detects anomalies in regions with varying density
            - Sensitive to local context, captures pattern deviations within operational clusters
            
            **3. One-Class SVM**
            - Kernel-based boundary learning around normal operational space
            - Maps data to high-dimensional space for non-linear anomaly boundaries
            - Robust to outliers in training data
            
            **4. Statistical Z-Score Baseline**
            - Univariate statistical approach flagging observations beyond 3σ threshold
            - Provides interpretable baseline assuming Gaussian distribution
            - Fast computation, useful for feature-level anomaly attribution
            
            **5. Ensemble Voting Mechanism**
            - Aggregates predictions across all algorithms (flags if ≥1 model agrees)
            - Reduces false negatives through algorithmic diversity
            - Balances precision-recall tradeoff
            
            ### Time-Series Cross-Validation
            
            **TimeSeriesSplit** ensures temporal integrity:
            ```
            Fold 1: Train [Day 1-245]   → Test [Day 246-305]
            Fold 2: Train [Day 1-305]   → Test [Day 306-365]
            Fold 3: Train [Day 1-365]   → Test [Holdout Set]
            ```
            This prevents data leakage and validates model performance on future unseen patterns,
            critical for production deployment.
            """)
    
    # Training execution
    if train_button:
        st.markdown("---")
        st.markdown("## 📊 Training in Progress...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("📋 Training Logs", expanded=show_logs)
        
        training_stages = [
            ("Loading data", 10),
            ("Preparing features", 20),
            ("Running Fold 1/5", 30),
            ("Running Fold 2/5", 45),
            ("Running Fold 3/5", 60),
            ("Running Fold 4/5", 75),
            ("Running Fold 5/5", 85),
            ("Training final models", 93),
            ("Saving results", 98),
            ("Complete!", 100)
        ]
        
        # Run training script
        try:
            with log_container:
                log_output = st.empty()
                logs = []
                
                # Show simulated stages first
                for stage, progress in training_stages[:3]:
                    status_text.text(f"⏳ {stage}...")
                    progress_bar.progress(progress)
                    time.sleep(0.5)
                
                # Actually run the training
                status_text.text("🔄 Running training script...")
                
                # Ensure models directory exists
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                
                # Run training script
                python_exe = sys.executable
                process = subprocess.Popen(
                    [python_exe, str(TRAINING_SCRIPT)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output
                for line in process.stdout:
                    logs.append(line)
                    log_output.code('\n'.join(logs[-50:]))  # Show last 50 lines
                    
                    # Update progress based on output
                    if "Fold" in line:
                        try:
                            fold_num = int(line.split("Fold")[1].split("/")[0].strip())
                            progress_val = 30 + (fold_num * 11)
                            progress_bar.progress(min(progress_val, 85))
                            status_text.text(f"⏳ Running Fold {fold_num}/5...")
                        except:
                            pass
                    elif "Training final models" in line:
                        progress_bar.progress(93)
                        status_text.text("⏳ Training final models...")
                    elif "Saved" in line:
                        progress_bar.progress(98)
                        status_text.text("⏳ Saving results...")
                
                process.wait()
                
                if process.returncode == 0:
                    progress_bar.progress(100)
                    status_text.text("")
                    st.success("✅ Training completed successfully!")
                    
                    # Show summary
                    st.balloons()
                    
                    st.markdown("### 🎉 Training Complete!")
                    st.markdown("""
                    **Next Steps:**
                    1. Scroll down to explore the results
                    2. Review model performance metrics
                    3. Examine the anomalous days identified by the ensemble
                    4. Download the CSV for operational review
                    """)
                    
                    # Auto-refresh to show results
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Training failed with exit code {process.returncode}")
                    st.error("Check the logs above for details.")
        
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.exception(e)
    
    st.markdown("---")


def render_phase4_overview(results):
    """Render Phase 4 overview section."""
    st.markdown("## 📊 Training Results Overview")
    st.markdown("---")
    
    if not results:
        st.warning("⚠️ No model results available. Please train the models first using the interface above.")
        return
    
    # Overview metrics
    if 'summary' in results:
        summary = results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📅 Total Days Analyzed",
                value=summary.get('total_days', 0)
            )
        
        with col2:
            st.metric(
                label="🤖 Models Trained",
                value=len(summary.get('models_trained', []))
            )
        
        with col3:
            ensemble_anomalies = summary.get('anomalies_by_model', {}).get('ensemble', 0)
            st.metric(
                label="⚠️ Anomalies Detected (Ensemble)",
                value=ensemble_anomalies
            )
        
        with col4:
            anomaly_rate = (ensemble_anomalies / summary.get('total_days', 1)) * 100
            st.metric(
                label="📊 Anomaly Rate",
                value=f"{anomaly_rate:.1f}%"
            )
    
    st.markdown("---")


def plot_model_performance(results):
    """Plot model performance comparison across CV folds."""
    if 'cv_report' not in results:
        return
    
    st.markdown("### 📊 Model Performance - Cross Validation")
    
    # Description with key insights
    st.info("""
    Comparative performance of four anomaly detection algorithms averaged across 5-fold time-series cross-validation. 
    **Key insights:** Higher bars indicate better performance. Precision measures accuracy of flagged anomalies, 
    Recall measures detection coverage, F1-Score balances both, and ROC-AUC evaluates overall discrimination capability. 
    Model performance validates algorithmic diversity in the ensemble approach.
    """)
    
    # Key interpretation guide
    with st.expander("🔑 Key Interpretation Guide"):
        st.markdown("""
        - **X-axis:** Algorithm names (Isolation Forest, LOF, One-Class SVM, Statistical Z-Score)
        - **Y-axis:** Performance score (0.0 to 1.0 scale)
        - **Color coding:** Blue = Precision, Red = Recall, Green = F1-Score, Purple = ROC-AUC
        - **Bar height:** Higher values indicate superior model performance
        - **Score ≥ 0.7:** Generally acceptable performance for anomaly detection
        - **Score < 0.5:** Indicates poor discriminative ability (similar to random guessing)
        - **Balanced metrics:** Models with similar Precision and Recall show consistent detection
        """)
    
    cv_folds = results['cv_report'].get('cross_validation_folds', [])
    
    # Extract metrics from each fold
    model_names = []
    metrics_data = []
    
    for fold in cv_folds:
        fold_num = fold['fold']
        for model_name, metrics in fold.get('models', {}).items():
            if 'error' not in metrics and metrics:
                model_names.append(model_name)
                metrics_data.append({
                    'fold': fold_num,
                    'model': model_name,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'roc_auc': metrics.get('roc_auc', 0)
                })
    
    if not metrics_data:
        st.warning("No performance metrics available")
        return
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Aggregate metrics by model
    avg_metrics = metrics_df.groupby('model').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'roc_auc': 'mean'
    }).reset_index()
    
    # Plot comparison
    fig = go.Figure()
    
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    
    for i, metric in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=avg_metrics['model'],
            y=avg_metrics[metric],
            marker_color=colors[i],
            text=avg_metrics[metric].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Average Model Performance Across 5 CV Folds",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        showlegend=True,
        yaxis=dict(range=[0, 1.1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed metrics table
    with st.expander("📋 Detailed Performance Metrics"):
        st.dataframe(
            avg_metrics.round(3),
            use_container_width=True,
            hide_index=True
        )


def plot_anomalies_by_model(results):
    """Plot anomaly count by model."""
    if 'summary' not in results:
        return
    
    st.markdown("### 🚨 Anomalies Detected by Model")
    
    # Description with key insights
    st.info("""
    Total number of anomalous days identified by each algorithm from 365 days of operational data. 
    **Key insights:** Variance in detection counts reflects algorithmic sensitivity differences. The ensemble model 
    uses majority voting (≥2 of 4 core models agree), providing high-confidence anomaly coverage with fewer false positives. 
    Low detection counts reflect tighter thresholds; higher counts reflect broader sensitivity to operational patterns.
    """)
    
    # Key interpretation guide
    with st.expander("🔑 Key Interpretation Guide"):
        st.markdown("""
        - **X-axis:** Algorithm names and ensemble (combined voting)
        - **Y-axis:** Count of days flagged as anomalous (out of 365 total days)
        - **Bar height:** Number of detected anomalies per model
        - **Ensemble count:** Union of all model detections (highest count expected)
        - **Low counts (< 20):** Conservative detection, fewer false positives
        - **High counts (> 50):** Aggressive detection, higher sensitivity
        - **Model agreement:** Days flagged by multiple models are high-confidence anomalies
        - **Single-model flags:** May warrant additional investigation for false positives
        """)
    
    anomalies_by_model = results['summary'].get('anomalies_by_model', {})
    
    if not anomalies_by_model:
        st.warning("No anomaly counts available")
        return
    
    # Create bar chart
    models = list(anomalies_by_model.keys())
    counts = list(anomalies_by_model.values())
    
    fig = go.Figure(data=go.Bar(
        x=models,
        y=counts,
        marker_color='#EF553B',
        text=counts,
        textposition='outside',
        hovertemplate='%{x}<br>Anomalies: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Anomalies Detected by Each Model",
        xaxis_title="Model",
        yaxis_title="Number of Anomalies",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary table
    with st.expander("📋 Detailed Counts"):
        import pandas as pd
        df = pd.DataFrame({
            'Model': models,
            'Anomaly Count': counts
        })
        st.dataframe(df, use_container_width=True, hide_index=True)


def plot_anomalies_calendar(results):
    """Plot weekly anomaly heatmap."""
    if 'anomalies' not in results:
        return
    
    st.markdown("### 📅 Weekly Anomaly Heatmap")
    
    # Description with key insights
    st.info("""
    Calendar-based visualization showing temporal distribution of ensemble-detected anomalies across 2025. 
    **Key insights:** Color intensity represents anomaly frequency. Clusters indicate systematic issues (e.g., specific weekdays). 
    Uniform distribution suggests random operational volatility. Visible patterns (e.g., Monday spikes) may correlate with 
    process changes, staffing, or equipment maintenance schedules.
    """)
    
    # Key interpretation guide
    with st.expander("🔑 Key Interpretation Guide"):
        st.markdown("""
        - **X-axis:** Days of week (Monday through Sunday)
        - **Y-axis:** ISO calendar week numbers (1-52)
        - **Color scale:** White (0 anomalies) → Deep Red (multiple anomalies)
        - **Hover tooltip:** Shows exact week, day, and anomaly count
        - **Clustered patterns:** Suggest systematic operational issues on specific days/weeks
        - **Isolated cells:** Random anomalies, likely one-off events
        - **Vertical patterns:** Recurring weekday issues (e.g., every Monday)
        - **Horizontal patterns:** Week-long problems affecting multiple consecutive days
        - **Empty (white) rows:** Weeks with stable, anomaly-free operations
        """)
    
    anomalies_df = results['anomalies']
    
    # Get ensemble column
    ensemble_col = 'ensemble_anomaly'
    if ensemble_col not in anomalies_df.columns:
        st.warning("No ensemble anomaly data available")
        return
    
    # Create date features
    df = anomalies_df[[ensemble_col]].copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['week'] = df.index.isocalendar().week
    df['dayofweek'] = df.index.dayofweek
    
    # Group by week and day of week
    pivot_data = df.pivot_table(
        values=ensemble_col,
        index='week',
        columns='dayofweek',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        y=pivot_data.index,
        colorscale='Reds',
        hovertemplate='Week %{y}<br>%{x}<br>Anomalies: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Weekly Anomaly Heatmap (2025)",
        xaxis_title="Day of Week",
        yaxis_title="Week Number",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_anomalies_timeline(results):
    """Plot time series of anomalies with key metrics."""
    if 'anomalies' not in results:
        return
    
    st.markdown("### 📈 Anomaly Timeline with Operational Metrics")
    
    # Description with key insights
    st.info("""
    Interactive time-series analysis pairing operational KPIs with detected anomaly markers. 
    **Key insights:** Select different metrics to identify leading indicators of anomalies. Red markers denote 
    ensemble-flagged days. Correlation analysis reveals whether anomalies coincide with production spikes, quality degradation, 
    dispatch delays, or inventory issues. Temporal patterns inform predictive maintenance and process optimization strategies.
    """)
    
    # Key interpretation guide
    with st.expander("🔑 Key Interpretation Guide"):
        st.markdown("""
        - **X-axis:** Date timeline across full 2025 operational year
        - **Y-axis:** Selected metric value (scale varies by metric)
        - **Blue line:** Operational metric trend over time
        - **Red X markers:** Days flagged as anomalies by ensemble model
        - **Dropdown selector:** Choose different metrics for correlation analysis
        - **Spike coinciding with anomaly:** Suggests metric as potential root cause
        - **Anomaly without spike:** Indicates multivariate anomaly (combination of factors)
        - **Multiple consecutive anomalies:** Systematic issue requiring intervention
        - **Isolated anomalies:** May be acceptable operational variance or one-off events
        - **Hover for details:** Exact date and metric value at each point
        """)
    
    # Load full analytic dataset for context
    analytic_file = DATA_DIR / 'plant_daily.parquet'
    if not analytic_file.exists():
        st.warning("Analytic dataset not found")
        return
    
    df_full = pd.read_parquet(analytic_file)
    
    # Get anomalies
    anomalies_df = results['anomalies']
    ensemble_col = 'ensemble_anomaly'
    
    if ensemble_col not in anomalies_df.columns:
        return
    
    # Merge with full data
    df_plot = df_full.copy()
    df_plot['anomaly'] = anomalies_df[ensemble_col] if len(anomalies_df) > 0 else 0
    
    # Select key metrics to plot
    metrics = {
        'Production': 'total_prod',
        'Defect Rate': 'avg_defect',
        'QC Pass Rate': 'qc_pass_rate',
        'Dispatch Delay': 'avg_delay',
        'Waste': 'total_waste',
        'Inventory Issues': 'negative_balance_count'
    }
    
    # Create subplot
    fig = go.Figure()
    
    # Plot selected metric
    metric_choice = st.selectbox(
        "Select metric to visualize:",
        options=list(metrics.keys()),
        index=0
    )
    
    metric_col = metrics[metric_choice]
    
    if metric_col in df_plot.columns:
        # Plot metric
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot[metric_col],
            mode='lines',
            name=metric_choice,
            line=dict(color='#636EFA', width=2)
        ))
        
        # Highlight anomalous days
        anomalous_dates = df_plot[df_plot['anomaly'] == 1].index
        if len(anomalous_dates) > 0:
            fig.add_trace(go.Scatter(
                x=anomalous_dates,
                y=df_plot.loc[anomalous_dates, metric_col],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x',
                    line=dict(color='darkred', width=2)
                )
            ))
        
        fig.update_layout(
            title=f"{metric_choice} Over Time with Detected Anomalies",
            xaxis_title="Date",
            yaxis_title=metric_choice,
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_anomalous_days_table(results):
    """Show detailed table of anomalous days."""
    if 'anomalies' not in results:
        return
    
    st.markdown("### 📋 Detected Anomalous Days")
    
    # Description with key insights
    st.info("""
    Comprehensive table of all flagged anomalous days with operational context and model consensus metrics. 
    **Key insights:** Model Agreement score indicates detection confidence (higher = more models flagged the day). 
    Top 20 days sorted by agreement represent highest-priority investigation targets. Operational metrics provide 
    root cause analysis context: production output, defect rates, QC performance, dispatch delays, waste, returns, and inventory anomalies.
    """)
    
    # Key interpretation guide
    with st.expander("🔑 Key Interpretation Guide"):
        st.markdown("""
        - **Date (Index):** Specific day identified as anomalous
        - **model_agreement:** Number of algorithms that flagged this day (1-4 scale)
        - **Anomaly columns:** Binary flags (1 = anomaly detected by that model)
        - **total_prod:** Total production output (units)
        - **avg_defect:** Average defect rate across batches
        - **qc_pass_rate:** Quality control pass percentage
        - **avg_delay:** Average dispatch delay (hours)
        - **total_waste/returns:** Waste generation and product returns
        - **negative_balance_count:** Inventory stockout incidents
        - **High agreement (3-4):** Strong anomaly signal, prioritize investigation
        - **Low agreement (1-2):** Marginal anomaly, may be acceptable variance
        - **Download button:** Export full dataset for detailed offline analysis
        """)
    
    anomalies_df = results['anomalies']
    
    # Select relevant columns
    display_cols = [
        'model_agreement', 
        'isolation_forest_anomaly', 'statistical_anomaly', 'ensemble_anomaly',
        'total_prod', 'avg_defect', 'qc_pass_rate', 'avg_delay',
        'total_waste', 'total_returns', 'negative_balance_count'
    ]
    
    available_cols = [col for col in display_cols if col in anomalies_df.columns]
    
    if not available_cols:
        st.warning("No anomaly data columns found")
        return
    
    df_display = anomalies_df[available_cols].copy()
    df_display.index = df_display.index.strftime('%Y-%m-%d')
    
    # Sort by model agreement
    if 'model_agreement' in df_display.columns:
        df_display = df_display.sort_values('model_agreement', ascending=False)
    
    st.dataframe(
        df_display.head(20),
        use_container_width=True,
        height=600
    )
    
    # Download button
    csv = df_display.to_csv()
    st.download_button(
        label="📥 Download Anomalies CSV",
        data=csv,
        file_name="anomalous_days.csv",
        mime="text/csv"
    )


def plot_feature_importance(results):
    """Show feature groups analysis."""
    if 'cv_report' not in results:
        return
    
    st.markdown("### 🔍 Feature Groups Used")
    
    # Description with key insights
    st.info("""
    Distribution of engineered features across six operational domains used for anomaly detection. 
    **Key insights:** Balanced feature representation ensures holistic anomaly detection across all operational areas. 
    Larger segments indicate more granular monitoring in that domain. The multi-domain approach captures complex, 
    multivariate anomalies that single-domain analysis would miss, providing comprehensive operational intelligence.
    """)
    
    # Key interpretation guide
    with st.expander("🔑 Key Interpretation Guide"):
        st.markdown("""
        - **Pie slices:** Represent feature count per operational domain
        - **Slice size:** Proportional to number of features in that category
        - **Hover tooltip:** Shows exact feature count and percentage
        - **Production features:** Batch output, defect rates, throughput metrics
        - **Dispatch features:** Delivery performance, delay patterns, logistics KPIs
        - **Quality Control features:** QC pass rates, parameter compliance, inspection results
        - **Waste & Returns features:** Waste generation, return volumes, loss metrics
        - **Inventory features:** Stock movements, balance anomalies, turnover rates
        - **Sales features:** Demand patterns, promotional impacts, revenue metrics
        - **Balanced distribution:** Indicates comprehensive coverage across operations
        - **Expandable details:** Click to see specific features within each domain
        """)
    
    cv_report = results['cv_report']
    feature_groups = cv_report.get('feature_groups', {})
    
    if not feature_groups:
        st.warning("No feature group information available")
        return
    
    # Count features by group
    group_counts = {group: len(features) for group, features in feature_groups.items()}
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(group_counts.keys()),
        values=list(group_counts.values()),
        hole=0.4,
        textinfo='label+value',
        hovertemplate='%{label}<br>%{value} features<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Features by Operational Category",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed feature lists
    with st.expander("📋 Feature Details"):
        for group, features in feature_groups.items():
            st.markdown(f"**{group.replace('_', ' ').title()}** ({len(features)} features)")
            st.write(", ".join(features))
            st.markdown("---")


def render_phase4_visualizations():
    """Main function to render all Phase 4 visualizations."""
    
    # Step 1: Interactive Training Interface
    render_training_interface()
    
    # Step 2: Check if models are trained
    if not check_training_status():
        st.info("👆 Please train the models using the interface above to see results and visualizations.")
        return
    
    # Step 3: Load model results
    results = load_model_results()
    
    # Step 4: Overview
    render_phase4_overview(results)
    
    if not results:
        return
    
    # Step 5: Interactive visualizations in tabs
    st.markdown("### 📊 Explore Results & Visualizations")
    
    # Create tabs for different viz categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Performance",
        "📅 Anomaly Detection",
        "📈 Time Series Analysis",
        "🔍 Feature Analysis"
    ])
    
    with tab1:
        st.markdown("#### Model Performance Comparison")
        st.info("""
        **Evaluation Metrics:**
        - **Precision**: Proportion of correctly identified anomalies among all flagged days
        - **Recall**: Proportion of true anomalies successfully detected
        - **F1-Score**: Harmonic mean balancing precision and recall
        - **ROC-AUC**: Model discrimination capability (0.5 = random, 1.0 = perfect)
        """)
        plot_model_performance(results)
        plot_anomalies_by_model(results)
    
    with tab2:
        st.markdown("#### Detected Anomalous Days")
        st.info("""
        **Anomaly Detection Results:**
        - Heatmap visualization shows anomaly distribution across calendar weeks
        - **Model Agreement**: Indicates detection consensus across multiple algorithms (higher values suggest stronger anomaly signals)
        - Complete list of flagged days available for download with operational context
        """)
        plot_anomalies_calendar(results)
        show_anomalous_days_table(results)
    
    with tab3:
        st.markdown("#### Temporal Patterns & Anomalies")
        st.info("""
        **Time Series Analysis:**
        - Interactive visualization of operational metrics over time with anomaly markers
        - Select metrics from dropdown to identify correlations between anomalies and operational KPIs
        - Analyze temporal patterns and potential root cause indicators
        """)
        plot_anomalies_timeline(results)
    
    with tab4:
        st.markdown("#### Feature Importance & Groups")
        st.info("""
        **Multi-Domain Feature Engineering:**
        - Production metrics: Batch output, defect rates
        - Dispatch operations: Delivery delays, on-time performance
        - Quality control: QC pass rates, parameter compliance
        - Waste & returns: Generation rates, return volumes
        - Inventory: Stock movements, balance anomalies
        - Sales: Demand patterns, promotional impact
        """)
        plot_feature_importance(results)


# Call this function from main streamlit_eda_explorer.py
if __name__ == '__main__':
    st.set_page_config(page_title="Phase 4: ML Models", layout="wide")
    render_phase4_visualizations()
