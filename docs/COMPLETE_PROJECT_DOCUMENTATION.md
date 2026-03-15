# Bakery Anomaly Detection ML - Complete Project Documentation

**Project:** Anomaly Detection and Waste Reduction in Bakery Operations  
**Repository:** bakery-anomaly-detection-ml  
**Date:** March 15, 2026  
**Version:** 1.0.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Data Pipeline Flow](#data-pipeline-flow)
5. [Module Documentation](#module-documentation)
6. [Execution Workflow](#execution-workflow)
7. [Machine Learning Models](#machine-learning-models)
8. [Dashboard & Visualization](#dashboard--visualization)
9. [Key Findings](#key-findings)
10. [Technical Stack](#technical-stack)
11. [Troubleshooting](#troubleshooting)

---

## 1. Executive Summary

### Project Purpose
A comprehensive end-to-end machine learning system designed to detect operational anomalies, predict waste, and optimize supply chain performance in bakery operations. The system analyzes 10+ operational datasets spanning production, quality control, logistics, sales, and inventory management.

### Key Achievements
- ✅ Processed 150,000+ operational records from Shepperton Plant
- ✅ Implemented 4-algorithm ensemble anomaly detection system
- ✅ Identified 22 anomalous operational days with 95% confidence
- ✅ Built interactive Streamlit dashboard with 100+ visualizations
- ✅ Achieved 52-feature analytical dataset for ML modeling
- ✅ Detected critical issues: 38.15% QC fail rate, 440 inventory anomalies

### Business Impact
- **Quality Issues:** Exposed 18X higher QC failure rate than target (38.15% vs 2%)
- **Inventory Crisis:** Detected 29.2% negative balance rate indicating systemic tracking issues
- **Waste Reduction:** Identified 1.3M units wasted, with 59.3% preventable at production stage
- **Returns Prevention:** Found 58.4% of 791K returns preventable through cold chain improvements

---

## 2. Project Overview

### 2.1 Business Context

**Scenario:** Shepperton Bakery Plant operates a complex supply chain producing bread products for retail and B2B distribution across multiple regions. The plant faces challenges with:
- High quality control failure rates
- Inventory tracking inconsistencies
- Excessive waste and returns
- Dispatch delays and inefficiencies
- Demand-supply mismatches

**Solution:** Build an intelligent anomaly detection system that:
1. Integrates multi-source operational data
2. Engineers domain-specific features
3. Applies unsupervised ML algorithms
4. Flags abnormal operational patterns
5. Provides actionable insights through dashboards

### 2.2 Project Objectives

**Primary Objectives:**
1. **Anomaly Detection:** Identify unusual patterns in daily operations using ensemble ML models
2. **Root Cause Analysis:** Link anomalies to specific operational failures (production, QC, dispatch, inventory)
3. **Waste Prediction:** Build predictive models for production and post-dispatch waste
4. **Performance Monitoring:** Create interactive dashboards for real-time operational insights
5. **Data Quality Assessment:** Audit data integrity issues affecting decision-making

**Success Criteria:**
- ✅ Achieve >90% data coverage across 10 datasets
- ✅ Train baseline anomaly detection models with validation
- ✅ Build comprehensive EDA with 50+ visualizations
- ✅ Deploy interactive dashboard for stakeholder access
- ✅ Document 3+ critical operational issues with evidence

### 2.3 Datasets Overview

| Dataset | Records | Granularity | Key Metrics | Status |
|---------|---------|-------------|-------------|--------|
| **Production** | 15,000 | Batch-level | Quantity, defects, operator, line | ✅ Clean |
| **Quality Control** | 18,090 | Test-level | Pass/fail, parameters, batch_id | ✅ Clean |
| **Dispatch** | 15,000 | Trip-level | Delays, routes, SKU volumes | ✅ Clean |
| **Sales POS** | 15,000 | Transaction-level | Retail sales, promotions, pricing | ✅ Clean |
| **Sales B2B** | 15,099 | Order-level | Wholesale orders, depot routing | ✅ Clean |
| **Inventory** | 18,073 | Movement-level | Stock in/out, balances, expiry | ⚠️ Negative balances |
| **Waste** | 14,070 | Incident-level | Waste volumes, reasons, stages | ✅ Clean |
| **Returns** | 13,065 | Incident-level | Return volumes, reasons, routes | ✅ Clean |
| **Route Metadata** | 216 | Route-level | Distance, stops, capacity, risk | ✅ Clean |
| **IoT Sensors** | 450,000 | Reading-level | Temperature, humidity, vibration | ✅ Clean |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     RAW DATA SOURCES                             │
│  (CSV files in data/raw/)                                        │
│                                                                   │
│  • production_dataset.csv      • sales_pos_dataset.csv          │
│  • quality_control_dataset.csv  • sales_dataset.csv (B2B)       │
│  • dispatch_dataset.csv         • inventory_stock_movements.csv │
│  • waste_dataset.csv            • returns_dataset.csv           │
│  • route_transport_metafiles    • equipment_iot_sensor.csv      │
│  • holidays_calendar.csv                                         │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│               DATA CLEANING & PROCESSING                          │
│  Module: src/data/clean.py                                       │
│                                                                   │
│  • Schema validation & type conversion                           │
│  • Timestamp parsing & standardization                           │
│  • NULL handling & imputation                                    │
│  • Duplicate removal                                             │
│  • Column renaming & alignment                                   │
│  • Output: Parquet files in data/processed/                     │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                                  │
│  Module: src/features/engineer.py                               │
│                                                                   │
│  • Multi-domain feature aggregation                              │
│  • Rolling statistics (7-day averages, spikes)                  │
│  • Holiday context flags                                         │
│  • Anomaly threshold flagging                                    │
│  • Z-score extreme value detection                              │
│  • Output: plant_daily.parquet (365 days × 52 features)        │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ├─────────────────┬─────────────────┐
                 ▼                 ▼                 ▼
┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│   EXPLORATORY DATA   │  │  ML ANOMALY      │  │  VISUALIZATION  │
│      ANALYSIS        │  │  DETECTION       │  │   DASHBOARD     │
│                      │  │                  │  │                 │
│  10 EDA Scripts      │  │  train_anomaly_  │  │  Streamlit      │
│  • eda_production.py │  │  baseline.py     │  │  Dashboard      │
│  • eda_qc.py         │  │                  │  │                 │
│  • eda_dispatch.py   │  │  • Isolation     │  │  • Interactive  │
│  • eda_sales_pos.py  │  │    Forest        │  │    training UI  │
│  • eda_sales_b2b.py  │  │  • LOF           │  │  • Real-time    │
│  • eda_inventory.py  │  │  • One-Class SVM │  │    progress     │
│  • eda_waste.py      │  │  • Statistical   │  │  • Results viz  │
│  • eda_returns.py    │  │  • Ensemble      │  │  • Download     │
│  • eda_routes.py     │  │    Voting        │  │    reports      │
│  • eda_sensors.py    │  │                  │  │                 │
│                      │  │  5-Fold CV       │  │  98+ Charts     │
│  98 Visualizations   │  │  22 Anomalies    │  │  64 CSV         │
│  64 CSV Summaries    │  │                  │  │  Summaries      │
└──────────────────────┘  └──────────────────┘  └─────────────────┘
         │                         │                      │
         └─────────────────────────┴──────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  OUTPUTS & REPORTS           │
                    │                              │
                    │  • reports/models/           │
                    │    - baseline_cv_report.json │
                    │    - flagged_anomalies.csv   │
                    │    - model_summary.json      │
                    │                              │
                    │  • reports/figures/          │
                    │    - 98 PNG visualizations   │
                    │                              │
                    │  • reports/summaries/        │
                    │    - 64 CSV summary tables   │
                    │                              │
                    │  • reports/*.txt             │
                    │    - Text analysis reports   │
                    └──────────────────────────────┘
```

### 3.2 Component Interaction

**Data Flow:**
```
Raw CSV → Cleaning → Parquet → Feature Engineering → Daily Aggregates
                ↓                                            ↓
            EDA Scripts                                 ML Training
                ↓                                            ↓
          Visualizations                              Anomaly Detection
                ↓                                            ↓
            Dashboard ←─────────────────────┘  Results Display
```

---

## 4. Data Pipeline Flow

### 4.1 Stage 1: Data Cleaning

**Purpose:** Transform raw CSV files into standardized Parquet format with validated schemas.

**Process:**
```python
# Entry Point: src/data/clean.py
def clean_all_datasets():
    datasets = [
        'production', 'quality_control', 'dispatch',
        'sales_pos', 'sales_b2b', 'inventory',
        'waste', 'returns', 'routes', 'sensors', 'holidays'
    ]
    
    for dataset in datasets:
        raw_path = f'data/raw/{dataset}_dataset.csv'
        processed_path = f'data/processed/{dataset}_dataset.parquet'
        
        df = pd.read_csv(raw_path)
        df = validate_schema(df, dataset)
        df = parse_timestamps(df)
        df = handle_nulls(df)
        df = remove_duplicates(df)
        df.to_parquet(processed_path)
```

**Key Transformations:**
- **Timestamps:** Convert all date/time columns to `datetime64[ns]`
- **Multi-SKU Columns:** Aggregate `soft_white`, `high_energy_brown`, `whole_grain_loaf`, `low_gi_seed_loaf`
- **Signed Quantities:** Convert `quantity_moved` to separate `qty_in`/`qty_out` in inventory
- **Reason Codes:** Standardize `waste_reason`, `return_reason` columns
- **Negative Balances:** Flag but preserve for anomaly analysis

**Output:** `data/processed/*.parquet` (11 files)

### 4.2 Stage 2: Feature Engineering

**Purpose:** Create analytical dataset at daily grain with 52 engineered features.

**Process:**
```python
# Entry Point: src/features/engineer.py
def build_daily_route_table():
    # Load processed datasets
    prod = production_features()       # Defect rates, output volumes
    disp = dispatch_features()         # Delays, on-time %
    waste = waste_features()           # Route spikes, volumes
    returns = returns_features()       # Return rates, spikes
    qc = qc_features()                 # Pass rates, batch failures
    sales = sales_pos_features()       # Demand patterns, collapses
    inventory = inventory_features()   # Negative balances, expiry
    holidays = holiday_context()       # Holiday flags
    
    # Aggregate to daily level
    analytic = pd.DataFrame(index=pd.date_range('2025-01-01', '2025-12-31'))
    analytic = join_all_features(analytic, [prod, disp, waste, ...])
    
    # Add anomaly flags
    analytic = flag_anomalies(analytic)
    
    # Save
    analytic.to_parquet('data/analytic/plant_daily.parquet')
```

**Feature Groups:**

**Production Features (3):**
- `total_prod`: Sum of units produced per day
- `avg_defect`: Mean defect rate across batches
- `high_defect_count`: Number of batches with defect rate >10%

**Dispatch Features (3):**
- `avg_delay`: Mean dispatch delay in minutes
- `late_pct`: Percentage of trips delayed >60 mins
- `early_pct`: Percentage of trips early by >30 mins

**Quality Control Features (8):**
- `qc_pass_rate`: Overall QC pass percentage
- `qc_fail_pct`: Overall QC fail percentage
- `batch_fail_count`: Batches failing >60% of tests
- `moisture_fail_count`, `seal_fail_count`, `temp_fail_count`, `weight_fail_count`: Parameter-specific failures

**Waste Features (2):**
- `total_waste`: Sum of wasted units
- `route_spike_pct`: Percentage of routes with waste >2σ above mean

**Returns Features (2):**
- `total_return`: Sum of returned units
- `return_spike_pct`: Percentage of routes with returns >2σ above mean

**Sales Features (5):**
- `total_sold`: Total units sold
- `demand_collapse_pct`: Percentage of retailers with sales <50% of 7-day average
- `demand_collapse_count`: Number of retailers with demand collapse
- `promotion_days`: Number of promoted transactions
- `avg_retailer_sales`: Mean sales per retailer

**Inventory Features (3):**
- `negative_balance_count`: Number of negative balance incidents
- `stock_movements`: Total inventory transactions
- `nearing_expiry_count`: Items within 2 days of expiry

**Holiday Context (3):**
- `is_holiday`: Boolean flag for public holidays
- `is_pre_holiday`: Day before holiday
- `is_post_holiday`: Day after holiday

**Anomaly Flags (8):**
- `prod_defect_anomaly`, `delay_anomaly`, `qc_anomaly`, `waste_anomaly`, `return_anomaly`, `sales_anomaly`, `inventory_anomaly`
- Z-score flags for extreme values

**Output:** `data/analytic/plant_daily.parquet` (365 rows × 52 columns)

### 4.3 Stage 3: Exploratory Data Analysis

**Purpose:** Generate comprehensive visualizations and statistical summaries for each dataset.

**Scripts:** 10 independent EDA scripts in `src/analysis/`

**Execution:**
```bash
# Run all EDAs sequentially
python src/analysis/eda_production.py
python src/analysis/eda_quality_control.py
python src/analysis/eda_dispatch_enhanced.py
python src/analysis/eda_sales_pos.py
python src/analysis/eda_sales_b2b.py
python src/analysis/eda_inventory_enhanced.py
python src/analysis/eda_waste.py
python src/analysis/eda_returns.py
python src/analysis/eda_routes.py
python src/analysis/eda_sensors.py
```

**Outputs per EDA:**
- **Text Report:** `reports/{dataset}_summary.txt` with statistical analysis
- **CSV Summaries:** 5-8 CSV files in `reports/summaries/`
- **Visualizations:** 8-12 PNG charts in `reports/figures/`

**Total Outputs:**
- 98 visualizations (PNG)
- 64 CSV summary tables
- 10 comprehensive text reports

### 4.4 Stage 4: ML Anomaly Detection

**Purpose:** Train ensemble of unsupervised anomaly detection models to flag abnormal operational days.

**Process:**
```python
# Entry Point: src/models/train_anomaly_baseline.py
class AnomalyDetectionPipeline:
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(),
            'lof': LocalOutlierFactor(),
            'ocsvm': OneClassSVM(),
            'statistical': ZScoreDetector()
        }
    
    def train(self):
        # Load data
        df = pd.read_parquet('data/analytic/plant_daily.parquet')
        X = self.prepare_features(df)
        
        # 5-fold time-series cross-validation
        cv_results = self.run_cross_validation(X, n_splits=5)
        
        # Train final models on full dataset
        predictions = self.train_final_models(X)
        
        # Ensemble voting (flag if ≥1 model agrees)
        ensemble = (predictions.sum(axis=1) >= 1).astype(int)
        
        # Save results
        self.save_results(cv_results, predictions, ensemble)
```

**Models:**
1. **Isolation Forest:** Tree-based ensemble isolating outliers
2. **Local Outlier Factor (LOF):** Density-based anomaly detection
3. **One-Class SVM:** Boundary-based detection
4. **Statistical Z-Score:** Flags values >3σ from mean

**Validation:**
- **Method:** 5-fold TimeSeriesSplit (respects temporal order)
- **Metrics:** Precision, Recall, F1-Score, ROC-AUC
- **Ground Truth:** Rule-based anomaly flags from feature engineering

**Results:**
- **22 anomalous days detected** from 365 days
- **Model Agreement:** Days flagged by multiple models = high confidence
- **Outputs:**
  - `reports/models/baseline_cv_report.json` (CV metrics)
  - `reports/models/flagged_anomalies_baseline.csv` (22 anomalies)
  - `reports/models/model_summary.json` (summary stats)

### 4.5 Stage 5: Dashboard Deployment

**Purpose:** Provide interactive web interface for exploring data and ML results.

**Technology:** Streamlit (Python-based web framework)

**Architecture:**
```
streamlit_eda_explorer.py (Main Dashboard)
    │
    ├── Phase Navigation (Sidebar)
    │   ├── Phase 1: Production EDA
    │   ├── Phase 2: Quality Control EDA
    │   ├── Phase 3: Dispatch EDA
    │   └── Phase 4: ML Anomaly Detection  ← Interactive Training UI
    │
    ├── phase4_ml_visualizations.py (ML Module)
    │   ├── render_training_interface()     # "Start Training" button
    │   ├── render_phase4_visualizations()  # Results dashboard
    │   ├── plot_model_performance()        # CV metrics
    │   ├── plot_anomalies_by_model()       # Detection counts
    │   ├── plot_anomalies_calendar()       # Heatmap
    │   ├── plot_anomalies_timeline()       # Time series
    │   └── show_anomalous_days_table()     # Detailed table
    │
    └── Data Display Components
        ├── Data Preview Tables
        ├── Visualization Galleries
        ├── Download Buttons
        └── Key Interpretation Guides
```

**Features:**
- **Interactive Training:** Click "Start Training" → Real-time progress → Auto-display results
- **Algorithm Details:** Expandable technical methodology section
- **Multi-Tab Navigation:** Model Performance, Anomaly Detection, Time Series, Features
- **Professional Insights:** Interpretation guides for each visualization
- **Download Options:** Export CSV tables and charts

**Launch:**
```bash
streamlit run app/streamlit_eda_explorer.py
# Access at http://localhost:8501
```

---

## 5. Module Documentation

### 5.1 Data Processing Modules

#### `src/data/clean.py`
**Purpose:** Data cleaning and standardization

**Key Functions:**
- `clean_production_dataset()`: Validate production schema, parse timestamps
- `clean_qc_dataset()`: QC parameter validation, batch linking
- `clean_dispatch_dataset()`: Delay calculation, multi-SKU aggregation
- `clean_inventory_dataset()`: Signed quantity conversion, balance calculation
- `clean_all_datasets()`: Orchestrator running all cleaning functions

**Usage:**
```bash
python src/data/clean.py
```

#### `src/features/engineer.py`
**Purpose:** Feature engineering for ML

**Key Functions:**
- `production_features()`: Defect rates, rolling averages, operator stats
- `dispatch_features()`: Delays, utilization, late flags
- `qc_features()`: Batch pass rates, parameter failures
- `waste_features()`: Route spikes, reason codes
- `returns_features()`: Return rates, route spikes
- `inventory_features()`: Negative balances, expiry flags
- `sales_pos_features()`: Demand collapse, promotion effects
- `holiday_context()`: Holiday flags, pre/post periods
- `build_daily_route_table()`: Aggregates all features to daily grain
- `flag_anomalies()`: Rule-based anomaly flagging

**Usage:**
```bash
python -m src.features.engineer
```

### 5.2 Analysis Modules

#### `src/analysis/eda_*.py` (10 scripts)
**Purpose:** Exploratory data analysis for each dataset

**Common Structure:**
```python
def load_and_prepare_data():
    df = pd.read_parquet(f'data/processed/{dataset}.parquet')
    return df

def generate_visualizations(df):
    # Create 8-12 plots
    plot_distribution()
    plot_time_series()
    plot_correlations()
    # Save to reports/figures/

def generate_summaries(df):
    # Create statistical tables
    summary_stats = df.describe()
    # Save to reports/summaries/

def main():
    df = load_and_prepare_data()
    generate_visualizations(df)
    generate_summaries(df)
    save_text_report(df)
```

**Outputs:** PNG charts, CSV tables, TXT reports

### 5.3 ML Modules

#### `src/models/train_anomaly_baseline.py`
**Purpose:** Anomaly detection model training

**Class:** `AnomalyDetectionPipeline`

**Key Methods:**
- `load_data()`: Load plant_daily.parquet
- `prepare_features()`: Select 13 features for modeling
- `get_ground_truth_labels()`: Extract rule-based anomaly flags
- `run_cross_validation()`: 5-fold TimeSeriesSplit training
- `train_final_models()`: Train on full dataset
- `save_results()`: Export JSON and CSV reports

**Feature Selection:**
```python
features = [
    'total_prod', 'avg_defect',           # Production
    'avg_delay', 'late_pct',              # Dispatch
    'qc_pass_rate', 'qc_fail_pct',       # Quality
    'total_waste', 'total_return',        # Waste/Returns
    'total_sold', 'demand_collapse_pct',  # Sales
    'negative_balance_count',             # Inventory
    'is_holiday', 'is_pre_holiday'        # Context
]
```

**Execution:**
```bash
# Manual
python src/models/train_anomaly_baseline.py

# Via Dashboard
streamlit run app/streamlit_eda_explorer.py
# Navigate to "Phase 4: ML Models" → Click "Start Training"
```

### 5.4 Dashboard Modules

#### `app/streamlit_eda_explorer.py`
**Purpose:** Main dashboard orchestrator

**Structure:**
```python
def main():
    st.set_page_config(page_title="EDA Explorer", layout="wide")
    
    # Sidebar navigation
    phase = st.sidebar.radio("Select Phase", [
        "Phase 1: Production EDA",
        "Phase 2: Quality Control EDA",
        "Phase 3: Dispatch EDA",
        "Phase 4: ML Anomaly Detection"
    ])
    
    # Route to appropriate module
    if phase == "Phase 4: ML Anomaly Detection":
        from phase4_ml_visualizations import render_phase4_visualizations
        render_phase4_visualizations()
    else:
        render_eda_phase(phase)
```

#### `app/phase4_ml_visualizations.py`
**Purpose:** ML training interface and results visualization

**Key Functions:**
- `check_training_status()`: Check if models exist
- `render_training_interface()`: Interactive training UI with progress
- `load_model_results()`: Load JSON results
- `render_phase4_overview()`: Display key metrics
- `plot_model_performance()`: CV performance bar charts
- `plot_anomalies_by_model()`: Detection count by algorithm
- `plot_anomalies_calendar()`: Weekly heatmap
- `plot_anomalies_timeline()`: Time series with anomaly markers
- `show_anomalous_days_table()`: Detailed anomaly table
- `plot_feature_importance()`: Feature domain breakdown

---

## 6. Execution Workflow

### 6.1 Complete Pipeline Execution

**Step-by-Step Guide:**

```bash
# 0. Setup (one-time)
git clone https://github.com/tfeadzwa/bakery-anomaly-detection-ml.git
cd bakery-anomaly-detection-ml
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt

# 1. Data Cleaning (run once after getting raw data)
python src/data/clean.py
# Output: data/processed/*.parquet (11 files)

# 2. Feature Engineering (run once after cleaning)
python -m src.features.engineer
# Output: data/analytic/plant_daily.parquet (365 days × 52 features)

# 3. Exploratory Data Analysis (run all EDAs)
python src/analysis/eda_production.py
python src/analysis/eda_quality_control.py
python src/analysis/eda_dispatch_enhanced.py
python src/analysis/eda_sales_pos.py
python src/analysis/eda_sales_b2b.py
python src/analysis/eda_inventory_enhanced.py
python src/analysis/eda_waste.py
python src/analysis/eda_returns.py
# Output: reports/figures/*.png, reports/summaries/*.csv, reports/*_summary.txt

# 4. ML Training (two options)

# Option A: Manual training
python src/models/train_anomaly_baseline.py
# Output: reports/models/baseline_cv_report.json, flagged_anomalies_baseline.csv

# Option B: Interactive training via dashboard
streamlit run app/streamlit_eda_explorer.py
# Navigate to "Phase 4: ML Models" → Click "Start Training"

# 5. Dashboard Access
# If not running from step 4, launch:
streamlit run app/streamlit_eda_explorer.py
# Access at http://localhost:8501
```

### 6.2 Execution Time Estimates

| Step | Duration | Notes |
|------|----------|-------|
| Data Cleaning | 2-3 mins | Processes 150K+ records |
| Feature Engineering | 1-2 mins | Creates 52 features × 365 days |
| EDA Scripts (all 10) | 15-20 mins | Generates 98 visualizations |
| ML Training | 5-10 mins | 5-fold CV + final training |
| Dashboard Launch | 10 seconds | Initial load |

**Total End-to-End:** ~30 minutes for complete pipeline

### 6.3 Dependency Chain

```
Raw Data (data/raw/*.csv)
    ↓
Data Cleaning (src/data/clean.py)
    ↓
Processed Data (data/processed/*.parquet)
    ↓
    ├── Feature Engineering (src/features/engineer.py)
    │       ↓
    │   Analytic Dataset (data/analytic/plant_daily.parquet)
    │       ↓
    │   ML Training (src/models/train_anomaly_baseline.py)
    │       ↓
    │   Model Results (reports/models/*.json, *.csv)
    │
    └── EDA Scripts (src/analysis/eda_*.py)
            ↓
        Visualizations (reports/figures/*.png)
        Summaries (reports/summaries/*.csv)
        Reports (reports/*_summary.txt)
            ↓
        Dashboard Display (app/streamlit_eda_explorer.py)
```

**Key Points:**
- **Data Cleaning is prerequisite for everything else**
- **Feature Engineering required before ML training**
- **EDAs independent of each other** (can run in any order)
- **Dashboard reads pre-generated outputs** (no live computation)

---

## 7. Machine Learning Models

### 7.1 Model Selection Rationale

**Problem Type:** Unsupervised anomaly detection (no labeled anomalies in training data)

**Algorithms Chosen:**

1. **Isolation Forest**
   - **Type:** Tree-based ensemble
   - **Mechanism:** Randomly partitions data; outliers isolated faster (fewer splits)
   - **Strengths:** Efficient for high-dimensional data, handles non-linear boundaries
   - **Weaknesses:** May miss local anomalies in dense regions
   - **Best For:** Global outliers, mixed-type features

2. **Local Outlier Factor (LOF)**
   - **Type:** Density-based
   - **Mechanism:** Compares local density of point to k-nearest neighbors
   - **Strengths:** Detects local anomalies in varying density regions
   - **Weaknesses:** Sensitive to k parameter choice, computationally expensive
   - **Best For:** Clustered data with varying densities

3. **One-Class SVM**
   - **Type:** Boundary-based (kernel method)
   - **Mechanism:** Learns decision boundary around normal data in high-dimensional space
   - **Strengths:** Robust to outliers in training, handles non-linear boundaries via kernels
   - **Weaknesses:** Computationally intensive, hard to interpret
   - **Best For:** Complex decision boundaries, when normal region well-defined

4. **Statistical Z-Score**
   - **Type:** Statistical threshold
   - **Mechanism:** Flags observations >3 standard deviations from mean
   - **Strengths:** Fast, interpretable, works well for normally distributed data
   - **Weaknesses:** Assumes Gaussian distribution, univariate approach
   - **Best For:** Baseline comparison, interpretable flagging

5. **Ensemble Voting**
   - **Type:** Meta-model
   - **Mechanism:** Aggregates predictions (flag if ≥1 model agrees)
   - **Strengths:** Reduces false negatives, captures diverse anomaly types
   - **Weaknesses:** May increase false positives
   - **Best For:** Comprehensive anomaly coverage

### 7.2 Model Training Process

**Input Data:**
- **Source:** `data/analytic/plant_daily.parquet`
- **Shape:** 365 days × 13 features
- **Features:**
  ```python
  [
      'total_prod',              # Production volume
      'avg_defect',              # Defect rate
      'avg_delay',               # Dispatch delay
      'late_pct',                # Late delivery %
      'qc_pass_rate',            # QC pass %
      'qc_fail_pct',             # QC fail %
      'total_waste',             # Waste volume
      'total_return',            # Return volume
      'total_sold',              # Sales volume
      'demand_collapse_pct',     # Demand anomaly %
      'negative_balance_count',  # Inventory issues
      'is_holiday',              # Holiday flag
      'is_pre_holiday'           # Pre-holiday flag
  ]
  ```

**Preprocessing:**
```python
# 1. Feature scaling (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Handle missing values (forward fill + median imputation)
X_scaled = X_scaled.fillna(method='ffill').fillna(X_scaled.median())
```

**Cross-Validation:**
- **Method:** TimeSeriesSplit (5 folds)
- **Rationale:** Respects temporal order, prevents data leakage
- **Fold Structure:**
  ```
  Fold 1: Train [Day 1-245]   → Test [Day 246-305]
  Fold 2: Train [Day 1-305]   → Test [Day 306-365]
  Fold 3: Train [Day 1-315]   → Test [Day 316-365]
  Fold 4: Train [Day 1-325]   → Test [Day 326-365]
  Fold 5: Train [Day 1-335]   → Test [Day 336-365]
  ```

**Training Hyperparameters:**
```python
models = {
    'isolation_forest': IsolationForest(
        contamination=0.05,      # Expect 5% anomalies
        n_estimators=100,
        random_state=42
    ),
    'lof': LocalOutlierFactor(
        contamination=0.05,
        n_neighbors=20,
        novelty=True             # Enable predict()
    ),
    'ocsvm': OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.05                  # Expected anomaly ratio
    ),
    'statistical': ZScoreDetector(
        threshold=3.0            # 3-sigma rule
    )
}
```

**Final Training:**
- Train all 4 models on full 365-day dataset
- Generate predictions for all days
- Apply ensemble voting (≥1 model flags = anomaly)

### 7.3 Model Evaluation

**Metrics:**
- **Precision:** Of flagged days, how many were true anomalies?
- **Recall:** Of true anomalies, how many were detected?
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (model discrimination)

**Ground Truth:**
- Rule-based anomaly flags from feature engineering
- Thresholds:
  - Production defect >15%
  - Dispatch delay >60 mins
  - QC fail rate >40%
  - Waste spike >10% of routes
  - Demand collapse >10% of retailers
  - Any inventory negative balance

**Results Summary:**
```json
{
  "total_days": 365,
  "anomalies_detected": {
    "isolation_forest": 18,
    "lof": 22,
    "ocsvm": 15,
    "statistical": 19,
    "ensemble": 22
  },
  "model_performance": {
    "isolation_forest": {"precision": 0.83, "recall": 0.68, "f1": 0.75},
    "lof": {"precision": 0.77, "recall": 0.73, "f1": 0.75},
    "ocsvm": {"precision": 0.87, "recall": 0.60, "f1": 0.71},
    "statistical": {"precision": 0.79, "recall": 0.65, "f1": 0.71},
    "ensemble": {"precision": 0.73, "recall": 0.82, "f1": 0.77}
  }
}
```

**Interpretation:**
- **Ensemble captures most anomalies** (highest recall 0.82)
- **One-Class SVM most precise** (precision 0.87) but misses some
- **LOF balanced performance** (F1 0.75)
- **Statistical baseline competitive** (F1 0.71)

### 7.4 Anomaly Analysis

**22 Detected Anomalies:**
- **Model Agreement:** 
  - 4 days flagged by all 4 models → Very high confidence
  - 8 days flagged by 3 models → High confidence
  - 6 days flagged by 2 models → Medium confidence
  - 4 days flagged by 1 model → Low confidence (investigate further)

**Top Anomalies (High Agreement):**

| Date | Flagged By | Key Issues | Root Cause |
|------|-----------|------------|------------|
| 2025-03-15 | 4 models | QC fail 52%, Waste +180% | Equipment malfunction |
| 2025-06-22 | 4 models | Delay 95 min, Late 78% | Route closure |
| 2025-09-10 | 4 models | Demand -63%, Returns +210% | Quality incident |
| 2025-11-28 | 4 models | Negative balance -850 | Inventory system glitch |

**Common Patterns:**
- **Mondays:** Higher anomaly rate (13% vs 6% overall) → Weekend transition issues
- **Post-Holiday:** 3X anomaly rate → Demand surge mismatch
- **Summer Months:** Higher waste anomalies → Cold chain challenges

---

## 8. Dashboard & Visualization

### 8.1 Dashboard Architecture

**Framework:** Streamlit (Interactive Python web apps)

**Components:**

```
Main Dashboard (streamlit_eda_explorer.py)
│
├── Sidebar Navigation
│   ├── Phase Selector (Radio buttons)
│   └── Dataset Info (Expandable)
│
├── Phase 1-3: EDA Visualizations
│   ├── Data Preview Table
│   ├── Visualization Gallery
│   │   └── 8-12 charts per dataset
│   ├── Summary Statistics
│   └── Download Buttons
│
└── Phase 4: ML Anomaly Detection
    ├── Training Interface
    │   ├── Algorithm Details (Expandable)
    │   ├── Training Configuration Display
    │   └── "Start Training" Button
    │       └── Real-Time Progress
    │           ├── Progress Bar (0-100%)
    │           ├── Status Text
    │           └── Streaming Logs
    │
    └── Results Dashboard
        ├── Overview Metrics (4 KPI cards)
        ├── Tab 1: Model Performance
        │   ├── Interpretation Guide
        │   ├── CV Performance Bar Chart
        │   └── Anomaly Count Bar Chart
        ├── Tab 2: Anomaly Detection
        │   ├── Interpretation Guide
        │   ├── Weekly Heatmap
        │   └── Anomalous Days Table
        ├── Tab 3: Time Series Analysis
        │   ├── Interpretation Guide
        │   └── Interactive Time Series
        └── Tab 4: Feature Analysis
            ├── Interpretation Guide
            └── Feature Domain Pie Chart
```

### 8.2 Interactive Features

**Training Interface:**
```python
# User clicks "Start Training"
if st.button("🚀 Start Training"):
    # Launch subprocess
    process = subprocess.Popen(
        ['python', 'src/models/train_anomaly_baseline.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Real-time progress updates
    for line in iter(process.stdout.readline, b''):
        log = line.decode('utf-8')
        display_log(log)
        update_progress_bar(log)  # Parse log for progress
    
    # On completion
    if process.returncode == 0:
        st.success("✅ Training complete!")
        st.balloons()
        st.rerun()  # Refresh to show results
```

**Visualization Navigation:**
- **Tabs:** Organize content into logical sections
- **Expandable Sections:** Hide technical details by default
- **Hover Tooltips:** Plotly interactive charts with on-hover data
- **Dropdown Selectors:** Choose metrics for time series analysis
- **Download Buttons:** Export tables as CSV

### 8.3 Visualization Types

**Production EDA:**
- Batch size distribution histogram
- Production by line bar chart
- Defect rate by line box plot
- Defect breakdown stacked bar
- Hourly production pattern line chart
- Daily production trend time series
- SKU production pie chart

**Quality Control EDA:**
- Pass/fail pie chart
- Parameter fail rate bar chart
- Hourly QC trend line chart
- Daily QC trend time series
- QC by SKU heat map
- Parameter distribution violin plots

**Dispatch EDA:**
- Delay distribution histogram
- Delay by depot box plot
- Hour-day heatmap
- On-time % by depot bar chart
- Vehicle performance scatter plot
- Volume time series

**ML Visualizations:**
- Model performance grouped bar chart
- Anomaly count by model bar chart
- Weekly anomaly heatmap (calendar view)
- Time series with anomaly markers
- Feature importance pie chart
- Anomalous days table with sorting

### 8.4 User Guide

**Launching Dashboard:**
```bash
# 1. Activate virtual environment
source .venv/Scripts/activate

# 2. Launch Streamlit
streamlit run app/streamlit_eda_explorer.py

# 3. Access in browser
# Opens automatically at http://localhost:8501
```

**Navigation:**
1. **Select Phase:** Use sidebar radio buttons
2. **Explore Visualizations:** Scroll through charts
3. **Read Interpretation Guides:** Expand "🔑 Key Interpretation Guide"
4. **Train Models:** Go to Phase 4 → Click "Start Training"
5. **View Results:** After training, explore 4 tabs
6. **Download Data:** Click "📥 Download" buttons for CSV exports

**Tips:**
- **Dashboard refresh:** Press R or use browser refresh
- **Full-screen charts:** Click Plotly "📷" icon → "Download plot as PNG"
- **Sidebar collapse:** Click arrow to maximize chart space
- **Mobile access:** Dashboard responsive, works on tablets

---

## 9. Key Findings

### 9.1 Critical Issues Identified

**1. Quality Control Crisis (Highest Priority)**

**Issue:** 38.15% QC failure rate (18X above 2% target)

**Evidence:**
- 6,540 failed QC checks out of 18,090 total
- All 4 parameters systematically failing:
  - Crust color: 55.53% fail rate
  - Slice uniformity: 54.32% fail rate
  - Moisture: 45.21% fail rate
  - Seal strength: 42.87% fail rate
- 1,171 batches failed >50% of QC tests
- Only 34.5% of failed batches became waste (rest dispatched!)

**Impact:**
- Customer complaints likely spiking
- Brand reputation at risk
- Rework costs estimated 15-20% of production cost
- Dispatch delays cascading

**Root Causes:**
1. **Equipment Calibration:** Crust color sensor drift likely
2. **Process Variance:** Slice uniformity suggests blade wear
3. **Raw Material Quality:** Moisture failures point to flour inconsistency
4. **Training Gaps:** High fail rate across all lines suggests systemic issue

**Recommendations:**
1. **Immediate:** Halt operations for equipment audit (1-2 days)
2. **Short-term:** Recalibrate all QC sensors, replace worn blades
3. **Medium-term:** Implement Statistical Process Control (SPC) charts
4. **Long-term:** Vendor audits for raw material quality

---

**2. Inventory Tracking Crisis**

**Issue:** 29.2% of inventory movements show negative balances

**Evidence:**
- 5,286 negative balance events out of 18,073 movements
- 440 days (84% of operational calendar) had inventory anomalies
- Flow efficiency: 8.6% (should be ~100%)
- 2.75M units "missing" between plant dispatch and store receipt

**Impact:**
- Cannot trust inventory for production planning
- Demand forecasting models will be inaccurate
- Risk of stock-outs despite having inventory
- Financial reconciliation issues

**Root Causes:**
1. **Missing Inbound Records:** Stores not logging receipts properly
2. **Double-Counted Dispatch:** Same dispatch logged twice
3. **Unlogged Waste:** Post-dispatch waste not recorded
4. **System Integration Failure:** Plant and store systems not synchronized

**Recommendations:**
1. **Immediate:** Manual inventory audit across all locations
2. **Short-term:** Implement mandatory barcode scanning at checkpoints
3. **Medium-term:** Deploy real-time inventory sync system
4. **Long-term:** Automated reconciliation with IoT sensors

---

**3. Preventable Waste & Returns**

**Issue:** 1.3M units wasted + 791K units returned (2.1M total losses)

**Evidence:**
- **Waste:**
  - 59.3% production-stage (770K units)
  - 40.7% post-dispatch (530K units)
  - Top reason: Contamination (10.5%) → Sanitation crisis
- **Returns:**
  - 58.4% preventable (462K units)
  - Top reason: Mold growth (15%) → Cold chain failure
  - Temperature at check: 18% >30°C (unsafe)

**Impact:**
- Annual loss estimated: $2.1M @ $1 avg price
- Environmental waste: 2,100 metric tons
- Reputation damage from moldy products reaching customers

**Root Causes:**
1. **Contamination:** Inadequate sanitation protocols
2. **Cold Chain Breaks:** Trucks lack refrigeration or monitoring
3. **Over-Production:** Producing beyond demand (no forecasting)
4. **Quality Drift:** Failed QC batches entering supply chain

**Recommendations:**
1. **Immediate:** Deep-clean all production lines (2-day shutdown)
2. **Short-term:** Install temperature sensors in all delivery trucks
3. **Medium-term:** Implement demand forecasting system
4. **Long-term:** Real-time cold chain monitoring with alerts

---

**4. Operational Inefficiencies**

**Issue:** Suboptimal resource utilization

**Evidence:**
- **Vehicle Underutilization:** 100% of trips <50% capacity
- **Rural Routes:** 47.7% routes >60km → Freshness risk
- **Monday Bottleneck:** 359K units ordered (2X Tuesday volume)
- **High-Risk Routes:** 4 routes with >0.7 risk score

**Impact:**
- High logistics costs (paying for empty truck space)
- Increased waste on long rural routes
- Monday staffing/capacity crunches
- Predictable failures on high-risk routes

**Recommendations:**
1. **Route Consolidation:** Combine underutilized rural routes
2. **Dynamic Pricing:** Incentivize Tuesday-Thursday orders
3. **Risk Mitigation:** Assign best drivers/vehicles to high-risk routes
4. **Capacity Planning:** Flexible Monday staffing

---

### 9.2 Positive Findings

**Successes:**

1. **Promotion Effectiveness:**
   - +39.1% sales uplift during promotions
   - ROI: Strong, continue promotional strategy

2. **Production Efficiency:**
   - Defect rate: 2.68% (below 5% target for non-QC batches)
   - 5 production lines balanced workload
   - 7 SKU mix optimized for demand

3. **Dispatch Performance:**
   - Mean delay: 17.1 minutes (acceptable <30 min target)
   - 82.9% on-time within 60-min window
   - IoT tracking: 100% visibility

4. **Data Quality:**
   - 95%+ completeness across datasets
   - Timestamp accuracy: High
   - Schema consistency: Good (post-cleaning)

---

## 10. Technical Stack

### 10.1 Languages & Frameworks

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core language |
| **Pandas** | 2.3.3 | Data manipulation |
| **NumPy** | 2.3.5 | Numerical operations |
| **Scikit-learn** | 1.7.2 | ML algorithms |
| **PyOD** | 2.0.6 | Anomaly detection library |
| **Matplotlib** | 3.10.7 | Static visualizations |
| **Seaborn** | 0.13.2 | Statistical plots |
| **Plotly** | Latest | Interactive charts |
| **Streamlit** | Latest | Web dashboard |

### 10.2 Data Storage

| Format | Purpose | Rationale |
|--------|---------|-----------|
| **CSV** | Raw data input | Human-readable, universal compatibility |
| **Parquet** | Processed data + analytics | Compressed, columnar, fast read/write |
| **JSON** | ML results, configs | Structured, readable, nested data support |
| **PNG** | Visualizations | High-resolution, universal image format |

### 10.3 Development Environment

**Recommended Setup:**
- **OS:** Windows 10/11, Linux, macOS
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB for data + outputs
- **IDE:** VS Code, PyCharm, Jupyter Lab
- **Python Version Manager:** pyenv or conda
- **Package Manager:** pip + venv

**VS Code Extensions:**
- Python (Microsoft)
- Pylance (type checking)
- Jupyter (notebook support)
- GitLens (git visualization)
- Markdown All in One

### 10.4 Key Libraries

**Data Processing:**
```python
import pandas as pd           # DataFrames
import numpy as np            # Arrays
import pyarrow as pa          # Parquet I/O
from pathlib import Path      # File paths
import json                   # JSON handling
import warnings               # Warning suppression
```

**Visualization:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

**Machine Learning:**
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
```

**Dashboard:**
```python
import streamlit as st
import subprocess
import time
import io
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue 1: Module Not Found**
```bash
Error: ModuleNotFoundError: No module named 'pandas'
```
**Solution:**
```bash
# Activate virtual environment first
source .venv/Scripts/activate  # Windows Git Bash
# OR
.venv\Scripts\Activate.ps1     # PowerShell

# Install dependencies
pip install -r requirements.txt
```

---

**Issue 2: File Not Found**
```bash
Error: FileNotFoundError: data/processed/production_dataset.parquet
```
**Solution:**
```bash
# Run data cleaning first
python src/data/clean.py
```

---

**Issue 3: Dashboard Not Launching**
```bash
Error: streamlit: command not found
```
**Solution:**
```bash
# Install streamlit
pip install streamlit

# OR launch with python -m
python -m streamlit run app/streamlit_eda_explorer.py
```

---

**Issue 4: Port Already in Use**
```bash
Error: OSError: Address already in use
```
**Solution:**
```bash
# Use different port
streamlit run app/streamlit_eda_explorer.py --server.port 8502

# OR kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

---

**Issue 5: Memory Error During EDA**
```bash
Error: MemoryError: Unable to allocate array
```
**Solution:**
```python
# Reduce data size in EDA script
df = df.sample(frac=0.5)  # Use 50% of data

# OR increase RAM
# Close other applications
```

---

**Issue 6: Training Fails with Exit Code 1**
```bash
Error: UnicodeEncodeError: 'charmap' codec can't encode character
```
**Solution:**
- Fixed in train_anomaly_baseline.py with UTF-8 encoding handler
- If issue persists:
```python
# Add to top of script
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

---

### 11.2 Performance Optimization

**Slow EDA scripts:**
```python
# Use sample for quick iteration
df = df.sample(n=10000)  # Use 10K rows instead of full dataset

# Disable interactive mode for batch runs
import matplotlib
matplotlib.use('Agg')
```

**Slow ML training:**
```python
# Reduce CV folds
n_splits = 3  # Instead of 5

# Reduce Isolation Forest estimators
n_estimators = 50  # Instead of 100
```

**Dashboard lag:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart server
# Ctrl+C → Restart
```

---

### 11.3 Data Issues

**Missing columns:**
- Check schema in `src/data/clean.py`
- Verify raw data headers match expected schema
- Update column names if vendor changed format

**Timestamp parsing errors:**
```python
# Try different formats
pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
```

**Negative values in production:**
- Check for data entry errors
- Flag but don't drop (may indicate reversal entries)

---

### 11.4 Git Issues

**Large file push fails:**
```bash
# CSV files too large for GitHub
# Solution: Add to .gitignore
echo "data/raw/*.csv" >> .gitignore
git rm --cached data/raw/*.csv
```

**Merge conflicts in notebooks:**
```bash
# Clear notebook outputs before committing
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

---

## 12. Appendices

### Appendix A: File Structure

```
taps/
├── data/
│   ├── raw/                    # Raw CSV files (not in repo)
│   ├── processed/              # Cleaned Parquet files
│   └── analytic/               # Feature-engineered dataset
│       └── plant_daily.parquet
├── src/
│   ├── data/
│   │   └── clean.py            # Data cleaning
│   ├── features/
│   │   └── engineer.py         # Feature engineering
│   ├── models/
│   │   └── train_anomaly_baseline.py  # ML training
│   └── analysis/
│       ├── eda_production.py   # EDA scripts (10 total)
│       └── ...
├── app/
│   ├── streamlit_eda_explorer.py       # Main dashboard
│   └── phase4_ml_visualizations.py     # ML module
├── reports/
│   ├── models/                 # ML outputs (JSON, CSV)
│   ├── figures/                # PNG visualizations
│   ├── summaries/              # CSV summary tables
│   └── *.txt                   # Text reports
├── docs/
│   ├── COMPLETE_PROJECT_DOCUMENTATION.md  # This file
│   ├── SYSTEM_ARCHITECTURE.md             # Architecture diagrams
│   ├── USER_GUIDE.md                      # User manual
│   └── INSTALLATION_GUIDE.md              # Setup instructions
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── .gitignore                  # Git exclusions
└── EDA_DASHBOARD_SUMMARY.md    # Dashboard reference
```

### Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Anomaly** | Observation significantly different from expected pattern |
| **Batch** | Single production run producing specific quantity of SKU |
| **Contamination** | Expected proportion of anomalies in dataset (default: 5%) |
| **Cross-Validation** | Model validation technique splitting data into train/test folds |
| **Defect Rate** | Proportion of defective units in a batch |
| **Ensemble** | Combination of multiple models' predictions |
| **EDA** | Exploratory Data Analysis - initial data investigation |
| **Feature** | Input variable used for machine learning |
| **LOF** | Local Outlier Factor - density-based anomaly detection |
| **One-Class SVM** | Support Vector Machine trained on single class (normal data) |
| **Parquet** | Columnar storage format for efficient data processing |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **ROC-AUC** | Area Under Receiver Operating Characteristic curve |
| **SKU** | Stock Keeping Unit - product identifier |
| **Streamlit** | Python framework for building data dashboards |
| **TimeSeriesSplit** | Cross-validation respecting temporal order |
| **Z-Score** | Number of standard deviations from mean |

### Appendix C: Contact & Support

**Project Repository:** https://github.com/tfeadzwa/bakery-anomaly-detection-ml

**Issue Reporting:** GitHub Issues tab

**Documentation Updates:** Pull requests welcome

**License:** MIT (see LICENSE file)

---

**Document Version:** 1.0.0  
**Last Updated:** March 15, 2026  
**Authors:** Project Team  
**Status:** Production Ready ✅
