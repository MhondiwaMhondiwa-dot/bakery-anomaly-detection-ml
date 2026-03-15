# System Architecture & Data Flow

**Project:** Bakery Anomaly Detection ML  
**Purpose:** Visual reference for system components and data flow

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Pipeline Flow](#2-data-pipeline-flow)
3. [Component Interactions](#3-component-interactions)
4. [Execution Workflow](#4-execution-workflow)
5. [Directory Structure](#5-directory-structure)
6. [Technology Stack](#6-technology-stack)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                              │
│                         (data/raw/*.csv)                            │
│                                                                     │
│  Production │ QC │ Dispatch │ Sales POS/B2B │ Waste │ Returns │   │
│  Inventory │ Routes │ IoT Sensors │ Holidays Calendar            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PROCESSING LAYER                          │
│                   (src/data/ & src/features/)                       │
│                                                                     │
│  ┌───────────────────┐         ┌──────────────────────┐           │
│  │  Data Cleaning    │────────>│ Feature Engineering  │           │
│  │  (clean.py)       │         │  (engineer.py)       │           │
│  │                   │         │                      │           │
│  │ • Validate schema │         │ • Aggregate daily    │           │
│  │ • Parse timestamps│         │ • Create 52 features │           │
│  │ • Handle nulls    │         │ • Flag anomalies     │           │
│  │ • Remove dupes    │         │ • Add context        │           │
│  └───────────────────┘         └──────────────────────┘           │
│           │                              │                         │
│           ▼                              ▼                         │
│  data/processed/*.parquet    data/analytic/plant_daily.parquet    │
└─────────────────────────────────────────────────────────────────────┘
                           │
                ┌──────────┴────────┬────────────────┐
                ▼                   ▼                ▼
┌─────────────────────┐  ┌─────────────────┐  ┌───────────────────┐
│  EXPLORATORY DATA   │  │  MACHINE        │  │  VISUALIZATION    │
│  ANALYSIS           │  │  LEARNING       │  │  DASHBOARD        │
│  (src/analysis/)    │  │  (src/models/)  │  │  (app/)           │
│                     │  │                 │  │                   │
│  10 EDA Scripts     │  │  4 Algorithms:  │  │  Streamlit App    │
│  • Production       │  │  • Isolation    │  │  • Phase 1-3: EDA │
│  • Quality Control  │  │    Forest       │  │  • Phase 4: ML    │
│  • Dispatch         │  │  • LOF          │  │                   │
│  • Sales POS        │  │  • One-Class    │  │  Features:        │
│  • Sales B2B        │  │    SVM          │  │  • Interactive    │
│  • Inventory        │  │  • Statistical  │  │    training       │
│  • Waste            │  │  • Ensemble     │  │  • Real-time      │
│  • Returns          │  │                 │  │    progress       │
│  • Routes           │  │  Validation:    │  │  • Results viz    │
│  • IoT Sensors      │  │  • 5-fold CV    │  │  • Downloads      │
│                     │  │  • TimeSeriesSp │  │                   │
│  Outputs:           │  │                 │  │  98+ Charts       │
│  • 98 PNG charts    │  │  Output:        │  │  Interactive      │
│  • 64 CSV tables    │  │  • 22 anomalies │  │  Exploration      │
│  • 10 TXT reports   │  │  • CV metrics   │  │                   │
└─────────────────────┘  └─────────────────┘  └───────────────────┘
         │                        │                      │
         └────────────────────────┴──────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  OUTPUTS & REPORTS          │
                    │  (reports/)                 │
                    │                             │
                    │  • models/                  │
                    │    - JSON metrics           │
                    │    - CSV anomalies          │
                    │  • figures/                 │
                    │    - 98 PNG visualizations  │
                    │  • summaries/               │
                    │    - 64 CSV tables          │
                    │  • *.txt analysis reports   │
                    └─────────────────────────────┘
```

---

## 2. Data Pipeline Flow

### **Stage 1: Data Acquisition**
```
Raw Data Sources (CSV)
    ↓
data/raw/
├── production_dataset.csv
├── quality_control_dataset.csv
├── dispatch_dataset.csv
├── sales_pos_dataset.csv
├── sales_dataset.csv (B2B)
├── inventory_stock_movements_dataset.csv
├── waste_dataset.csv
├── returns_dataset.csv
├── route_transport_multivehicle.csv
├── equipment_iot_sensor_dataset.csv
└── holidays_calendar.csv
```

### **Stage 2: Data Cleaning (clean.py)**
```
Transformations Applied:
┌──────────────────────────────────────────┐
│ • Schema Validation                      │
│   - Expected columns present             │
│   - Correct data types                   │
│                                          │
│ • Timestamp Parsing                      │
│   - Convert to datetime64[ns]           │
│   - Handle multiple formats              │
│                                          │
│ • NULL Handling                          │
│   - Forward fill time series             │
│   - Median imputation for numerics       │
│   - Mode for categoricals                │
│                                          │
│ • Duplicate Removal                      │
│   - Identify based on composite keys     │
│   - Keep most recent record              │
│                                          │
│ • Column Standardization                 │
│   - Rename to snake_case                 │
│   - Multi-SKU column aggregation         │
│   - Signed quantity conversion           │
└──────────────────────────────────────────┘
    ↓
Output: data/processed/*.parquet (11 files)
```

### **Stage 3: Feature Engineering (engineer.py)**
```
Input: data/processed/*.parquet (11 files)
    ↓
┌─────────────────────────────────────────────────────────┐
│  FEATURE GROUPS                                         │
│                                                         │
│  Production (3):      Dispatch (3):    QC (8):         │
│  • total_prod         • avg_delay      • qc_pass_rate  │
│  • avg_defect         • late_pct       • qc_fail_pct   │
│  • high_defect_count  • early_pct      • batch_fail_ct │
│                                        • param fails   │
│  Waste (2):           Returns (2):     Sales (5):      │
│  • total_waste        • total_return   • total_sold    │
│  • route_spike_pct    • return_spike   • demand_drop   │
│                                                         │
│  Inventory (3):       Holiday Context (3):             │
│  • negative_bal_ct    • is_holiday                     │
│  • stock_movements    • is_pre_holiday                 │
│  • expiry_count       • is_post_holiday                │
│                                                         │
│  Anomaly Flags (8): Rule-based thresholds              │
└─────────────────────────────────────────────────────────┘
    ↓
Aggregation: Daily Grain (365 days × 52 features)
    ↓
Output: data/analytic/plant_daily.parquet
```

### **Stage 4: Analysis & Modeling**

**Path A: Exploratory Data Analysis**
```
data/processed/*.parquet
    ↓
10 EDA Scripts
    ├── eda_production.py
    ├── eda_quality_control.py
    ├── eda_dispatch_enhanced.py
    ├── eda_sales_pos.py
    ├── eda_sales_b2b.py
    ├── eda_inventory_enhanced.py
    ├── eda_waste.py
    ├── eda_returns.py
    ├── eda_routes.py
    └── eda_sensors.py
    ↓
Outputs:
├── reports/figures/*.png (98 charts)
├── reports/summaries/*.csv (64 tables)
└── reports/*_summary.txt (10 reports)
```

**Path B: Machine Learning**
```
data/analytic/plant_daily.parquet
    ↓
src/models/train_anomaly_baseline.py
    ↓
┌──────────────────────────────────────┐
│  ML Training Pipeline                │
│                                      │
│  1. Feature Selection (13/52)       │
│  2. Preprocessing (StandardScaler)  │
│  3. Cross-Validation (5-fold)       │
│  4. Model Training (4 algorithms)   │
│     • Isolation Forest              │
│     • Local Outlier Factor          │
│     • One-Class SVM                 │
│     • Statistical Z-Score           │
│  5. Ensemble Voting (≥1 agreement)  │
│  6. Results Saving                  │
└──────────────────────────────────────┘
    ↓
Outputs:
├── reports/models/baseline_cv_report.json
├── reports/models/flagged_anomalies_baseline.csv (22)
└── reports/models/model_summary.json
```

### **Stage 5: Visualization & Deployment**
```
Outputs from Stages 3-4
    ↓
app/streamlit_eda_explorer.py
    ├── Phase 1: Production EDA
    ├── Phase 2: Quality Control EDA
    ├── Phase 3: Dispatch EDA
    └── Phase 4: ML Anomaly Detection
            ├── Training Interface
            └── Results Dashboard
    ↓
Web Browser: http://localhost:8501
```

---

## 3. Component Interactions

### **Data Flow Diagram**
```
┌─────────┐
│ Raw CSV │
└────┬────┘
     │
     ▼
┌────────────┐
│ clean.py   │───────┐
└────┬───────┘       │
     │               │  Processed
     ▼               │  Parquet Files
┌────────────┐       │
│ Processed  │◄──────┘
│  Parquet   │
└─┬─────┬────┘
  │     │
  │     ▼
  │ ┌──────────────┐
  │ │ engineer.py  │
  │ └──────┬───────┘
  │        │
  │        ▼
  │  ┌──────────────┐
  │  │plant_daily   │
  │  │  .parquet    │
  │  └──┬─────┬─────┘
  │     │     │
  ▼     ▼     ▼
┌───┐ ┌───┐ ┌──────┐
│EDA│ │ ML│ │Dashbd│
└─┬─┘ └─┬─┘ └──┬───┘
  │     │      │
  └─────┴──────┘
        │
        ▼
  ┌──────────┐
  │ Reports  │
  └──────────┘
```

### **Module Dependency Graph**
```
src/data/clean.py (NO DEPENDENCIES)
    │
    └──> data/processed/*.parquet
            │
            ├──> src/features/engineer.py
            │       │
            │       └──> data/analytic/plant_daily.parquet
            │               │
            │               └──> src/models/train_anomaly_baseline.py
            │                       │
            │                       └──> reports/models/*.json, *.csv
            │
            └──> src/analysis/eda_*.py (10 scripts - INDEPENDENT)
                    │
                    └──> reports/figures/*.png
                         reports/summaries/*.csv
                         reports/*_summary.txt

app/streamlit_eda_explorer.py
    ├── Reads: data/processed/*.parquet
    ├── Reads: reports/figures/*.png
    ├── Reads: reports/summaries/*.csv
    └── Launches: src/models/train_anomaly_baseline.py (subprocess)
            │
            └──> Displays: reports/models/*.json, *.csv
```

---

## 4. Execution Workflow

### **Complete Pipeline Execution**
```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: ENVIRONMENT SETUP (One-Time)                       │
├─────────────────────────────────────────────────────────────┤
│  1. Clone repository                                        │
│  2. Create virtual environment: python -m venv .venv        │
│  3. Activate: source .venv/Scripts/activate                 │
│  4. Install dependencies: pip install -r requirements.txt   │
│  Time: 5-10 minutes                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: DATA PREPARATION                                   │
├─────────────────────────────────────────────────────────────┤
│  1. Place raw CSV files in data/raw/                        │
│  2. Run data cleaning: python src/data/clean.py             │
│  3. Run feature engineering: python -m src.features.engineer│
│  Time: 3-5 minutes                                          │
│  Output: data/processed/*.parquet + data/analytic/plant_daily│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: EXPLORATORY ANALYSIS (Optional but Recommended)    │
├─────────────────────────────────────────────────────────────┤
│  Run all 10 EDA scripts:                                    │
│  • python src/analysis/eda_production.py                    │
│  • python src/analysis/eda_quality_control.py              │
│  • python src/analysis/eda_dispatch_enhanced.py            │
│  • python src/analysis/eda_sales_pos.py                     │
│  • python src/analysis/eda_sales_b2b.py                     │
│  • python src/analysis/eda_inventory_enhanced.py           │
│  • python src/analysis/eda_waste.py                         │
│  • python src/analysis/eda_returns.py                       │
│  • (routes & sensors optional)                              │
│  Time: 15-20 minutes (all scripts)                          │
│  Output: 98 PNG charts + 64 CSV tables + 10 TXT reports     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: ML MODEL TRAINING                                  │
├─────────────────────────────────────────────────────────────┤
│  Option A: Interactive (Recommended)                        │
│    1. Launch dashboard: streamlit run app/streamlit_eda_e...│
│    2. Navigate to Phase 4                                   │
│    3. Click "Start Training"                                │
│    4. Watch real-time progress                              │
│                                                             │
│  Option B: Command-Line                                     │
│    python src/models/train_anomaly_baseline.py             │
│                                                             │
│  Time: 5-10 minutes                                         │
│  Output: 3 files in reports/models/ (JSON + CSV)            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: RESULTS EXPLORATION                                │
├─────────────────────────────────────────────────────────────┤
│  Dashboard automatically displays results:                  │
│  • Model Performance (CV metrics)                           │
│  • Anomaly Detection (calendar heatmap)                     │
│  • Time Series Analysis (interactive timeline)              │
│  • Feature Analysis (importance breakdown)                  │
│  • Anomalous Days Table (detailed view)                     │
│                                                             │
│  Download options for CSV/PNG exports                       │
└─────────────────────────────────────────────────────────────┘
```

### **Daily Usage After Initial Setup**
```bash
# 1. Activate environment
source .venv/Scripts/activate

# 2. Launch dashboard
streamlit run app/streamlit_eda_explorer.py

# Done! Dashboard opens at http://localhost:8501
```

---

## 5. Directory Structure

```
taps/
│
├── data/                           # Data storage
│   ├── raw/                        # Raw CSV files (11 datasets)
│   ├── processed/                  # Cleaned Parquet files (11 files)
│   └── analytic/                   # ML-ready aggregated dataset
│       └── plant_daily.parquet     # 365 days × 52 features
│
├── src/                            # Source code
│   ├── data/                       # Data processing scripts
│   │   └── clean.py                # Data cleaning pipeline
│   ├── features/                   # Feature engineering
│   │   └── engineer.py             # Feature creation (52 features)
│   ├── models/                     # Machine learning
│   │   └── train_anomaly_baseline.py  # ML training script
│   ├── analysis/                   # Exploratory data analysis
│   │   ├── eda_production.py
│   │   ├── eda_quality_control.py
│   │   ├── eda_dispatch_enhanced.py
│   │   ├── eda_sales_pos.py
│   │   ├── eda_sales_b2b.py
│   │   ├── eda_inventory_enhanced.py
│   │   ├── eda_waste.py
│   │   ├── eda_returns.py
│   │   ├── eda_routes.py
│   │   └── eda_sensors.py
│   └── utils/                      # Utility functions (if any)
│
├── app/                            # Dashboard applications
│   ├── streamlit_eda_explorer.py   # Main EDA dashboard
│   └── phase4_ml_visualizations.py # ML training UI + results
│
├── reports/                        # Generated outputs
│   ├── models/                     # ML model results
│   │   ├── baseline_cv_report.json        # CV metrics
│   │   ├── flagged_anomalies_baseline.csv # 22 anomalies
│   │   └── model_summary.json             # Summary stats
│   ├── figures/                    # 98 PNG visualizations
│   │   ├── production_*.png
│   │   ├── qc_*.png
│   │   ├── dispatch_*.png
│   │   └── ...
│   ├── summaries/                  # 64 CSV summary tables
│   │   ├── production_by_line.csv
│   │   ├── qc_fail_breakdown.csv
│   │   └── ...
│   └── *.txt                       # 10 text analysis reports
│       ├── production_summary.txt
│       ├── quality_control_summary.txt
│       └── ...
│
├── docs/                           # Project documentation
│   ├── COMPLETE_PROJECT_DOCUMENTATION.md  # 📘 Full guide
│   ├── INSTALLATION_GUIDE.md              # 📗 Setup instructions
│   ├── SYSTEM_ARCHITECTURE.md             # 📊 This document
│   ├── INVENTORY_CRISIS_REPORT.md         # Analysis reports
│   └── ... (additional reports)
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── .gitignore                      # Git exclusions
└── EDA_DASHBOARD_SUMMARY.md        # Dashboard reference
```

---

## 6. Technology Stack

### **Layer 1: Data Storage & I/O**
```
┌─────────────┬───────────┬─────────────────────────────────┐
│ Format      │ Library   │ Purpose                         │
├─────────────┼───────────┼─────────────────────────────────┤
│ CSV         │ pandas    │ Raw data input                  │
│ Parquet     │ pyarrow   │ Processed data storage          │
│ JSON        │ json      │ ML results, configurations      │
│ PNG         │ matplotlib│ Visualization outputs           │
└─────────────┴───────────┴─────────────────────────────────┘
```

### **Layer 2: Data Processing**
```
┌─────────────┬───────────┬─────────────────────────────────┐
│ Task        │ Library   │ Key Functions                   │
├─────────────┼───────────┼─────────────────────────────────┤
│ Manipulation│ pandas    │ read_csv, to_parquet, groupby   │
│ Numerical   │ numpy     │ Array operations, aggregations  │
│ Datetime    │ pandas    │ to_datetime, date_range         │
│ Statistics  │ scipy     │ Statistical tests               │
└─────────────┴───────────┴─────────────────────────────────┘
```

### **Layer 3: Machine Learning**
```
┌──────────────────┬─────────────┬─────────────────────────┐
│ Algorithm        │ Library     │ Use Case                │
├──────────────────┼─────────────┼─────────────────────────┤
│ Isolation Forest │ sklearn     │ Global outlier detection│
│ LOF              │ sklearn     │ Local outlier detection │
│ One-Class SVM    │ sklearn     │ Boundary-based anomaly  │
│ Z-Score          │ scipy/numpy │ Statistical threshold   │
│ StandardScaler   │ sklearn     │ Feature normalization   │
│ TimeSeriesSplit  │ sklearn     │ Temporal cross-validation│
└──────────────────┴─────────────┴─────────────────────────┘
```

### **Layer 4: Visualization**
```
┌─────────────┬───────────┬─────────────────────────────────┐
│ Type        │ Library   │ Charts Created                  │
├─────────────┼───────────┼─────────────────────────────────┤
│ Static      │ matplotlib│ Line, bar, scatter, histogram   │
│ Statistical │ seaborn   │ Box, violin, heatmap            │
│ Interactive │ plotly    │ Time series, 3D, animations     │
└─────────────┴───────────┴─────────────────────────────────┘
```

### **Layer 5: Web Dashboard**
```
┌────────────┬───────────┬──────────────────────────────────┐
│ Component  │ Technology│ Purpose                          │
├────────────┼───────────┼──────────────────────────────────┤
│ Framework  │ Streamlit │ Web app infrastructure           │
│ UI Elements│ Streamlit │ Buttons, tabs, expandables       │
│ Charts     │ Plotly    │ Interactive visualizations       │
│ Tables     │ pandas    │ Data display                     │
└────────────┴───────────┴──────────────────────────────────┘
```

### **Full Dependency List**
```
CORE:
• Python 3.10+
• pandas 2.3.3
• numpy 2.3.5
• scikit-learn 1.7.2

VISUALIZATION:
• matplotlib 3.10.7
• seaborn 0.13.2
• plotly (latest)

DASHBOARD:
• streamlit (latest)

DATA I/O:
• pyarrow 22.0.0 (Parquet)

ML TOOLS:
• mlflow 3.7.0 (experiment tracking)
• scipy 1.15.2 (statistics)

TOTAL: ~80 packages
See requirements.txt for complete list
```

---

## 7. Scalability Considerations

### **Current Capacity**
- **Data Volume:** 150K+ records across 10 datasets
- **Time Range:** 365 days (1 year)
- **Features:** 52 engineered features
- **Processing Time:** ~30 minutes end-to-end
- **Memory Usage:** <4GB RAM

### **Scaling Strategies**

**Horizontal Scaling (More Data):**
```
Current: 1 plant × 1 year
    ↓
Future: N plants × M years
    │
    ├──> Partition by plant_id + year
    ├──> Parallel processing with Dask
    ├──> Incremental model updates
    └──> Distributed dashboard (multi-page)
```

**Vertical Scaling (More Features):**
```
Current: 52 features
    ↓
Future: 100+ features
    │
    ├──> Feature selection algorithms
    ├──> Dimensionality reduction (PCA)
    ├──> Feature importance ranking
    └──> Automated feature engineering
```

**Real-Time Processing:**
```
Current: Batch processing
    ↓
Future: Streaming
    │
    ├──> Apache Kafka for data ingestion
    ├──> Online learning for models
    ├──> Live dashboard updates
    └──> Real-time alerting system
```

---

## 8. Security & Data Privacy

### **Data Handling**
- ✅ Raw data stored locally (not in Git repo)
- ✅ No sensitive PII in datasets
- ✅ Results anonymized (no employee/customer names)
- ✅ Access control via server authentication

### **Best Practices**
- Keep `data/raw/` in `.gitignore`
- Use environment variables for credentials
- Sanitize outputs before sharing
- Regular security audits

---

## 9. Quick Reference Commands

### **Data Pipeline**
```bash
# Full pipeline execution
python src/data/clean.py                    # 2-3 mins
python -m src.features.engineer             # 1-2 mins
python src/models/train_anomaly_baseline.py # 5-10 mins
```

### **EDA Scripts**
```bash
# Run all EDAs (15-20 mins total)
for script in src/analysis/eda_*.py; do python "$script"; done
```

### **Dashboard**
```bash
# Launch main dashboard
streamlit run app/streamlit_eda_explorer.py

# Use custom port
streamlit run app/streamlit_eda_explorer.py --server.port 8502
```

### **Environment Management**
```bash
# Activate
source .venv/Scripts/activate  # Windows Git Bash
.venv\Scripts\Activate.ps1     # Windows PowerShell
source .venv/bin/activate      # Linux/macOS

# Deactivate
deactivate
```

---

## 10. Related Documentation

- 📘 [Complete Project Documentation](COMPLETE_PROJECT_DOCUMENTATION.md) - Comprehensive guide
- 📗 [Installation Guide](INSTALLATION_GUIDE.md) - Setup instructions
- 📕 [EDA Dashboard Summary](../EDA_DASHBOARD_SUMMARY.md) - Dashboard reference
- 📙 [README](../README.md) - Project overview

---

**Document Version:** 1.0.0  
**Last Updated:** March 15, 2026  
**Purpose:** Technical reference for system architecture  
**Audience:** Developers, data scientists, system architects
