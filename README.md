# 🍞Anomaly Detection and Waste Reduction in Bakery Operations

**End-to-End Machine Learning System for Bakery Supply Chain Optimization**

An integrated data-driven analytics and machine learning platform that analyzes bakery production, logistics, sales, and quality data to detect anomalies, predict waste, and optimize supply-chain performance.

---

## 📋 Project Overview

SmartBakery is a data-driven analytics and machine learning project designed to detect operational anomalies and reduce product waste in bakery production and supply-chain operations. Using large-scale transactional data generated across the bakery lifecycle — including production, quality control, dispatch, retail sales, returns, waste, inventory movements, transport routes, equipment sensors, and calendar events — the system provides end-to-end visibility into how bread products are manufactured, distributed, sold, and lost.

The project applies **anomaly detection**, **time-series analysis**, and **predictive modeling** techniques to identify unusual patterns such as abnormal production volumes, quality defects, late or inefficient dispatch routes, mismatches between supply and demand, excessive returns, and avoidable waste. By linking batch-level production data with quality inspections, logistics metadata, point-of-sale demand signals, and environmental or equipment conditions, SmartBakery enables **root-cause analysis** of operational failures that lead to spoilage, stock-outs, or inefficiencies.

In addition to detecting anomalies, the system supports **demand forecasting** and **waste prediction** by incorporating contextual factors such as public holidays, promotions, regional demand differences, and route complexity. The resulting insights can be used to optimize production planning, improve dispatch scheduling, enhance quality control processes, and reduce financial losses caused by expired or unsold products.

**SmartBakery demonstrates how machine learning and data analytics can be applied in a realistic bakery operations context to improve efficiency, sustainability, and decision-making, making it suitable for both academic research and practical industry adoption.**

### 🔑 Core Capabilities

- ✅ **Anomaly Detection** across production, logistics, and sales operations
- ✅ **Batch-Level Traceability** and root-cause analysis for quality failures
- ✅ **Waste & Returns Prediction** using supervised learning models
- ✅ **Demand Forecasting** with holiday and promotion effects
- ✅ **Route Performance Analysis** and dispatch optimization
- ✅ **Inventory Reconciliation** and stock-out detection
- ✅ **Multi-Source Data Integration** from 10+ operational datasets
- ✅ **Real-Time IoT Analytics** for temperature and equipment monitoring

---

## 🚨 Critical Findings

### **Data Integrity Crisis**
- **29.2% of inventory movements show negative balances** (5,286 out of 18,073 records)
- **Flow efficiency: 8.6%** (should be 100%) - 2.75M units "missing" between plant dispatch and store receipt
- **Root cause**: Missing inbound records, double-counted dispatch, or unlogged waste
- **Impact**: Cannot trust inventory for production planning or demand forecasting

### **Quality Control Emergency**
- **36.15% QC fail rate** (6,540 fails out of 18,090 checks) - **18X above target (2%)**
- **Top failing parameters**: crust_color_level (55.53%), slice_uniformity (54.32%)
- **Impact**: 3 out of 4 batches rejected → massive rework costs, dispatch delays, waste

### **Waste & Returns Drivers**
- **1.3M units wasted**: 59.3% at production stage, 40.7% post-dispatch
- **791K units returned**: 58.4% preventable (cold chain/quality failures)
- **Top waste reason**: Contaminated (10.5%) - sanitation crisis
- **Top return reason**: Mould Growth (15%) - cold chain failure

### **Logistics Insights**
- **47.7% rural routes** (>60km) - freshness degradation risk
- **4 high-risk routes** (>0.7 risk score) - need priority monitoring
- **100% vehicles underutilized** (<50% capacity) - consolidation opportunity
- **Monday B2B ordering peak**: 359K units (weekly restocking pattern)

---

## 📊 Datasets Analyzed

| Dataset | Records | Key Insights |
|---------|---------|--------------|
| **Production** | 15,000 batches | 444K defects (2.68% rate), 5 lines, 7 SKUs |
| **Quality Control** | 18,090 checks | **36.15% fail rate**, crust_color worst (55.53%) |
| **Dispatch** | 15,000 trips | Mean 17.1 min delay, 12 routes, IoT-tracked |
| **Sales (B2C)** | 15,000 transactions | 465K units, $1.43 avg, **+39.1% promo uplift** |
| **Sales (B2B)** | 15,099 orders | 2.45M units, 162 units/order (5.2X retail) |
| **Waste** | 14,070 incidents | **1.3M units**, 59.3% production-stage |
| **Returns** | 13,065 incidents | **791K units**, 58.4% preventable |
| **Inventory** | 18,073 movements | **🚨 29.2% negative balances**, 8.6% flow efficiency |
| **Route Metadata** | 216 configs | 69 routes, 101 vehicles, 31.7 km/h avg speed |
| **IoT Sensors** | 450,000 readings | Temp/humidity/vibration monitoring |

---

## 🚀 Quick Start

### **Prerequisites**
- **Python 3.10 or higher**
- **Git** (for cloning repository)
- **8GB+ RAM** (16GB recommended for full ML training)
- **Windows 10/11, Ubuntu 20.04+, or macOS 10.15+**

For detailed system requirements, see [Installation Guide](docs/INSTALLATION_GUIDE.md#1-system-requirements).

### **1. Clone Repository**
```bash
git clone https://github.com/tfeadzwa/bakery-anomaly-detection-ml.git
cd bakery-anomaly-detection-ml
```

### **2. Setup Virtual Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows (Git Bash):
source .venv/Scripts/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate
```

### **3. Install Dependencies**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages (~80 dependencies)
pip install -r requirements.txt
```

**Estimated time:** 5-10 minutes

**Note:** Windows users may need [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for some packages. See [Installation Guide](docs/INSTALLATION_GUIDE.md#23-visual-studio-c-build-tools-windows-only) for details.

### **4. Prepare Data**

**Step 4.1: Place Raw Data**
```bash
# Ensure all raw CSV files are in data/raw/ folder:
# - production_dataset.csv
# - quality_control_dataset.csv
# - dispatch_dataset.csv
# - sales_pos_dataset.csv
# - sales_dataset.csv (B2B)
# - inventory_stock_movements_dataset.csv
# - waste_dataset.csv
# - returns_dataset.csv
# - route_transport_multivehicle.csv
# - equipment_iot_sensor_dataset.csv
# - holidays_calendar.csv
```

**Step 4.2: Run Data Cleaning**
```bash
python src/data/clean.py
```
Output: `data/processed/*.parquet` (11 cleaned datasets)

**Step 4.3: Run Feature Engineering**
```bash
python -m src.features.engineer
```
Output: `data/analytic/plant_daily.parquet` (365 days × 52 features)

### **5. Launch Interactive Dashboard**
```bash
streamlit run app/streamlit_eda_explorer.py
```

Opens automatically at: http://localhost:8501

**Dashboard Features:**
- **Phase 1-3:** Exploratory data analysis for all 10 datasets (98 visualizations)
- **Phase 4:** Interactive ML training interface with real-time progress
- **Download Options:** Export CSV tables and PNG charts

### **6. Train ML Models**

**Option A: Interactive Training (Recommended)**
1. Launch dashboard: `streamlit run app/streamlit_eda_explorer.py`
2. Navigate to "Phase 4: ML Anomaly Detection"
3. Click "🚀 Start Training" button
4. Watch real-time progress (5-10 minutes)
5. Explore results in 4 interactive tabs

**Option B: Command-Line Training**
```bash
python src/models/train_anomaly_baseline.py
```

**Results Saved To:**
- `reports/models/baseline_cv_report.json` (CV metrics)
- `reports/models/flagged_anomalies_baseline.csv` (22 detected anomalies)
- `reports/models/model_summary.json` (summary statistics)

### **7. Generate EDA Reports (Optional)**

Run individual exploratory data analysis scripts:

```bash
python src/analysis/eda_production.py
python src/analysis/eda_quality_control.py
python src/analysis/eda_dispatch_enhanced.py
python src/analysis/eda_sales_pos.py
python src/analysis/eda_sales_b2b.py
python src/analysis/eda_inventory_enhanced.py
python src/analysis/eda_waste.py
python src/analysis/eda_returns.py
```

**Outputs:** 98 PNG visualizations + 64 CSV summaries + 10 text reports in `reports/`

---

## 📖 Complete Documentation

### **📚 Core Documentation**

| Document | Description | Topics Covered |
|----------|-------------|----------------|
| 📘 [**Complete Project Documentation**](docs/COMPLETE_PROJECT_DOCUMENTATION.md) | Comprehensive 100+ page end-to-end guide | • Executive summary<br>• System architecture<br>• Data pipeline flow diagrams<br>• Module documentation<br>• ML model specifications<br>• Dashboard usage<br>• Key findings<br>• Troubleshooting |
| 📗 [**Installation Guide**](docs/INSTALLATION_GUIDE.md) | Step-by-step setup for all platforms | • Prerequisites installation<br>• Virtual environment setup<br>• Dependency installation<br>• Data preparation<br>• Verification steps<br>• Platform-specific troubleshooting |
| 📊 [**System Architecture**](docs/SYSTEM_ARCHITECTURE.md) | Visual reference with flow diagrams | • High-level architecture<br>• Data pipeline stages<br>• Component interactions<br>• Execution workflow<br>• Technology stack<br>• Quick reference commands |
| 📕 [**EDA Dashboard Summary**](EDA_DASHBOARD_SUMMARY.md) | Interactive dashboard guide | • Dashboard features<br>• Navigation instructions<br>• Visualization reference<br>• Download options |

### **🔗 Quick Navigation**

**For Installation:** Start with [Installation Guide](docs/INSTALLATION_GUIDE.md) → Follow [Quick Start](#-quick-start)  
**For Understanding:** Read [Complete Documentation](docs/COMPLETE_PROJECT_DOCUMENTATION.md) → Review [System Architecture](docs/SYSTEM_ARCHITECTURE.md)  
**For Usage:** Launch dashboard → Explore [EDA Dashboard Summary](EDA_DASHBOARD_SUMMARY.md)  
**For Troubleshooting:** Check [Troubleshooting](#-troubleshooting) → See [Installation Guide - Section 7](docs/INSTALLATION_GUIDE.md#7-troubleshooting)

### **📈 Generated Reports & Outputs**

After running the pipeline, outputs are saved in `reports/`:

| Location | Contents | Description |
|----------|----------|-------------|
| `reports/models/` | 3 JSON/CSV files | ML training results: CV metrics, 22 detected anomalies, model summary |
| `reports/figures/` | 98 PNG visualizations | Charts from 10 EDA scripts (production, QC, dispatch, sales, inventory, etc.) |
| `reports/summaries/` | 64 CSV tables | Statistical summaries, aggregations, breakdown tables |
| `reports/*.txt` | 10 text reports | Comprehensive analysis reports for each dataset |

**Example Reports:**
- `reports/production_summary.txt` - Production EDA findings
- `reports/quality_control_summary.txt` - QC crisis analysis (38.15% fail rate)
- `reports/inventory_summary.txt` - Inventory tracking issues (29.2% negative balances)
- `reports/models/flagged_anomalies_baseline.csv` - 22 anomalous days detected by ML
- `reports/models/baseline_cv_report.json` - Cross-validation performance metrics

---

## 📂 Project Structure

```
taps/
├── data/
│   ├── raw/                              # Raw CSV datasets (11 files)
│   ├── processed/                        # Cleaned Parquet files (11 files)
│   └── analytic/                         # ML-ready feature dataset
│       └── plant_daily.parquet           # 365 days × 52 features
│
├── src/
│   ├── data/
│   │   └── clean.py                      # Data cleaning pipeline
│   ├── features/
│   │   └── engineer.py                   # Feature engineering (52 features)
│   ├── models/
│   │   └── train_anomaly_baseline.py     # ML training (4 algorithms + ensemble)
│   └── analysis/
│       ├── eda_production.py             # Production EDA
│       ├── eda_quality_control.py        # QC EDA
│       ├── eda_dispatch_enhanced.py      # Dispatch EDA
│       ├── eda_sales_pos.py              # Retail sales EDA
│       ├── eda_sales_b2b.py              # Wholesale sales EDA
│       ├── eda_inventory_enhanced.py     # Inventory EDA
│       ├── eda_waste.py                  # Waste EDA
│       └── eda_returns.py                # Returns EDA
│
├── app/
│   ├── streamlit_eda_explorer.py         # Main interactive dashboard
│   └── phase4_ml_visualizations.py       # ML training interface + results
│
├── reports/
│   ├── models/                           # ML outputs (JSON, CSV)
│   │   ├── baseline_cv_report.json       # Cross-validation metrics
│   │   ├── flagged_anomalies_baseline.csv # 22 detected anomalies
│   │   └── model_summary.json            # Summary statistics
│   ├── figures/                          # 98 PNG visualizations
│   ├── summaries/                        # 64 CSV summary tables
│   └── *.txt                             # 10 text analysis reports
│
├── docs/
│   ├── COMPLETE_PROJECT_DOCUMENTATION.md # 📘 Full system documentation
│   ├── INSTALLATION_GUIDE.md             # 📗 Setup instructions
│   ├── INVENTORY_CRISIS_REPORT.md        # Critical issue analysis
│   ├── QC_EDA_IMPLEMENTATION_SUMMARY.md  # Quality control findings
│   └── ... (additional reports)
│
├── requirements.txt                      # Python dependencies (~80 packages)
├── README.md                             # This file
└── .gitignore                            # Git exclusions
```

---

## 🔧 Technologies & Dependencies

### **Core Stack**
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **Pandas** | 2.3.3 | Data manipulation & analysis |
| **NumPy** | 2.3.5 | Numerical operations |
| **Scikit-learn** | 1.7.2 | Machine learning algorithms |
| **Matplotlib** | 3.10.7 | Static visualizations |
| **Seaborn** | 0.13.2 | Statistical plots |
| **Plotly** | Latest | Interactive charts |
| **Streamlit** | Latest | Web dashboard framework |
| **PyArrow** | 22.0.0 | Parquet file I/O |
| **MLflow** | 3.7.0 | Experiment tracking |

### **ML Algorithms**
- **Isolation Forest** - Tree-based ensemble anomaly detection
- **Local Outlier Factor (LOF)** - Density-based anomaly detection
- **One-Class SVM** - Boundary-based anomaly detection
- **Statistical Z-Score** - Threshold-based flagging
- **Ensemble Voting** - Meta-model combining all 4 algorithms

**Total Dependencies:** ~80 packages (see [requirements.txt](requirements.txt))

---

## 🎯 Machine Learning Models

### **Anomaly Detection System**

**Problem:** Detect unusual operational days from 365 days of plant operations

**Approach:** Unsupervised ensemble learning with 4 algorithms

**Input Features (13 selected from 52 engineered):**
- Production: `total_prod`, `avg_defect`
- Dispatch: `avg_delay`, `late_pct`
- Quality: `qc_pass_rate`, `qc_fail_pct`
- Waste/Returns: `total_waste`, `total_return`
- Sales: `total_sold`, `demand_collapse_pct`
- Inventory: `negative_balance_count`
- Context: `is_holiday`, `is_pre_holiday`

**Training:**
- **Validation:** 5-fold TimeSeriesSplit cross-validation
- **Ground Truth:** Rule-based anomaly flags from feature engineering
- **Metrics:** Precision, Recall, F1-Score, ROC-AUC

**Results:**
- **22 anomalous days detected** from 365 days (6% anomaly rate)
- **High-confidence anomalies:** 4 days flagged by all 4 models
- **Model agreement enhances reliability:** Ensemble reduces false positives

**Applications:**
- Early warning system for operational failures
- Root cause analysis for quality/logistics issues
- Production planning optimization
- Resource allocation for high-risk days

See [Complete Documentation](docs/COMPLETE_PROJECT_DOCUMENTATION.md#7-machine-learning-models) for detailed model specifications.

---

## ✅ Project Status

### **Completion Summary**

| Component | Status | Details |
|-----------|--------|---------|
| **Data Cleaning** | ✅ Complete | 11 datasets processed to Parquet format |
| **Feature Engineering** | ✅ Complete | 52 features engineered, 365-day analytical dataset |
| **Exploratory Data Analysis** | ✅ Complete | 10 EDA scripts, 98 visualizations, 64 CSV summaries |
| **ML Model Training** | ✅ Complete | 4 algorithms + ensemble, 22 anomalies detected |
| **Interactive Dashboard** | ✅ Complete | Streamlit dashboard with Phase 4 ML interface |
| **Documentation** | ✅ Complete | Comprehensive guides + analysis reports |

### **EDA Scripts Completion**

| Dataset | Script | Output Summary | Status |
|---------|--------|---------------|--------|
| **Production** | `eda_production.py` | 6 CSV + 7 PNG | ✅ |
| **Quality Control** | `eda_quality_control.py` | 5 CSV + 7 PNG | ✅ |
| **Dispatch** | `eda_dispatch_enhanced.py` | 2 CSV + 3 PNG | ✅ |
| **Sales (B2C)** | `eda_sales_pos.py` | 6 CSV + 10 PNG | ✅ |
| **Sales (B2B)** | `eda_sales_b2b.py` | 4 CSV + 10 PNG | ✅ |
| **Inventory** | `eda_inventory_enhanced.py` | 5 CSV + 12 PNG | ✅ |
| **Waste** | `eda_waste.py` | 4 CSV + 2 PNG | ✅ |
| **Returns** | `eda_returns.py` | 3 CSV + 3 PNG | ✅ |
| **Routes** | `eda_routes.py` | Metadata analysis | ✅ |
| **IoT Sensors** | `eda_sensors.py` | Temperature monitoring | ✅ |

**Total Outputs:**
- 📊 98 PNG visualizations
- 📈 64 CSV summary tables
- 📝 10 comprehensive text reports
- 🤖 1 trained ML model (4 algorithms + ensemble)
- 🚨 22 detected operational anomalies

See [EDA Status Report](docs/EDA_STATUS_REPORT.md) for detailed completion status.

---

## 🔍 Key Business Insights

### **1. Quality Control Crisis (CRITICAL)**
- **Issue:** 38.15% QC failure rate (18X above 2% target)
- **Impact:** 6,540 failed checks, massive rework costs, dispatch delays
- **Top Failing Parameters:** 
  - Crust color: 55.53% fail
  - Slice uniformity: 54.32% fail
  - Moisture: 45.21% fail
  - Seal strength: 42.87% fail
- **Root Cause:** Equipment calibration drift + process variance
- **Recommendation:** Immediate equipment audit, implement SPC charts

### **2. Inventory Tracking Crisis**
- **Issue:** 29.2% negative balance rate
- **Impact:** Cannot trust inventory for planning, 2.75M units "missing"
- **Root Cause:** Missing inbound records, double-counting, unlogged waste
- **Recommendation:** Manual audit + barcode scanning + real-time sync

### **3. Waste & Returns Reduction Opportunity**
- **Waste:** 1.3M units (59.3% production-stage, 40.7% post-dispatch)
- **Returns:** 791K units (58.4% preventable)
- **Top Waste Reason:** Contamination (10.5%) → Sanitation issue
- **Top Return Reason:** Mold growth (15%) → Cold chain failure
- **Recommendation:** Deep-clean production lines, install truck temperature sensors

### **4. Logistics Optimization Potential**
- **Vehicle Underutilization:** 100% of trips <50% capacity
- **High-Risk Routes:** 47.7% rural routes >60km
- **Monday Bottleneck:** 359K units ordered (2X Tuesday volume)
- **Recommendation:** Route consolidation, dynamic pricing, flexible staffing

### **Potential ROI: ~$1.2M/year**
- Waste reduction (30%): $557,700
- Returns reduction (35% of preventable): $231,660
- Quality improvement: $235,000 rework savings
- Logistics optimization: $180,000 fuel + time savings

---

## 🛠️ Troubleshooting

### **Common Installation Issues**

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Python not found** | `'python' is not recognized` | Add Python to PATH during installation, or use `py` launcher |
| **Virtual env activation fails** | PowerShell execution policy error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| **pip install fails** | "Microsoft Visual C++ required" | Install [VS Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| **Module not found** | `ModuleNotFoundError: No module named 'pandas'` | Activate virtual environment first: `source .venv/Scripts/activate` |
| **Streamlit not found** | `streamlit: command not found` | Use `python -m streamlit run app/streamlit_eda_explorer.py` |
| **Port already in use** | `Address already in use` | Use different port: `streamlit run app/streamlit_eda_explorer.py --server.port 8502` |

For detailed troubleshooting, see [Installation Guide - Section 7](docs/INSTALLATION_GUIDE.md#7-troubleshooting).

### **Data Issues**

| Issue | Symptom | Solution |
|-------|---------|----------|
| **File not found** | `FileNotFoundError: data/processed/...` | Run data cleaning: `python src/data/clean.py` |
| **Feature dataset missing** | `plant_daily.parquet not found` | Run feature engineering: `python -m src.features.engineer` |
| **Memory error during EDA** | `MemoryError: Unable to allocate` | Close other applications, or reduce data sample in EDA script |

### **Dashboard Issues**

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Dashboard won't load** | Blank page or error | Clear Streamlit cache: `streamlit cache clear`, then restart |
| **Training hangs** | Progress bar stuck | Check if `train_anomaly_baseline.py` is running in terminal (check task manager) |
| **Results not displaying** | No charts after training | Verify files exist in `reports/models/`, refresh browser (press R) |

### **Need More Help?**
- 📘 [Complete Project Documentation](docs/COMPLETE_PROJECT_DOCUMENTATION.md#11-troubleshooting)
- 📗 [Installation Guide](docs/INSTALLATION_GUIDE.md)
- 🐛 [Open GitHub Issue](https://github.com/tfeadzwa/bakery-anomaly-detection-ml/issues)

---

## 🎯 Future Enhancements

### **Phase 5: Predictive Models (Planned)**
- [ ] Supervised waste prediction models (Random Forest, XGBoost)
- [ ] Returns forecasting with weather/IoT integration
- [ ] Demand forecasting with holiday + promotion effects
- [ ] Route optimization using vehicle routing problem (VRP) algorithms

### **Phase 6: Real-Time Monitoring (Planned)**
- [ ] Live dashboard with automated anomaly alerts
- [ ] Integration with IoT sensors for real-time temperature monitoring
- [ ] Email/SMS alerts for critical QC failures
- [ ] Production line KPI tracking dashboards

### **Phase 7: Automation (Planned)**
- [ ] Scheduled data pipeline execution (daily batch jobs)
- [ ] Automated model retraining on new data
- [ ] MLOps deployment with MLflow + Docker
- [ ] REST API for anomaly detection service

### **Phase 8: Advanced Analytics (Planned)**
- [ ] Root cause analysis using causal inference
- [ ] What-if scenario modeling for production planning
- [ ] Multi-objective optimization (minimize waste + maximize throughput)
- [ ] Network analysis of supply chain bottlenecks

**Contributions Welcome!** See [Contributing](#-contributing) section.

---

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### **Ways to Contribute**
- 🐛 **Report Bugs:** Open GitHub Issues with detailed descriptions
- 💡 **Suggest Features:** Propose enhancements via Issues
- 📝 **Improve Documentation:** Submit PRs for typo fixes or clarifications
- 🧪 **Add Tests:** Write unit tests for data processing functions
- 📊 **New Visualizations:** Contribute additional EDA charts
- 🤖 **ML Models:** Implement alternative anomaly detection algorithms

### **Development Workflow**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Commit: `git commit -m "Add: New feature description"`
5. Push: `git push origin feature/new-feature`
6. Submit Pull Request with detailed description

### **Code Standards**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include comments for complex logic
- Test changes before submitting PR

---

## 📧 Contact & Support

**Project Maintainer:** Tafadzwa Muredzi  
**Repository:** https://github.com/tfeadzwa/bakery-anomaly-detection-ml  
**Issues:** [GitHub Issues Tab](https://github.com/tfeadzwa/bakery-anomaly-detection-ml/issues)

For questions about:
- **Installation:** See [Installation Guide](docs/INSTALLATION_GUIDE.md)
- **Usage:** See [Complete Documentation](docs/COMPLETE_PROJECT_DOCUMENTATION.md)
- **Data Issues:** Contact project supervisor for dataset access
- **Bugs:** Open GitHub Issue with error logs

---

## 📄 License

This project is **proprietary and confidential**. Unauthorized distribution is prohibited.

**Academic Use:** Contact maintainer for permission to use in research/education.

---

## 🎓 Project Context & Philosophy

### **Business Goal**
Reduce supply chain waste by 30% through data-driven interventions, saving $600K+ annually while improving product quality and customer satisfaction.

### **Core Philosophy**
> **"Inventory is not just another dataset—it is the state of the system."**

> **"If inventory is broken, something upstream failed."**

> **"Waste is the final loss, but it starts at production."**

### **Systems Thinking Approach**
This project treats the bakery supply chain as an **interconnected system** where:
- Production defects cascade into QC failures
- QC failures cascade into dispatch delays
- Dispatch delays cascade into waste and returns
- Inventory tracking reflects the cumulative health of all upstream processes

**Key Insight:** You cannot optimize one component (e.g., dispatch) in isolation without understanding its impact on everything else (inventory, waste, returns). This project uses data to expose hidden failures across the entire value chain.

### **Technical Approach**
1. **Multi-Source Integration:** Combine 10+ datasets to create holistic view
2. **Feature Engineering:** Transform raw data into meaningful operational metrics
3. **Unsupervised Learning:** Detect anomalies without labeled ground truth
4. **Ensemble Methods:** Combine multiple algorithms for robust detection
5. **Interactive Visualization:** Enable stakeholders to explore insights

---

## 📊 Project Impact Summary

### **Data Processed**
- **150,000+ operational records** across 10 datasets
- **365 days** of continuous plant operations analyzed
- **52 engineered features** capturing operational dynamics
- **98 visualizations** exposing hidden patterns
- **22 anomalies detected** with multi-model agreement

### **Critical Issues Identified**
1. ❌ **38.15% QC failure rate** (18X above target)
2. ❌ **29.2% inventory negative balance rate** (systemic tracking failure)
3. ❌ **1.3M units wasted** (59.3% preventable at production stage)
4. ❌ **791K units returned** (58.4% preventable via cold chain fixes)

### **Actionable Recommendations Delivered**
1. ✅ Immediate equipment calibration audit (1-2 day shutdown)
2. ✅ Real-time inventory sync system with barcode scanning
3. ✅ Production line deep-clean + temperature sensor deployment
4. ✅ Route consolidation + dynamic pricing for Monday bottleneck

### **Estimated Business Value**
- **Annual cost savings:** $1.2M (waste reduction, rework savings, logistics optimization)
- **Quality improvement:** 38% → 10% QC fail rate target
- **Waste reduction:** 30% target = 390K units saved
- **Returns reduction:** 35% of preventable = 162K units saved

---

## 🏆 Key Achievements

✅ **Complete end-to-end ML pipeline** from raw data to deployed dashboard  
✅ **Production-ready code** with modular architecture  
✅ **Comprehensive documentation** (100+ pages)  
✅ **Interactive dashboards** for non-technical stakeholders  
✅ **Validated ML models** with cross-validation and ensemble methods  
✅ **Actionable insights** backed by statistical evidence  
✅ **Scalable architecture** ready for additional datasets/plants  

---

**Last Updated:** March 15, 2026  
**Version:** 2.0.0 - Complete Documentation Release  
**Status:** ✅ Production Ready | 📘 Fully Documented | 🚀 Deployment Ready

---

**Built with ❤️ using Python, scikit-learn, Streamlit, and Plotly**
