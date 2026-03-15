# Installation & Setup Guide

Complete step-by-step installation instructions for the Bakery Anomaly Detection ML project.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Prerequisites Installation](#2-prerequisites-installation)
3. [Project Setup](#3-project-setup)
4. [Dependency Installation](#4-dependency-installation)
5. [Data Preparation](#5-data-preparation)
6. [Verification](#6-verification)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. System Requirements

### Minimum Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **RAM:** 8 GB
- **Storage:** 5 GB free space
- **Internet:** Required for package downloads
- **Display:** 1280×720 minimum resolution (for dashboard)

### Recommended Requirements
- **RAM:** 16 GB (for smoother EDA execution)
- **CPU:** Multi-core processor (4+ cores)
- **Storage:** 10 GB free space (for generated visualizations)
- **Browser:** Chrome, Firefox, or Edge (latest version)

---

## 2. Prerequisites Installation

### 2.1 Python Installation

**Check Existing Installation:**
```bash
python --version
# Should be 3.10 or higher
```

**Windows Installation:**

1. **Download Python:**
   - Visit: https://www.python.org/downloads/
   - Download Python 3.10+ installer (64-bit recommended)

2. **Install:**
   - Run installer
   - ✅ **CRITICAL:** Check "Add Python to PATH"
   - Choose "Install Now"
   - Wait for completion

3. **Verify:**
   ```bash
   python --version
   pip --version
   ```

**Linux (Ubuntu/Debian) Installation:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
python3.10 --version
```

**macOS Installation:**
```bash
# Using Homebrew
brew install python@3.10

# Verify
python3 --version
pip3 --version
```

### 2.2 Git Installation

**Check Existing Installation:**
```bash
git --version
```

**Windows:**
- Download from: https://git-scm.com/download/win
- Install with default options
- Recommended: Select "Git Bash Here" context menu option

**Linux:**
```bash
sudo apt install git
```

**macOS:**
```bash
brew install git
```

### 2.3 Visual Studio C++ Build Tools (Windows Only)

Some Python packages require C++ compilers.

**Installation:**
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer
3. Select "Desktop development with C++"
4. Install (requires ~7GB)

**Alternative:** Install Visual Studio Community with C++ workload

---

## 3. Project Setup

### 3.1 Clone Repository

**Option A: HTTPS (recommended for read-only access)**
```bash
# Navigate to your projects folder
cd c:\Users\YourUsername\Documents\mywork

# Clone repository
git clone https://github.com/tfeadzwa/bakery-anomaly-detection-ml.git taps

# Enter directory
cd taps
```

**Option B: SSH (if you have SSH keys configured)**
```bash
git clone git@github.com:tfeadzwa/bakery-anomaly-detection-ml.git taps
cd taps
```

**Option C: Download ZIP**
1. Visit repository URL
2. Click "Code" → "Download ZIP"
3. Extract to `C:\Users\YourUsername\Documents\mywork\taps`
4. Open terminal in extracted folder

### 3.2 Verify Project Structure

```bash
# List root directory
ls
# Should see: README.md, requirements.txt, app/, src/, data/, etc.

# Check data folder
ls data/
# Should see: raw/, processed/, analytic/ folders
```

---

## 4. Dependency Installation

### 4.1 Create Virtual Environment

**Why Virtual Environment?**
- Isolates project dependencies
- Prevents version conflicts
- Easier to reproduce environment

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Success Indicator:** Command prompt shows `(.venv)` prefix

### 4.2 Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4.3 Install Project Dependencies

**Method 1: Install All Dependencies (Recommended)**
```bash
pip install -r requirements.txt
```

This installs ~80 packages including:
- pandas, numpy, scikit-learn (core ML)
- matplotlib, seaborn, plotly (visualization)
- streamlit (dashboard)
- mlflow (experiment tracking)
- pyarrow (Parquet support)

**Estimated Time:** 5-10 minutes depending on internet speed

**Method 2: Install Core Dependencies Only (Minimal)**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit pyarrow
```

### 4.4 Windows-Specific Dependencies

**pywin32 Installation:**
```bash
# Included in requirements.txt, but if manual install needed:
pip install pywin32==311

# Post-install script (run once)
python .venv/Scripts/pywin32_postinstall.py -install
```

### 4.5 Verify Installations

**Check Installed Packages:**
```bash
pip list
```

Should include:
```
pandas                    2.3.3
numpy                     2.3.5
scikit-learn              1.7.2
matplotlib                3.10.7
seaborn                   0.13.2
plotly                    5.x.x
streamlit                 1.x.x
pyarrow                   22.0.0
mlflow                    3.7.0
```

**Test Package Imports:**
```bash
python -c "import pandas; print('pandas OK')"
python -c "import numpy; print('numpy OK')"
python -c "import sklearn; print('scikit-learn OK')"
python -c "import streamlit; print('streamlit OK')"
python -c "import plotly; print('plotly OK')"
```

All should print "OK" without errors.

---

## 5. Data Preparation

### 5.1 Place Raw Data Files

**Data Location:** `data/raw/`

**Required Files:**
```
data/raw/
├── production_dataset.csv
├── quality_control_dataset.csv
├── dispatch_dataset.csv
├── sales_pos_dataset.csv
├── sales_dataset.csv                 # B2B sales
├── inventory_stock_movements_dataset.csv
├── waste_dataset.csv
├── returns_dataset.csv
├── route_transport_multivehicle.csv
├── equipment_iot_sensor_dataset.csv
└── holidays_calendar.csv
```

**If Data Missing:**
- Contact project supervisor for dataset files
- Place all CSV files in `data/raw/` folder
- Do NOT modify raw files

### 5.2 Run Data Cleaning

**Clean All Datasets:**
```bash
python src/data/clean.py
```

**Expected Output:**
```
Cleaning production dataset...
Cleaning quality control dataset...
Cleaning dispatch dataset...
Cleaning sales POS dataset...
Cleaning sales B2B dataset...
Cleaning inventory dataset...
Cleaning waste dataset...
Cleaning returns dataset...
Cleaning route metadata...
Cleaning IoT sensor dataset...
Cleaning holidays calendar...

✓ All datasets cleaned successfully!
Output: data/processed/*.parquet
```

**Time:** ~2-3 minutes

### 5.3 Run Feature Engineering

**Generate Analytical Dataset:**
```bash
python -m src.features.engineer
```

**Expected Output:**
```
Loading processed datasets...
Building production features...
Building dispatch features...
Building QC features...
Building waste features...
Building returns features...
Building inventory features...
Building sales features...
Adding holiday context...
Aggregating to daily grain...
Flagging anomalies...

✓ Feature engineering complete!
Written to data\analytic\plant_daily.parquet
Shape: (365, 52)
```

**Time:** ~1-2 minutes

---

## 6. Verification

### 6.1 Verify File Structure

**Check Processed Data:**
```bash
ls data/processed/
```

Should contain 11 `.parquet` files.

**Check Analytic Data:**
```bash
ls data/analytic/
```

Should contain `plant_daily.parquet`.

### 6.2 Test Dashboard Launch

**Launch Streamlit Dashboard:**
```bash
streamlit run app/streamlit_eda_explorer.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**In Browser:**
- Dashboard opens automatically
- Sidebar shows "Phase" selector
- Select "Phase 1: Production EDA" → Should display production charts
- Select "Phase 4: ML Anomaly Detection" → Should show training interface

**Stop Dashboard:** Press `Ctrl+C` in terminal

### 6.3 Test ML Training

**Option A: Interactive Training (Recommended)**
1. Launch dashboard: `streamlit run app/streamlit_eda_explorer.py`
2. Navigate to "Phase 4: ML Anomaly Detection"
3. Click "🚀 Start Training"
4. Watch progress bar (5-10 minutes)
5. Verify results display after completion

**Option B: Command-Line Training**
```bash
python src/models/train_anomaly_baseline.py
```

**Expected Output:**
```
Loading data...
Preparing features...
Running cross-validation...
Training final models...
Saving results...

[SUCCESS] PHASE 4 COMPLETE
Results saved to reports/models/
```

**Verify Outputs:**
```bash
ls reports/models/
# Should see:
# - baseline_cv_report.json
# - flagged_anomalies_baseline.csv
# - model_summary.json
```

### 6.4 Run Sample EDA Script

**Test Production EDA:**
```bash
python src/analysis/eda_production.py
```

**Expected Output:**
```
Loading production dataset...
Generating visualizations...
Creating summaries...
Saving report...

✓ Production EDA complete!
Outputs:
- reports/figures/production_*.png (8 charts)
- reports/summaries/production_*.csv (6 tables)
- reports/production_summary.txt
```

**Time:** ~2-3 minutes

---

## 7. Troubleshooting

### Issue 1: Python Not Found

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solution:**
```bash
# Windows: Use full path
C:\Users\YourUsername\AppData\Local\Programs\Python\Python310\python.exe --version

# Add to PATH manually:
# System Properties → Environment Variables → Path → Edit → New
# Add: C:\Users\YourUsername\AppData\Local\Programs\Python\Python310

# OR use py launcher:
py --version
py -m venv .venv
```

### Issue 2: Virtual Environment Activation Fails

**Error (PowerShell):**
```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Error (Linux/Mac):**
```
Permission denied
```

**Solution:**
```bash
chmod +x .venv/bin/activate
source .venv/bin/activate
```

### Issue 3: pip Install Fails with "Microsoft Visual C++ required"

**Error:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**
- Install Visual Studio Build Tools (see Section 2.3)
- Restart terminal after installation
- Retry: `pip install -r requirements.txt`

### Issue 4: Slow Package Installation

**Symptoms:** pip install hangs or takes >30 minutes

**Solution:**
```bash
# Use different PyPI mirror
pip install --index-url https://pypi.org/simple -r requirements.txt

# OR upgrade pip first
python -m pip install --upgrade pip

# OR install in batches
pip install pandas numpy
pip install scikit-learn matplotlib seaborn
pip install streamlit plotly
pip install -r requirements.txt  # Install remaining
```

### Issue 5: Data Files Not Found

**Error:**
```
FileNotFoundError: data/raw/production_dataset.csv
```

**Solution:**
1. Verify files exist: `ls data/raw/`
2. Check file names match exactly (case-sensitive on Linux/Mac)
3. Ensure files are CSV format (not .xlsx or .txt)
4. If missing, obtain from project supervisor

### Issue 6: Streamlit Command Not Found

**Error:**
```
streamlit: command not found
```

**Solution:**
```bash
# Verify virtual environment is activated
# Should see (.venv) in prompt

# If not activated:
source .venv/Scripts/activate  # Windows Git Bash
.venv\Scripts\activate.bat     # Windows CMD

# Reinstall streamlit
pip install streamlit

# Use python -m as alternative
python -m streamlit run app/streamlit_eda_explorer.py
```

### Issue 7: Port 8501 Already in Use

**Error:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Option A: Use different port
streamlit run app/streamlit_eda_explorer.py --server.port 8502

# Option B: Kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID [PID_NUMBER] /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

### Issue 8: Memory Error During EDA

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. **Close Other Applications:** Free up RAM
2. **Reduce Data Sample:**
   ```python
   # Edit EDA script temporarily
   df = df.sample(frac=0.5)  # Use 50% of data
   ```
3. **Upgrade RAM:** Consider adding more RAM if <8GB

### Issue 9: Unicode Encoding Errors (Windows)

**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution:**
- Already fixed in `train_anomaly_baseline.py`
- If occurs in other scripts, add:
  ```python
  import sys
  import io
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
  ```

### Issue 10: Git Clone Fails with SSL Error

**Error:**
```
SSL certificate problem: unable to get local issuer certificate
```

**Solution:**
```bash
# Temporary workaround (not recommended for sensitive repos)
git config --global http.sslVerify false
git clone https://...

# OR download ZIP instead (see Section 3.1 Option C)
```

---

## 8. Quick Reference Commands

### Daily Workflow Commands

```bash
# 1. Activate environment
source .venv/Scripts/activate  # Git Bash
.venv\Scripts\activate.bat     # Windows CMD

# 2. Launch dashboard
streamlit run app/streamlit_eda_explorer.py

# 3. Run training (alternative to dashboard)
python src/models/train_anomaly_baseline.py

# 4. Generate EDA reports
python src/analysis/eda_production.py
python src/analysis/eda_quality_control.py
# ... (repeat for other datasets)

# 5. Re-run feature engineering (if data changes)
python -m src.features.engineer

# 6. Deactivate environment (when done)
deactivate
```

### Package Management

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Install single package
pip install package-name

# Remove package
pip uninstall package-name

# Check outdated packages
pip list --outdated

# Export current environment
pip freeze > requirements.txt
```

### Git Commands

```bash
# Pull latest changes
git pull origin main

# Check status
git status

# Discard local changes
git checkout -- .

# Update to specific version
git checkout [commit-hash]
```

---

## 9. Platform-Specific Notes

### Windows Notes
- **File Paths:** Use backslashes `\` or forward slashes `/` (both work)
- **Terminal:** Git Bash recommended over CMD for Unix-like commands
- **pywin32:** Required for some Windows system integrations
- **Encoding:** UTF-8 handling already implemented in scripts

### Linux Notes
- **Python Command:** May need `python3` instead of `python`
- **pip Command:** May need `pip3` instead of `pip`
- **Permissions:** May need `sudo` for system-wide installs (avoid; use venv)
- **Dependencies:** Some packages may need `python3-dev`, `build-essential`

### macOS Notes
- **Python Path:** Default Python 2.7 exists; use `python3`
- **Homebrew:** Recommended package manager
- **Xcode:** May prompt to install Command Line Tools (accept)
- **M1/M2 Macs:** Most packages now support ARM architecture

---

## 10. Next Steps After Installation

Once installation is complete:

1. **Read Main Documentation:**
   - Review [COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md)
   - Understand project architecture and workflow

2. **Explore Dashboard:**
   - Launch dashboard: `streamlit run app/streamlit_eda_explorer.py`
   - Navigate through all 4 phases
   - Review visualizations and summaries

3. **Run ML Training:**
   - Navigate to Phase 4 in dashboard
   - Click "Start Training"
   - Explore results in 4 result tabs

4. **Review Generated Outputs:**
   - Check `reports/figures/` for PNG visualizations
   - Check `reports/models/` for ML results
   - Check `reports/summaries/` for CSV tables

5. **Customize & Extend:**
   - Modify EDA scripts to add custom analyses
   - Adjust ML hyperparameters in `train_anomaly_baseline.py`
   - Add new datasets to pipeline

---

## 11. Support & Resources

### Documentation
- **Main Docs:** [COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md)
- **README:** [../README.md](../README.md)
- **Dashboard Summary:** [../EDA_DASHBOARD_SUMMARY.md](../EDA_DASHBOARD_SUMMARY.md)

### External Resources
- **Python Official:** https://docs.python.org/3/
- **Pandas Docs:** https://pandas.pydata.org/docs/
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Streamlit Docs:** https://docs.streamlit.io/
- **Plotly Docs:** https://plotly.com/python/

### Troubleshooting
- **GitHub Issues:** Report bugs via repository Issues tab
- **Stack Overflow:** Tag questions with `python`, `pandas`, `scikit-learn`
- **Community Forums:** Reddit r/learnpython, r/datascience

---

**Installation Guide Version:** 1.0.0  
**Last Updated:** March 15, 2026  
**Tested On:** Windows 11, Ubuntu 22.04, macOS 13  
**Status:** Verified ✅
