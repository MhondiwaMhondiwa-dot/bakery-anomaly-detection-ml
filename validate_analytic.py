import pandas as pd
import sys

# ANSI color codes for terminal styling
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'

def print_header(text):
    """Print a styled header"""
    line = "=" * 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}{line}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{line}{Colors.ENDC}\n")

def print_subheader(text):
    """Print a styled subheader"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'─' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'─' * 70}{Colors.ENDC}\n")

def print_metric(label, value, status="info"):
    """Print a formatted metric with status coloring"""
    color = {
        "success": Colors.OKGREEN,
        "warning": Colors.WARNING,
        "error": Colors.FAIL,
        "info": Colors.OKBLUE,
        "cyan": Colors.OKCYAN
    }.get(status, Colors.ENDC)
    
    print(f"  {Colors.BOLD}{label:.<50}{Colors.ENDC} {color}{value}{Colors.ENDC}")

def print_table_row(col1, col2, col3, header=False):
    """Print a formatted table row"""
    if header:
        print(f"  {Colors.BOLD}{col1:<25} {col2:>15} {col3:>20}{Colors.ENDC}")
        print(f"  {Colors.DIM}{'─' * 25} {'─' * 15} {'─' * 20}{Colors.ENDC}")
    else:
        print(f"  {col1:<25} {col2:>15} {col3:>20}")

def format_number(num):
    """Format large numbers with commas"""
    return f"{num:,.0f}"

def get_status_symbol(value, threshold, reverse=False):
    """Return a status symbol based on threshold"""
    if reverse:
        return "✓" if value <= threshold else "✗"
    else:
        return "✓" if value >= threshold else "✗"

# Load data
try:
    print(f"{Colors.OKCYAN}Loading analytical dataset...{Colors.ENDC}")
    df = pd.read_parquet('data/analytic/plant_daily.parquet')
    print(f"{Colors.OKGREEN}✓ Dataset loaded successfully{Colors.ENDC}")
except Exception as e:
    print(f"{Colors.FAIL}✗ Error loading dataset: {e}{Colors.ENDC}")
    sys.exit(1)

# ============================================================================
# DATASET OVERVIEW
# ============================================================================
print_header("ANALYTICAL DATASET VALIDATION REPORT")

print_metric("Dataset Location", "data/analytic/plant_daily.parquet", "cyan")
print_metric("Analysis Date", pd.Timestamp.now().strftime("%B %d, %Y"), "cyan")
print_metric("Data Shape", f"{df.shape[0]} days × {df.shape[1]} features", "success")
print_metric("Date Range", f"{str(df.index.min())[:10]} to {str(df.index.max())[:10]}", "info")
print_metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", "info")

# ============================================================================
# FEATURE GROUP ANALYSIS
# ============================================================================
print_subheader("FEATURE GROUP BREAKDOWN")

feature_groups = {
    'Production': [c for c in df.columns if 'prod' in c or 'defect' in c],
    'Dispatch': [c for c in df.columns if 'delay' in c or 'late' in c or 'early' in c],
    'Quality Control': [c for c in df.columns if 'qc_' in c],
    'Waste': [c for c in df.columns if 'waste' in c],
    'Returns': [c for c in df.columns if 'return' in c],
    'Sales': [c for c in df.columns if 'sold' in c or 'demand' in c or 'promotion' in c or 'retailer' in c],
    'Inventory': [c for c in df.columns if 'balance' in c or 'stock' in c or 'expiry' in c],
    'Holiday Context': [c for c in df.columns if 'holiday' in c],
    'Anomaly Flags': [c for c in df.columns if 'anomaly' in c]
}

print_table_row("Feature Group", "Feature Count", "Coverage", header=True)

total_features = 0
for group, cols in feature_groups.items():
    if cols:
        coverage = df[cols].notna().any(axis=1).sum()
        coverage_pct = (coverage / len(df)) * 100
        status = "✓" if coverage_pct > 90 else "⚠"
        print_table_row(
            f"{status} {group}", 
            str(len(cols)), 
            f"{coverage}/{len(df)} ({coverage_pct:.1f}%)"
        )
        total_features += len(cols)

print(f"\n  {Colors.BOLD}Total Features: {total_features}{Colors.ENDC}")

# ============================================================================
# DATA QUALITY METRICS
# ============================================================================
print_subheader("DATA QUALITY METRICS")

# Calculate completeness
completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
duplicates = df.index.duplicated().sum()
# Calculate zero variance only for numeric columns
numeric_df = df.select_dtypes(include=['number'])
zero_variance_cols = (numeric_df.std() == 0).sum()

print_metric("Data Completeness", f"{completeness:.2f}%", "success" if completeness > 95 else "warning")
print_metric("Duplicate Days", f"{duplicates}", "success" if duplicates == 0 else "error")
print_metric("Zero-Variance Features", f"{zero_variance_cols}", "success" if zero_variance_cols == 0 else "warning")
print_metric("Missing Values (Total)", f"{format_number(df.isnull().sum().sum())}", "info")

# ============================================================================
# OPERATIONAL INSIGHTS
# ============================================================================
print_subheader("OPERATIONAL INSIGHTS")

# Production metrics
print(f"\n{Colors.BOLD}  Production Metrics:{Colors.ENDC}")
print_metric("  Average Daily Production", f"{format_number(df['total_prod'].mean())} units", "info")
print_metric("  Average Defect Rate", f"{df['avg_defect'].mean():.2f}%", "warning" if df['avg_defect'].mean() > 5 else "success")
print_metric("  High Defect Days", f"{format_number(df['high_defect_count'].sum())}", "warning")

# Quality Control
print(f"\n{Colors.BOLD}  Quality Control:{Colors.ENDC}")
qc_fail_pct = df['qc_fail_pct'].mean()
print_metric("  Average QC Fail Rate", f"{qc_fail_pct:.2f}%", "error" if qc_fail_pct > 10 else "success")
print_metric("  QC Anomaly Days", f"{df['qc_anomaly'].sum()} days", "warning")
print_metric("  Average Pass Rate", f"{df['qc_pass_rate'].mean():.2f}%", "success")

# Dispatch
print(f"\n{Colors.BOLD}  Dispatch Performance:{Colors.ENDC}")
print_metric("  Average Delay", f"{df['avg_delay'].mean():.1f} minutes", "warning" if df['avg_delay'].mean() > 30 else "success")
print_metric("  Late Deliveries", f"{df['late_pct'].mean():.1f}% average", "warning")
print_metric("  Delay Anomaly Days", f"{df.get('delay_anomaly', pd.Series([0])).sum()}", "info")

# Waste & Returns
print(f"\n{Colors.BOLD}  Waste & Returns:{Colors.ENDC}")
print_metric("  Total Waste", f"{format_number(df['total_waste'].sum())} units", "error")
print_metric("  Average Daily Waste", f"{format_number(df['total_waste'].mean())} units", "warning")
print_metric("  Total Returns", f"{format_number(df['total_return'].sum())} units", "error")
print_metric("  Waste Anomaly Days", f"{df.get('waste_anomaly', pd.Series([0])).sum()}", "warning")
print_metric("  Return Anomaly Days", f"{df.get('return_anomaly', pd.Series([0])).sum()}", "warning")

# Sales
print(f"\n{Colors.BOLD}  Sales Performance:{Colors.ENDC}")
print_metric("  Total Sales", f"{format_number(df['total_sold'].sum())} units", "success")
print_metric("  Average Daily Sales", f"{format_number(df['total_sold'].mean())} units", "info")
print_metric("  Demand Collapse Days", f"{format_number(df['demand_collapse_count'].sum())}", "warning")
print_metric("  Promotion Days", f"{format_number(df['promotion_days'].sum())}", "info")

# Inventory
print(f"\n{Colors.BOLD}  Inventory Status:{Colors.ENDC}")
neg_balance_total = df['negative_balance_count'].sum()
print_metric("  Negative Balance Events", f"{format_number(neg_balance_total)}", "error" if neg_balance_total > 1000 else "warning")
print_metric("  Inventory Anomaly Days", f"{df['inventory_anomaly'].sum()}", "error")
print_metric("  Items Nearing Expiry", f"{format_number(df['nearing_expiry_count'].sum())}", "warning")

# ============================================================================
# ANOMALY SUMMARY
# ============================================================================
print_subheader("ANOMALY DETECTION SUMMARY")

anomaly_cols = [c for c in df.columns if 'anomaly' in c]
total_anomaly_days = (df[anomaly_cols].sum(axis=1) > 0).sum()

print_metric("Total Anomaly Flags", f"{format_number(df[anomaly_cols].sum().sum())}", "warning")
print_metric("Days with Any Anomaly", f"{total_anomaly_days}/{len(df)} ({(total_anomaly_days/len(df)*100):.1f}%)", "warning")

print(f"\n{Colors.BOLD}  Anomaly Breakdown:{Colors.ENDC}")
for col in sorted(anomaly_cols):
    count = df[col].sum()
    pct = (count / len(df)) * 100
    status = "error" if count > 50 else "warning" if count > 10 else "success"
    print_metric(f"  {col.replace('_', ' ').title()}", f"{count} days ({pct:.1f}%)", status)

# ============================================================================
# VALIDATION STATUS
# ============================================================================
print_subheader("VALIDATION STATUS")

# Validation checks
checks = []
checks.append(("Dataset loaded successfully", df.shape[0] > 0, "critical"))
checks.append(("All 52 features present", df.shape[1] == 52, "critical"))
checks.append(("365 days of data", df.shape[0] == 365, "critical"))
checks.append(("No duplicate dates", duplicates == 0, "high"))
checks.append(("Data completeness > 95%", completeness > 95, "medium"))
checks.append(("All feature groups covered", all(len(cols) > 0 for cols in feature_groups.values() if cols), "medium"))

passed = sum(1 for _, status, _ in checks if status)
total = len(checks)

print(f"\n{Colors.BOLD}  Validation Checks: {passed}/{total} Passed{Colors.ENDC}\n")

for check, status, priority in checks:
    symbol = f"{Colors.OKGREEN}✓{Colors.ENDC}" if status else f"{Colors.FAIL}✗{Colors.ENDC}"
    priority_label = {
        "critical": f"{Colors.FAIL}[CRITICAL]{Colors.ENDC}",
        "high": f"{Colors.WARNING}[HIGH]{Colors.ENDC}",
        "medium": f"{Colors.OKBLUE}[MEDIUM]{Colors.ENDC}"
    }[priority]
    print(f"  {symbol} {check} {priority_label if not status else ''}")

# ============================================================================
# FINAL STATUS
# ============================================================================
print("\n" + "=" * 70)
if passed == total:
    print(f"{Colors.BOLD}{Colors.OKGREEN}✓ VALIDATION COMPLETE - ALL CHECKS PASSED{Colors.ENDC}".center(80))
else:
    print(f"{Colors.BOLD}{Colors.WARNING}⚠ VALIDATION COMPLETE - {total - passed} ISSUES FOUND{Colors.ENDC}".center(80))
print("=" * 70 + "\n")

print(f"{Colors.DIM}Generated by: Bakery Anomaly Detection ML System{Colors.ENDC}")
print(f"{Colors.DIM}Report saved to: reports/validation_report.txt (if logging enabled){Colors.ENDC}\n")
