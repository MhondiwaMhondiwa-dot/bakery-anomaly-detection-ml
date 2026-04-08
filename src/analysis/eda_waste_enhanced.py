"""
Exploratory Data Analysis for Waste Dataset

This script analyzes the FINAL LOSS dataset - products destroyed or discarded
at production or post-dispatch stages. Waste represents direct financial loss
and is the primary target for prediction and prevention.

Key Analyses:
- Waste by stage (production vs post-dispatch)
- Root cause analysis (waste reasons)
- SKU-level waste patterns
- Shift performance and temporal patterns
- Temperature correlation with spoilage
- Batch traceability (linking to production/QC)
- Route and retailer-level post-dispatch waste
- Handling condition impact

Author: Baker's Inn Analytics Team
Date: 2025
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(message)s'
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
SUMMARIES_DIR = REPORTS_DIR / 'summaries'

# Create output directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare():
    """
    Load waste dataset and prepare derived fields.
    
    Returns:
        pd.DataFrame: Waste data with derived fields
    """
    # Load waste dataset from raw folder (original CSV with correct schema)
    waste_csv = BASE_DIR / 'data' / 'raw' / 'waste_dataset.csv'
    df = pd.read_csv(waste_csv)
    logging.info(f"Loaded {len(df):,} waste records")

    # Use quantity_wasted column directly (this is the correct column name)
    if 'quantity_wasted' in df.columns:
        df['qty_waste'] = pd.to_numeric(df['quantity_wasted'], errors='coerce').fillna(0)
    else:
        # Fallback: sum multi-SKU columns if present
        sku_cols = [c for c in ['Soft white','High Energy Brown','Whole grain loaf','Low GI Seed loaf'] if c in df.columns]
        if sku_cols:
            df['qty_waste'] = pd.to_numeric(df[sku_cols].sum(axis=1), errors='coerce').fillna(0)
        else:
            df['qty_waste'] = 0
    
    # Derive stage from waste_reason semantics:
    # - 'Quality Failure' and 'Damaged' occur at production/plant level
    # - 'Expired Waste' and 'Returned Unsold' occur after product leaves the plant
    # (route_id is present on ALL records so cannot be used as the split signal)
    if 'waste_reason' in df.columns:
        production_reasons = {'Quality Failure', 'Damaged'}
        df['stage'] = df['waste_reason'].apply(
            lambda r: 'production' if r in production_reasons else 'post_dispatch'
        )
    elif 'route_id' in df.columns:
        df['stage'] = df['route_id'].apply(lambda x: 'post_dispatch' if pd.notna(x) else 'production')
    else:
        df['stage'] = 'unknown'
    
    # Use waste_reason directly (not waste_reason_code)
    if 'waste_reason' in df.columns:
        df['waste_reason_code'] = df['waste_reason']

    # Parse timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Derive time-based features only if timestamp parsed successfully
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Calculate waste rate (assuming nominal batch size for context)
    # This is a proxy - in real scenario, compare to production volumes

    return df

def summary_stats(df):
    """
    Generate comprehensive summary statistics for waste dataset.
    
    Args:
        df: Waste DataFrame
    """
    summary_path = REPORTS_DIR / 'waste_enhanced_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WASTE DATASET - EXPLORATORY DATA ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset overview
        f.write("🗑️ DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Waste Records: {len(df):,}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Total Units Wasted: {df['qty_waste'].sum():,}\n")
        f.write(f"Average Waste per Incident: {df['qty_waste'].mean():.1f} units\n")
        f.write(f"Median Waste: {df['qty_waste'].median():.0f} units\n")
        f.write(f"Plants: {df['plant_id'].nunique()}\n")
        if 'sku' in df.columns:
            f.write(f"SKUs Affected: {df['sku'].nunique()}\n")
        else:
            sku_cols = [c for c in ['soft_white','high_energy_brown','whole_grain_loaf','low_gi_seed_loaf'] if c in df.columns]
            f.write(f"SKUs Affected (columns): {len(sku_cols)}\n")
        f.write(f"Unique Batches: {df['batch_id'].nunique()}\n\n")
        
        # CRITICAL: Waste by stage
        f.write("🏭 WASTE BY STAGE (Production vs Post-Dispatch)\n")
        f.write("-" * 80 + "\n")
        stage_waste = df.groupby('stage').agg({
            'qty_waste': ['sum', 'mean', 'count'],
            'waste_id': 'count'
        })
        stage_waste.columns = ['Total Units', 'Avg Units/Incident', 'Incidents', 'Record Count']
        stage_pct = (stage_waste['Total Units'] / stage_waste['Total Units'].sum() * 100).round(1)
        stage_waste['% of Total'] = stage_pct
        f.write(f"{stage_waste}\n\n")
        
        prod_waste = df[df['stage'] == 'production']['qty_waste'].sum()
        post_waste = df[df['stage'] == 'post_dispatch']['qty_waste'].sum()
        
        if prod_waste > post_waste:
            f.write(f"⚠️ **PRODUCTION WASTE DOMINANT:** {prod_waste:,} units ({prod_waste/(prod_waste+post_waste)*100:.1f}%)\n")
            f.write("   Interpretation: Manufacturing defects, quality issues, or process inefficiencies are the primary waste driver\n")
            f.write("   Action Items:\n")
            f.write("   - Review production line equipment maintenance schedules\n")
            f.write("   - Analyze batch sizing vs demand patterns\n")
            f.write("   - Implement tighter quality control checkpoints\n")
            f.write("   - Investigate waste_reason codes in production stages\n\n")
        else:
            f.write(f"⚠️ **POST-DISPATCH WASTE DOMINANT:** {post_waste:,} units ({post_waste/(prod_waste+post_waste)*100:.1f}%)\n")
            f.write("   Interpretation: Products are being wasted after leaving the plant (handling, spoilage, unsold returns)\n")
            f.write("   Action Items:\n")
            f.write("   - Audit cold chain compliance at depots and during transport\n")
            f.write("   - Review demand forecasting accuracy to reduce overstocking\n")
            f.write("   - Examine handling_condition data for rough handling patterns\n")
            f.write("   - Analyze returned unsold products by route/depot\n\n")
        
        # Root cause analysis
        f.write("🔍 WASTE ROOT CAUSE ANALYSIS (Top 10 Reasons)\n")
        f.write("-" * 80 + "\n")
        reason_waste = df.groupby('waste_reason_code').agg({
            'qty_waste': 'sum',
            'waste_id': 'count'
        }).sort_values('qty_waste', ascending=False).head(10)
        reason_waste.columns = ['Total Units Wasted', 'Incidents']
        reason_waste['% of Total'] = (reason_waste['Total Units Wasted'] / df['qty_waste'].sum() * 100).round(2)
        f.write(f"{reason_waste}\n\n")
        
        top_reason = reason_waste.index[0]
        top_reason_pct = reason_waste.iloc[0]['% of Total']
        top_reason_incidents = reason_waste.iloc[0]['Incidents']
        f.write(f"🎯 **TOP WASTE REASON:** {top_reason}\n")
        f.write(f"   - Accounts for {top_reason_pct:.1f}% of all waste ({reason_waste.iloc[0]['Total Units Wasted']:,.0f} units)\n")
        f.write(f"   - Responsible for {int(top_reason_incidents):,} waste incidents\n")
        f.write(f"   - Interpretation: This is the single highest-impact waste driver in the supply chain\n")
        f.write(f"   - Priority action: Immediate root cause analysis and corrective measures for {top_reason}\n")
        f.write(f"     → Review all waste_by_reason.csv entries to prioritize prevention efforts\n\n")
        
        # SKU performance (using multi-SKU columns)
        f.write("🍞 SKU-LEVEL WASTE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        sku_cols = [c for c in ['Soft white','High Energy Brown','Whole grain loaf','Low GI Seed loaf'] if c in df.columns]
        if sku_cols:
            sku_waste_data = []
            for sku_col in sku_cols:
                sku_total = pd.to_numeric(df[sku_col], errors='coerce').sum()
                sku_incidents = (pd.to_numeric(df[sku_col], errors='coerce') > 0).sum()
                sku_waste_data.append({
                    'SKU': sku_col,
                    'Total Wasted': sku_total,
                    'Incidents': sku_incidents,
                    '% of Total': (sku_total / df['qty_waste'].sum() * 100) if df['qty_waste'].sum() > 0 else 0
                })
            sku_waste = pd.DataFrame(sku_waste_data).sort_values('Total Wasted', ascending=False)
            f.write(f"{sku_waste}\n\n")
            
            if len(sku_waste) > 0:
                top_sku = sku_waste.iloc[0]['SKU']
                top_sku_waste = sku_waste.iloc[0]['Total Wasted']
                top_sku_pct = sku_waste.iloc[0]['% of Total']
                f.write(f"🔴 **HIGHEST-WASTE SKU:** {top_sku}\n")
                f.write(f"   - Total wasted: {top_sku_waste:,.0f} units ({top_sku_pct:.1f}% of total waste)\n")
                f.write(f"   - Affected incidents: {int(sku_waste.iloc[0]['Incidents']):,}\n")
                f.write(f"   - Interpretation: This SKU has demand forecasting or handling issues\n")
                f.write(f"   - Actions: Check shelf-life, analyze temperature sensitivity, review demand patterns\n\n")
            
            # Identify high-waste SKUs
            high_waste_skus = sku_waste[sku_waste['% of Total'] > 10]
            f.write(f"⚠️ High-Waste SKUs (>10% of total waste): {len(high_waste_skus)}\n")
            if len(high_waste_skus) > 0:
                f.write(f"{high_waste_skus[['SKU', 'Total Wasted', '% of Total']]}\n")
                f.write("   → Focus prevention efforts on these SKUs first (80/20 rule)\n")
        else:
            f.write("No SKU columns found in dataset\n")
        f.write("\n")
        
        # Plant-level waste
        f.write("🏭 PLANT-LEVEL WASTE PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        plant_waste = df.groupby('plant_id').agg({
            'qty_waste': ['sum', 'mean', 'count']
        }).sort_values(('qty_waste', 'sum'), ascending=False)
        plant_waste.columns = ['Total Wasted', 'Avg per Incident', 'Incidents']
        plant_waste['% of Total'] = (plant_waste['Total Wasted'] / df['qty_waste'].sum() * 100).round(2)
        f.write(f"{plant_waste}\n\n")
        
        # Shift analysis
        f.write("⏰ SHIFT-LEVEL WASTE PATTERNS\n")
        f.write("-" * 80 + "\n")
        shift_waste = df.groupby('shift').agg({
            'qty_waste': ['sum', 'mean', 'count']
        }).sort_values(('qty_waste', 'sum'), ascending=False)
        shift_waste.columns = ['Total Wasted', 'Avg per Incident', 'Incidents']
        shift_waste['% of Total'] = (shift_waste['Total Wasted'] / df['qty_waste'].sum() * 100).round(2)
        f.write(f"{shift_waste}\n\n")
        
        worst_shift = shift_waste.index[0]
        worst_shift_pct = shift_waste.iloc[0]['% of Total']
        worst_shift_avg = shift_waste.iloc[0]['Avg per Incident']
        f.write(f"⚠️ **WORST PERFORMING SHIFT:** {worst_shift}\n")
        f.write(f"   - Accounts for {worst_shift_pct:.1f}% of total waste\n")
        f.write(f"   - Average waste per incident: {worst_shift_avg:.1f} units (higher wastage per event)\n")
        f.write(f"   - Interpretation: This shift has operational challenges (staffing, training, fatigue, supervision)\n")
        f.write(f"   - Actions: Review shift procedures, audit staffing levels, conduct retraining, improve supervision\n\n")
        
        # Temperature analysis (optional)
        f.write("🌡️ TEMPERATURE CORRELATION WITH WASTE\n")
        f.write("-" * 80 + "\n")
        if 'temperature_at_check' in df.columns:
            temp_stats = df['temperature_at_check'].describe()
            f.write(f"{temp_stats}\n\n")

            # High temperature waste
            high_temp_threshold = 35  # Celsius
            high_temp_waste = df[df['temperature_at_check'] > high_temp_threshold]
            f.write(f"Waste incidents with temp > {high_temp_threshold}°C: {len(high_temp_waste):,} ({len(high_temp_waste)/len(df)*100:.1f}%)\n")
            f.write(f"Units wasted at high temp: {high_temp_waste['qty_waste'].sum():,} ({high_temp_waste['qty_waste'].sum()/df['qty_waste'].sum()*100:.1f}% of total)\n\n")
        else:
            f.write("Temperature data not available. Skipping temperature analysis.\n\n")
        
        # Handling condition (optional)
        f.write("🤲 HANDLING CONDITION IMPACT\n")
        f.write("-" * 80 + "\n")
        if 'handling_condition' in df.columns:
            handling_waste = df.groupby('handling_condition').agg({
                'qty_waste': ['sum', 'count']
            }).sort_values(('qty_waste', 'sum'), ascending=False)
            handling_waste.columns = ['Total Wasted', 'Incidents']
            handling_waste['% of Total'] = (handling_waste['Total Wasted'] / df['qty_waste'].sum() * 100).round(2)
            f.write(f"{handling_waste}\n\n")
        else:
            f.write("Handling condition data not available. Skipping section.\n\n")
        
        # Temporal patterns
        f.write("📅 TEMPORAL WASTE PATTERNS\n")
        f.write("-" * 80 + "\n")
        
        # Day of week
        dow_waste = df.groupby('day_name').agg({
            'qty_waste': ['sum', 'mean', 'count']
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        dow_waste.columns = ['Total', 'Avg', 'Incidents']
        f.write("Waste by Day of Week:\n")
        f.write(f"{dow_waste}\n\n")
        
        # Weekend vs weekday
        weekend_waste = df.groupby('is_weekend').agg({
            'qty_waste': ['sum', 'mean']
        })
        weekend_waste.index = ['Weekday', 'Weekend']
        f.write("Weekday vs Weekend:\n")
        f.write(f"{weekend_waste}\n\n")
        
        # Hourly patterns
        hourly_waste = df.groupby('hour')['qty_waste'].agg(['sum', 'mean', 'count']).round(1)
        peak_hour = hourly_waste['sum'].idxmax()
        f.write(f"Peak Waste Hour: {peak_hour}:00 ({hourly_waste.loc[peak_hour, 'sum']:.0f} units)\n\n")
        
        # Post-dispatch waste specifics
        post_dispatch_df = df[df['stage'] == 'post_dispatch']
        if len(post_dispatch_df) > 0:
            f.write("🚛 POST-DISPATCH WASTE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Post-Dispatch Waste: {len(post_dispatch_df):,} incidents, {post_dispatch_df['qty_waste'].sum():,} units\n\n")
            
            # Route-level waste
            if 'route_id' in post_dispatch_df.columns:
                route_waste = post_dispatch_df.groupby('route_id').agg({
                    'qty_waste': ['sum', 'count']
                }).sort_values(('qty_waste', 'sum'), ascending=False).head(15)
                route_waste.columns = ['Total Wasted', 'Incidents']
                f.write("Top 15 Routes by Waste:\n")
                f.write(f"{route_waste}\n\n")
            
            # Depot-level waste (retailer_id not in dataset, use depot_id)
            if 'depot_id' in post_dispatch_df.columns:
                depot_waste = post_dispatch_df.groupby('depot_id').agg({
                    'qty_waste': ['sum', 'count']
                }).sort_values(('qty_waste', 'sum'), ascending=False).head(15)
                depot_waste.columns = ['Total Wasted', 'Incidents']
                f.write("Top 15 Depots by Waste:\n")
                f.write(f"{depot_waste}\n\n")
        
        # Batch traceability
        f.write("🔗 BATCH TRACEABILITY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Unique batches with waste: {df['batch_id'].nunique():,}\n")
        batch_waste = df.groupby('batch_id')['qty_waste'].sum().sort_values(ascending=False).head(10)
        f.write("Top 10 batches by waste quantity:\n")
        f.write(f"{batch_waste}\n\n")
        
        # ACTION ITEMS
        f.write("🎯 KEY INSIGHTS & ACTION ITEMS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 QUANTITATIVE SUMMARY:\n")
        f.write(f"   • Total waste volume: {df['qty_waste'].sum():,} units across {len(df):,} incidents\n")
        f.write(f"   • Average waste per incident: {df['qty_waste'].mean():.1f} units\n")
        f.write(f"   • Financial impact: CRITICAL - waste represents direct financial loss\n")
        f.write(f"   • Date range analyzed: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}\n\n")
        
        f.write(f"🎯 PRIMARY FOCUS AREA:\n")
        if prod_waste > post_waste:
            f.write(f"   → PRODUCTION WASTE (>50% of total)\n")
            f.write(f"   • Total: {prod_waste:,} units\n")
            f.write(f"   • Root causes likely: Equipment failures, quality defects, batch sizing\n")
        else:
            f.write(f"   → POST-DISPATCH WASTE (>50% of total)\n")
            f.write(f"   • Total: {post_waste:,} units\n")
            f.write(f"   • Root causes likely: Cold chain breaks, handling damage, shelf-life expiry\n\n")
        
        f.write(f"🔍 SPECIFIC ACTION ITEMS (PRIORITY ORDER):\n\n")
        f.write(f"1. **TOP WASTE REASON - {top_reason.upper()}** ({top_reason_pct:.1f}%)\n")
        f.write(f"   → Impact: {reason_waste.iloc[0]['Total Units Wasted']:,.0f} units wasted\n")
        f.write(f"   → Frequency: {int(top_reason_incidents):,} incidents\n")
        f.write(f"   Actions:\n")
        f.write(f"      - Conduct immediate root cause analysis (RCA)\n")
        f.write(f"      - Identify systemic failures preventing this waste\n")
        f.write(f"      - Implement preventive controls\n")
        f.write(f"      - Track prevention effectiveness weekly\n\n")
        
        f.write(f"2. **HIGH-WASTE SKUS** (>10% of total)\n")
        if len(high_waste_skus) > 0:
            f.write(f"   → Products at risk: {', '.join(high_waste_skus['SKU'].values)}\n")
        f.write(f"   Actions:\n")
        f.write(f"      - Review demand forecasting accuracy for these SKUs\n")
        f.write(f"      - Analyze shelf-life vs. inventory turnover\n")
        f.write(f"      - Check temperature control compliance\n")
        f.write(f"      - Consider supply chain optimization\n\n")
        
        f.write(f"3. **SHIFT PERFORMANCE** ({worst_shift})\n")
        f.write(f"   → Excess waste: {worst_shift_pct:.1f}% of total\n")
        f.write(f"   Actions:\n")
        f.write(f"      - Audit staffing levels and skill mix\n")
        f.write(f"      - Review operational procedures during this shift\n")
        f.write(f"      - Conduct targeted training/retraining\n")
        f.write(f"      - Increase supervision/monitoring\n\n")
        
        f.write(f"4. **LOGISTICAL HOTSPOTS** (Top 15 routes/depots)\n")
        f.write(f"   Actions:\n")
        f.write(f"      - Audit top routes for cold chain compliance\n")
        f.write(f"      - Inspect depot equipment and storage conditions\n")
        f.write(f"      - Review handling procedures with drivers/depot staff\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("📋 NEXT STEPS:\n")
        f.write("   1. Review waste_enhanced_summary.txt (this report) with operations team\n")
        f.write("   2. Examine waste_by_reason.csv to identify specific failure modes\n")
        f.write("   3. Cross-reference waste_by_route_top30.csv with route performance metrics\n")
        f.write("   4. Implement corrective actions for top 3 waste drivers\n")
        f.write("   5. Track KPIs weekly: Total waste units, waste per incident, waste by reason\n")
        f.write("   6. Re-run this analysis monthly to track progress\n")
        f.write("=" * 80 + "\n")
        f.write("✅ Waste summary complete!\n")
    
    logging.info(f"Wrote {summary_path}")

def grouped_summaries(df):
    """
    Generate grouped summary CSV files.
    
    Args:
        df: Waste DataFrame
    """
    # 1. Waste by stage
    stage_summary = df.groupby('stage').agg({
        'qty_waste': ['sum', 'mean', 'count'],
        'temperature_at_check': 'mean'
    }).round(2)
    stage_summary.columns = ['_'.join(col).strip() for col in stage_summary.columns]
    stage_summary.to_csv(SUMMARIES_DIR / 'waste_by_stage.csv')
    logging.info("Wrote waste_by_stage.csv")
    
    # 2. Waste by reason
    reason_summary = df.groupby('waste_reason_code').agg({
        'qty_waste': ['sum', 'mean', 'count']
    }).round(2)
    reason_summary.columns = ['_'.join(col).strip() for col in reason_summary.columns]
    reason_summary = reason_summary.sort_values('qty_waste_sum', ascending=False)
    reason_summary.to_csv(SUMMARIES_DIR / 'waste_by_reason.csv')
    logging.info("Wrote waste_by_reason.csv")
    
    # 3. Waste by SKU (using multi-SKU columns)
    sku_cols = [c for c in ['Soft white','High Energy Brown','Whole grain loaf','Low GI Seed loaf'] if c in df.columns]
    if sku_cols:
        sku_waste_data = []
        for sku_col in sku_cols:
            sku_total = pd.to_numeric(df[sku_col], errors='coerce').sum()
            sku_mean = pd.to_numeric(df[sku_col], errors='coerce').mean()
            sku_count = (pd.to_numeric(df[sku_col], errors='coerce') > 0).sum()
            sku_waste_data.append({
                'sku': sku_col,
                'qty_waste_sum': sku_total,
                'qty_waste_mean': sku_mean,
                'qty_waste_count': sku_count
            })
        sku_summary = pd.DataFrame(sku_waste_data).sort_values('qty_waste_sum', ascending=False)
        sku_summary.to_csv(SUMMARIES_DIR / 'waste_by_sku.csv', index=False)
        logging.info("Wrote waste_by_sku.csv")
    
    # 4. Waste by plant
    plant_summary = df.groupby('plant_id').agg({
        'qty_waste': ['sum', 'mean', 'count']
    }).round(2)
    plant_summary.columns = ['_'.join(col).strip() for col in plant_summary.columns]
    plant_summary = plant_summary.sort_values('qty_waste_sum', ascending=False)
    plant_summary.to_csv(SUMMARIES_DIR / 'waste_by_plant.csv')
    logging.info("Wrote waste_by_plant.csv")
    
    # 5. Waste by shift
    shift_summary = df.groupby('shift').agg({
        'qty_waste': ['sum', 'mean', 'count']
    }).round(2)
    shift_summary.columns = ['_'.join(col).strip() for col in shift_summary.columns]
    shift_summary = shift_summary.sort_values('qty_waste_sum', ascending=False)
    shift_summary.to_csv(SUMMARIES_DIR / 'waste_by_shift.csv')
    logging.info("Wrote waste_by_shift.csv")
    
    # 6. Waste by handling condition
    handling_summary = df.groupby('handling_condition').agg({
        'qty_waste': ['sum', 'mean', 'count'],
        'temperature_at_check': 'mean'
    }).round(2)
    handling_summary.columns = ['_'.join(col).strip() for col in handling_summary.columns]
    handling_summary = handling_summary.sort_values('qty_waste_sum', ascending=False)
    handling_summary.to_csv(SUMMARIES_DIR / 'waste_by_handling.csv')
    logging.info("Wrote waste_by_handling.csv")
    
    # 7. Post-dispatch waste by route (top 30)
    post_dispatch = df[df['stage'] == 'post_dispatch']
    if len(post_dispatch) > 0:
        route_summary = post_dispatch.groupby('route_id').agg({
            'qty_waste': ['sum', 'mean', 'count'],
            'temperature_at_check': 'mean'
        }).round(2)
        route_summary.columns = ['_'.join(col).strip() for col in route_summary.columns]
        route_summary = route_summary.sort_values('qty_waste_sum', ascending=False).head(30)
        route_summary.to_csv(SUMMARIES_DIR / 'waste_by_route_top30.csv')
        logging.info("Wrote waste_by_route_top30.csv")
        
        # 8. Post-dispatch waste by depot (top 30) - use depot_id instead of retailer_id
        if 'depot_id' in post_dispatch.columns:
            depot_summary = post_dispatch.groupby('depot_id').agg({
                'qty_waste': ['sum', 'mean', 'count'],
                'temperature_at_check': 'mean' if 'temperature_at_check' in post_dispatch.columns else 'count'
            }).round(2)
            depot_summary.columns = ['_'.join(col).strip() for col in depot_summary.columns]
            depot_summary = depot_summary.sort_values('qty_waste_sum', ascending=False).head(30)
            depot_summary.to_csv(SUMMARIES_DIR / 'waste_by_depot_top30.csv')
            logging.info("Wrote waste_by_depot_top30.csv")

def visualizations(df):
    """
    Generate comprehensive visualizations for waste dataset.
    
    Args:
        df: Waste DataFrame
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Waste by stage (production vs post-dispatch)
    fig, ax = plt.subplots(figsize=(10, 6))
    stage_waste = df.groupby('stage')['qty_waste'].sum().sort_values(ascending=False)
    colors = ['#d62728' if i == 0 else '#ff7f0e' for i in range(len(stage_waste))]
    stage_waste.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Total Waste by Stage (Production vs Post-Dispatch)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Units Wasted', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(stage_waste.values):
        pct = v / stage_waste.sum() * 100
        ax.text(i, v, f'{v:,.0f}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_by_stage.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_by_stage.png")
    
    # 2. Waste by reason (top 10)
    fig, ax = plt.subplots(figsize=(12, 7))
    reason_waste = df.groupby('waste_reason_code')['qty_waste'].sum().sort_values(ascending=True).tail(10)
    colors = ['red' if x > reason_waste.median() * 1.5 else 'orange' if x > reason_waste.median() else 'gold' for x in reason_waste]
    reason_waste.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Top 10 Waste Reasons by Volume', fontsize=16, fontweight='bold')
    ax.set_xlabel('Units Wasted', fontsize=12)
    ax.set_ylabel('Waste Reason', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_by_reason_top10.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_by_reason_top10.png")
    
    # 3. Waste by SKU (using multi-SKU columns)
    fig, ax = plt.subplots(figsize=(12, 7))
    sku_cols = [c for c in ['Soft white','High Energy Brown','Whole grain loaf','Low GI Seed loaf'] if c in df.columns]
    if sku_cols:
        sku_waste_data = {sku_col: pd.to_numeric(df[sku_col], errors='coerce').sum() for sku_col in sku_cols}
        sku_waste = pd.Series(sku_waste_data).sort_values(ascending=True)
        colors = ['darkred' if x > sku_waste.median() * 1.5 else 'coral' for x in sku_waste]
        sku_waste.plot(kind='barh', ax=ax, color=colors)
        ax.set_title('Waste Volume by SKU', fontsize=16, fontweight='bold')
        ax.set_xlabel('Units Wasted', fontsize=12)
        ax.set_ylabel('SKU', fontsize=12)
        ax.axvline(sku_waste.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {sku_waste.median():.0f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'waste_by_sku.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved waste_by_sku.png")
    else:
        logging.warning("No SKU columns found, skipping waste_by_sku.png")
    
    # 4. Daily waste trend
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_waste = df.groupby('date')['qty_waste'].sum().reset_index()
    daily_waste['date'] = pd.to_datetime(daily_waste['date'])
    
    ax.plot(daily_waste['date'], daily_waste['qty_waste'], linewidth=2, color='darkred', alpha=0.7)
    ax.fill_between(daily_waste['date'], daily_waste['qty_waste'], alpha=0.3, color='salmon')
    
    # 7-day moving average
    daily_waste['ma7'] = daily_waste['qty_waste'].rolling(window=7, center=True).mean()
    ax.plot(daily_waste['date'], daily_waste['ma7'], linewidth=3, color='darkblue', label='7-Day Moving Avg')
    
    ax.set_title('Daily Waste Trend', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Units Wasted', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_daily_trend.png")
    
    # 5. Waste by shift
    fig, ax = plt.subplots(figsize=(10, 6))
    shift_waste = df.groupby('shift')['qty_waste'].sum().sort_values(ascending=False)
    colors = ['#d62728', '#ff7f0e', '#2ca02c'][:len(shift_waste)]
    shift_waste.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Waste by Shift', fontsize=16, fontweight='bold')
    ax.set_xlabel('Shift', fontsize=12)
    ax.set_ylabel('Units Wasted', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(shift_waste.values):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_by_shift.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_by_shift.png")
    
    # 6. Temperature distribution with waste
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['temperature_at_check'].dropna(), bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(35, color='red', linestyle='--', linewidth=2, label='High Temp Threshold (35°C)')
    ax.set_title('Temperature Distribution at Waste Check', fontsize=16, fontweight='bold')
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_temperature_distribution.png")
    
    # 7. Waste by handling condition
    fig, ax = plt.subplots(figsize=(10, 6))
    handling_waste = df.groupby('handling_condition')['qty_waste'].sum().sort_values(ascending=False)
    handling_waste.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Waste by Handling Condition', fontsize=16, fontweight='bold')
    ax.set_xlabel('Handling Condition', fontsize=12)
    ax.set_ylabel('Units Wasted', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(handling_waste.values):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_by_handling_condition.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_by_handling_condition.png")
    
    # 8. Day of week pattern
    fig, ax = plt.subplots(figsize=(12, 6))
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_waste = df.groupby('day_name')['qty_waste'].sum().reindex(dow_order)
    colors_dow = ['lightcoral' if day not in ['Saturday', 'Sunday'] else 'darkred' for day in dow_order]
    ax.bar(dow_order, dow_waste.values, color=colors_dow)
    ax.set_title('Waste by Day of Week', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Total Units Wasted', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(dow_waste.values):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_day_of_week.png")
    
    # 9. Post-dispatch: Top routes with waste
    post_dispatch = df[df['stage'] == 'post_dispatch']
    if len(post_dispatch) > 0 and post_dispatch['route_id'].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        route_waste = post_dispatch.groupby('route_id')['qty_waste'].sum().sort_values(ascending=True).tail(15)
        route_waste.plot(kind='barh', ax=ax, color='darkred')
        ax.set_title('Top 15 Routes by Post-Dispatch Waste', fontsize=16, fontweight='bold')
        ax.set_xlabel('Units Wasted', fontsize=12)
        ax.set_ylabel('Route ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'waste_by_route_top15.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved waste_by_route_top15.png")
    
    # 10. Stage breakdown pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    stage_waste = df.groupby('stage')['qty_waste'].sum()
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # Support up to 3 stages
    explode = tuple([0.1 if i == 0 else 0 for i in range(len(stage_waste))])  # Explode first slice only
    ax.pie(stage_waste, labels=stage_waste.index, autopct='%1.1f%%', startangle=90,
           colors=colors[:len(stage_waste)], explode=explode, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Waste Distribution: Production vs Post-Dispatch', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'waste_stage_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved waste_stage_pie.png")

def main():
    """
    Main execution function for Waste EDA.
    """
    # Load and prepare data
    df = load_and_prepare()
    
    # Generate summary statistics
    summary_stats(df)
    
    # Generate grouped summaries
    grouped_summaries(df)
    
    # Generate visualizations
    visualizations(df)
    
    logging.info("✅ Waste EDA complete!")

if __name__ == '__main__':
    main()
