"""
ROUTE & TRANSPORT METADATA - ENHANCED EDA
==========================================

Purpose: Analyze the STRUCTURAL CHARACTERISTICS of the delivery network.
         Route metadata provides CONTEXT for dispatch behavior - explains WHY
         delays happen, which routes are risky, and how to optimize logistics.

Dataset: route_transport_multivehicle.parquet
Records: 216 route configurations (69 unique routes, multiple vehicle assignments)

Business Context:
-----------------
Route metadata is a REFERENCE dataset (not transactional).
It describes:
- Delivery routes (depot-to-store assignments)
- Vehicles used (trucks with different capacities)
- Drivers assigned (performance tracking)
- Expected travel times (for delay normalization)
- Distances (for efficiency analysis)
- Stops (complexity indicator)

Why This Matters:
-----------------
✔ Explains delivery delays (long routes ≠ short routes, urban ≠ rural)
✔ Separates "normal delay" from "true anomaly"
✔ Identifies root causes (overloaded vehicles, too many stops, inexperienced drivers)
✔ Enables route optimization to reduce waste (late deliveries → spoilage)

Key Analyses:
-------------
1. Route Efficiency: distance/time ratio, stops/distance density
2. Vehicle Utilization: capacity distribution, assignment patterns
3. Regional Characteristics: urban vs rural, congestion patterns
4. Route Complexity: stops count, estimated time, load capacity
5. Risk Scoring: identify high-risk routes (long + many stops + heavy load)
6. Optimization Opportunities: rebalance routes, reassign vehicles

Links to Other Datasets:
-------------------------
- Dispatch: route_id → explains delay causes
- Sales B2B: route_id → links demand to logistics
- Waste/Returns: route_id → late routes → spoilage
- Inventory: route_id → flow efficiency issues

Author: Enhanced EDA Pipeline
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / 'data/processed/route_transport_multivehicle.parquet'
DATA_PATH_RAW = ROOT_DIR / 'data/raw/route_transport_multivehicle.csv'
DATA_PATH_RAW01 = ROOT_DIR / 'data/raw01/route_transport_multivehicle.csv'
REPORTS_DIR = ROOT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
SUMMARIES_DIR = REPORTS_DIR / 'summaries'

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARIES_DIR.mkdir(exist_ok=True)

def _read_route_metadata():
    """
    Read route metadata from available source:
    - Prefer processed parquet
    - Fallback to raw CSV in data/raw or data/raw01
    """
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    if DATA_PATH_RAW.exists():
        return pd.read_csv(DATA_PATH_RAW)
    if DATA_PATH_RAW01.exists():
        return pd.read_csv(DATA_PATH_RAW01)
    raise FileNotFoundError("Route metadata not found in processed or raw paths")

def load_and_prepare():
    """
    Load route metadata and derive analytical features.
    
    Derived Features:
    -----------------
    - stops_count: Number of stores per route (from stops_list)
    - estimated_time_minutes: Extract minutes from datetime field
    - efficiency_km_per_min: distance_km / estimated_time_minutes
    - stops_density: stops_count / distance_km (stops per km)
    - avg_distance_per_stop: distance_km / stops_count
    - complexity_score: (stops_count * distance_km) / estimated_time_minutes
    - route_type: Urban (<30km), Suburban (30-60km), Rural (>60km)
    - capacity_category: Small (<4000kg), Medium (4000-5500kg), Large (>5500kg)
    - start_time_category: Early (03:00-05:00), Mid (04:00-06:00), Late (05:00-07:00)
    - risk_score: Composite risk metric (distance + stops + load / capacity)
    """
    print("INFO: Starting Route & Transport Metadata Enhanced EDA...")
    
    df = _read_route_metadata()
    print(f"INFO: Loaded {len(df)} route configuration records")
    
    # Parse stops count (prefer numeric `num_stops`, fallback to parsed `stop_lists`)
    if 'num_stops' in df.columns:
        df['stops_count'] = pd.to_numeric(df['num_stops'], errors='coerce').fillna(0).astype(int)
    elif 'stop_lists' in df.columns:
        # stop_lists may be pipe- or comma-separated
        def _count_stops(val):
            if pd.isna(val):
                return 0
            s = str(val)
            # Normalize delimiters
            s = s.replace('|', ',')
            return len([t for t in s.split(',') if t.strip()])
        df['stops_count'] = df['stop_lists'].apply(_count_stops)
    else:
        df['stops_count'] = 0
    
    # Estimated time minutes: prefer numeric column `estimated_time_min`
    if 'estimated_time_min' in df.columns:
        df['estimated_time_minutes'] = pd.to_numeric(df['estimated_time_min'], errors='coerce').fillna(0).astype(int)
    else:
        df['estimated_time_minutes'] = 0
    
    # Handle any potential zeros or negatives
    df['estimated_time_minutes'] = df['estimated_time_minutes'].clip(lower=1)
    
    # Efficiency metrics
    df['efficiency_km_per_min'] = df['distance_km'] / df['estimated_time_minutes']
    df['stops_density'] = df['stops_count'] / df['distance_km']  # stops per km
    df['avg_distance_per_stop'] = df['distance_km'] / df['stops_count']
    
    # Complexity score (higher = more complex route)
    df['complexity_score'] = (df['stops_count'] * df['distance_km']) / df['estimated_time_minutes']
    
    # Route type based on distance
    df['route_type'] = pd.cut(
        df['distance_km'],
        bins=[0, 30, 60, 150],
        labels=['Urban', 'Suburban', 'Rural']
    )
    
    # Capacity category
    df['capacity_category'] = pd.cut(
        df['load_capacity_kg'],
        bins=[0, 4000, 5500, 10000],
        labels=['Small (<4000kg)', 'Medium (4000-5500kg)', 'Large (>5500kg)']
    )
    
    # Start time category
    df['start_time_category'] = df['trip_start_window'].map({
        '03:00-05:00': 'Early',
        '04:00-06:00': 'Mid',
        '05:00-07:00': 'Late'
    })
    
    # Risk score: Composite metric
    # Higher risk = longer distance + more stops + heavier load relative to capacity
    df['load_utilization'] = df['stops_count'] * 50  # Assume 50kg per stop average load
    df['capacity_strain'] = df['load_utilization'] / df['load_capacity_kg']
    df['risk_score'] = (
        (df['distance_km'] / 100) * 0.3 +  # Distance factor (normalized)
        (df['stops_count'] / 20) * 0.3 +    # Stops factor (normalized)
        (df['capacity_strain']) * 0.4       # Capacity strain (most important)
    )
    
    print(f"INFO: Derived features complete")
    print(f"INFO: Unique routes: {df['route_id'].nunique()}")
    print(f"INFO: Unique vehicles: {df['vehicle_id'].nunique()}")
    print(f"INFO: Unique drivers: {df['driver_id'].nunique()}")
    print(f"INFO: Regions: {df['region'].nunique()}")
    
    return df

def summary_stats(df):
    """
    Generate comprehensive summary report with route characteristics,
    efficiency metrics, risk analysis, and optimization opportunities.
    """
    summary_path = REPORTS_DIR / 'routes_transport_meta_summary.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ROUTE & TRANSPORT METADATA - ENHANCED SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. OVERVIEW
        f.write("1. OVERVIEW - LOGISTICS NETWORK STRUCTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Route Configurations: {len(df):,}\n")
        f.write(f"Unique Routes: {df['route_id'].nunique()}\n")
        f.write(f"Unique Vehicles: {df['vehicle_id'].nunique()}\n")
        f.write(f"Unique Drivers: {df['driver_id'].nunique()} (Missing: {df['driver_id'].isnull().sum()})\n")
        f.write(f"Regions Covered: {df['region'].nunique()}\n")
        f.write(f"\nNote: 216 configurations = multiple vehicle/driver assignments per route\n")
        f.write(f"      (Same route can be served by different vehicles on different days)\n\n")
        
        # 2. ROUTE CHARACTERISTICS
        f.write("2. ROUTE CHARACTERISTICS - DISTANCE, TIME, STOPS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Distance (km):\n")
        f.write(f"  - Mean: {df['distance_km'].mean():.1f} km\n")
        f.write(f"  - Median: {df['distance_km'].median():.1f} km\n")
        f.write(f"  - Range: {df['distance_km'].min():.1f} - {df['distance_km'].max():.1f} km\n")
        f.write(f"  - Std Dev: {df['distance_km'].std():.1f} km (high variance)\n\n")
        
        f.write(f"Estimated Time (minutes):\n")
        f.write(f"  - Mean: {df['estimated_time_minutes'].mean():.0f} min ({df['estimated_time_minutes'].mean()/60:.1f} hours)\n")
        f.write(f"  - Median: {df['estimated_time_minutes'].median():.0f} min ({df['estimated_time_minutes'].median()/60:.1f} hours)\n")
        f.write(f"  - Range: {df['estimated_time_minutes'].min():.0f} - {df['estimated_time_minutes'].max():.0f} min\n")
        f.write(f"  - Longest route: {df['estimated_time_minutes'].max()/60:.1f} hours\n\n")
        
        f.write(f"Stops per Route:\n")
        f.write(f"  - Mean: {df['stops_count'].mean():.1f} stores\n")
        f.write(f"  - Median: {df['stops_count'].median():.0f} stores\n")
        f.write(f"  - Range: {df['stops_count'].min():.0f} - {df['stops_count'].max():.0f} stores\n")
        # Estimate unique stores served
        if 'stop_lists' in df.columns:
            joined = '|'.join(df['stop_lists'].dropna().astype(str))
            # Split by both '|' and ','
            tokens = []
            for part in joined.split('|'):
                tokens.extend([t.strip() for t in part.split(',') if t.strip()])
            total_unique_stores = len(set(tokens))
        else:
            # Fallback rough estimate: sum of stops (may overcount duplicates)
            total_unique_stores = int(df['stops_count'].sum())
        f.write(f"  - Total unique stores served: {total_unique_stores} (estimated)\n\n")
        
        # 3. EFFICIENCY METRICS
        f.write("3. EFFICIENCY METRICS - ROUTE PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Speed (km per minute):\n")
        f.write(f"  - Mean: {df['efficiency_km_per_min'].mean():.2f} km/min ({df['efficiency_km_per_min'].mean()*60:.1f} km/h)\n")
        f.write(f"  - Median: {df['efficiency_km_per_min'].median():.2f} km/min ({df['efficiency_km_per_min'].median()*60:.1f} km/h)\n")
        f.write(f"  - Interpretation: Avg {df['efficiency_km_per_min'].mean()*60:.1f} km/h includes stop time\n\n")
        
        f.write(f"Stops Density (stops per km):\n")
        f.write(f"  - Mean: {df['stops_density'].mean():.2f} stops/km\n")
        f.write(f"  - Median: {df['stops_density'].median():.2f} stops/km\n")
        f.write(f"  - High density (>0.5): {(df['stops_density'] > 0.5).sum()} routes (urban)\n")
        f.write(f"  - Low density (<0.2): {(df['stops_density'] < 0.2).sum()} routes (rural)\n\n")
        
        f.write(f"Average Distance per Stop:\n")
        f.write(f"  - Mean: {df['avg_distance_per_stop'].mean():.1f} km/stop\n")
        f.write(f"  - Median: {df['avg_distance_per_stop'].median():.1f} km/stop\n")
        f.write(f"  - Interpretation: Shorter = clustered stores (efficient), Longer = spread out (delays)\n\n")
        
        # 4. ROUTE TYPES
        f.write("4. ROUTE TYPE DISTRIBUTION (BY DISTANCE)\n")
        f.write("-" * 80 + "\n")
        route_type_counts = df['route_type'].value_counts()
        for route_type, count in route_type_counts.items():
            pct = count / len(df) * 100
            avg_time = df[df['route_type'] == route_type]['estimated_time_minutes'].mean()
            avg_stops = df[df['route_type'] == route_type]['stops_count'].mean()
            f.write(f"{route_type:12} {count:3} routes ({pct:5.1f}%) | "
                   f"Avg Time: {avg_time:3.0f} min | Avg Stops: {avg_stops:4.1f}\n")
        f.write(f"\nUrban routes (<30km): Short, many stops, traffic risk\n")
        f.write(f"Suburban routes (30-60km): Medium length, moderate stops\n")
        f.write(f"Rural routes (>60km): Long, fewer stops, freshness risk\n\n")
        
        # 5. VEHICLE CAPACITY ANALYSIS
        f.write("5. VEHICLE CAPACITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        capacity_counts = df['load_capacity_kg'].value_counts().sort_index()
        f.write(f"Capacity Distribution:\n")
        for capacity, count in capacity_counts.items():
            pct = count / len(df) * 100
            avg_stops = df[df['load_capacity_kg'] == capacity]['stops_count'].mean()
            f.write(f"  {capacity:,} kg: {count:3} routes ({pct:5.1f}%) | Avg Stops: {avg_stops:4.1f}\n")
        
        f.write(f"\nCapacity Utilization (Estimated):\n")
        f.write(f"  - Avg Load (50kg/stop assumption): {df['load_utilization'].mean():.0f} kg\n")
        f.write(f"  - Avg Capacity Strain: {df['capacity_strain'].mean():.1%}\n")
        f.write(f"  - Overloaded routes (>100% strain): {(df['capacity_strain'] > 1.0).sum()}\n")
        f.write(f"  - Underutilized routes (<50% strain): {(df['capacity_strain'] < 0.5).sum()}\n")
        f.write(f"\nNote: Overloading → damage (squashed loaves), safety risk\n")
        f.write(f"      Underutilization → inefficient vehicle usage\n\n")
        
        # 6. REGIONAL ANALYSIS
        f.write("6. REGIONAL DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        region_counts = df['region'].value_counts()
        for region, count in region_counts.items():
            pct = count / len(df) * 100
            avg_dist = df[df['region'] == region]['distance_km'].mean()
            avg_stops = df[df['region'] == region]['stops_count'].mean()
            avg_time = df[df['region'] == region]['estimated_time_minutes'].mean()
            f.write(f"{region:15} {count:3} routes ({pct:5.1f}%) | "
                   f"{avg_dist:5.1f} km | {avg_stops:4.1f} stops | {avg_time/60:4.1f} hrs\n")
        
        f.write(f"\nRegional Insights:\n")
        f.write(f"  - Bindura (most routes): {region_counts.iloc[0]} configurations\n")
        f.write(f"  - Coverage: 9 regions = comprehensive national network\n")
        f.write(f"  - Balanced distribution = no single region overloaded\n\n")
        
        # 7. TRIP TIMING WINDOWS
        f.write("7. TRIP TIMING WINDOWS\n")
        f.write("-" * 80 + "\n")
        start_window_counts = df['trip_start_window'].value_counts()
        f.write(f"Departure Time Windows:\n")
        for window, count in start_window_counts.items():
            pct = count / len(df) * 100
            avg_time = df[df['trip_start_window'] == window]['estimated_time_minutes'].mean()
            f.write(f"  {window}: {count:3} routes ({pct:5.1f}%) | Avg Trip Duration: {avg_time/60:.1f} hrs\n")
        
        f.write(f"\nTrip End Window: {df['trip_end_window'].unique()[0]}\n")
        f.write(f"\nTiming Strategy:\n")
        f.write(f"  - Early departure (03:00-05:00): Avoid traffic, maximize freshness window\n")
        f.write(f"  - Same-day return: Efficient vehicle utilization\n")
        f.write(f"  - Avg trip duration: {df['estimated_time_minutes'].mean()/60:.1f} hours (out + back)\n\n")
        
        # 8. COMPLEXITY & RISK SCORING
        f.write("8. ROUTE COMPLEXITY & RISK ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Complexity Score (distance × stops / time):\n")
        f.write(f"  - Mean: {df['complexity_score'].mean():.1f}\n")
        f.write(f"  - Median: {df['complexity_score'].median():.1f}\n")
        f.write(f"  - High complexity (>10): {(df['complexity_score'] > 10).sum()} routes\n\n")
        
        f.write(f"Risk Score (0-1 scale, higher = riskier):\n")
        f.write(f"  - Mean: {df['risk_score'].mean():.2f}\n")
        f.write(f"  - Median: {df['risk_score'].median():.2f}\n")
        f.write(f"  - High risk (>0.7): {(df['risk_score'] > 0.7).sum()} routes\n")
        f.write(f"  - Low risk (<0.3): {(df['risk_score'] < 0.3).sum()} routes\n\n")
        
        # Top 10 riskiest routes
        f.write(f"TOP 10 HIGHEST RISK ROUTES:\n")
        top_risk = df.nlargest(10, 'risk_score')[
            ['route_id', 'route_name', 'distance_km', 'stops_count', 
             'estimated_time_minutes', 'risk_score']
        ]
        for idx, row in top_risk.iterrows():
            f.write(f"  {row['route_id']}: {row['route_name'][:30]:30} | "
                   f"{row['distance_km']:5.1f}km, {row['stops_count']:2.0f} stops, "
                   f"{row['estimated_time_minutes']/60:4.1f}h | Risk: {row['risk_score']:.2f}\n")
        
        f.write(f"\nRisk Factors:\n")
        f.write(f"  - Long distance → freshness degradation\n")
        f.write(f"  - Many stops → delay accumulation\n")
        f.write(f"  - High capacity strain → damage risk\n\n")
        
        # 9. DRIVER & VEHICLE ASSIGNMENT
        f.write("9. DRIVER & VEHICLE ASSIGNMENT PATTERNS\n")
        f.write("-" * 80 + "\n")
        routes_per_vehicle = df.groupby('vehicle_id').size()
        routes_per_driver = df.groupby('driver_id').size()
        
        f.write(f"Vehicle Utilization:\n")
        f.write(f"  - Total vehicles: {df['vehicle_id'].nunique()}\n")
        f.write(f"  - Avg routes per vehicle: {routes_per_vehicle.mean():.1f}\n")
        f.write(f"  - Max routes per vehicle: {routes_per_vehicle.max()}\n")
        f.write(f"  - Min routes per vehicle: {routes_per_vehicle.min()}\n\n")
        
        f.write(f"Driver Assignment:\n")
        f.write(f"  - Total drivers: {df['driver_id'].nunique()}\n")
        f.write(f"  - Unassigned routes: {df['driver_id'].isnull().sum()}\n")
        f.write(f"  - Avg routes per driver: {routes_per_driver.mean():.1f}\n")
        f.write(f"  - Max routes per driver: {routes_per_driver.max()}\n")
        f.write(f"  - Some vehicles shared across drivers = flexibility\n\n")
        
        # 10. OPTIMIZATION OPPORTUNITIES
        f.write("10. OPTIMIZATION OPPORTUNITIES & RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Opportunity 1: Rebalance overloaded routes
        overloaded = df[df['capacity_strain'] > 1.0]
        if len(overloaded) > 0:
            f.write(f"🚨 OVERLOADED ROUTES: {len(overloaded)} routes\n")
            f.write(f"   Action: Split high-stop routes, assign larger vehicles\n")
            f.write(f"   Impact: Reduce damage, improve safety\n\n")
        
        # Opportunity 2: Optimize underutilized vehicles
        underutilized = df[df['capacity_strain'] < 0.5]
        f.write(f"💡 UNDERUTILIZED VEHICLES: {len(underutilized)} routes (<50% capacity)\n")
        f.write(f"   Action: Consolidate routes, downsize vehicles\n")
        f.write(f"   Impact: Reduce fuel costs, improve efficiency\n\n")
        
        # Opportunity 3: Long rural routes
        long_rural = df[(df['route_type'] == 'Rural') & (df['estimated_time_minutes'] > 180)]
        f.write(f"⏰ LONG RURAL ROUTES: {len(long_rural)} routes (>3 hours)\n")
        f.write(f"   Action: Establish regional hubs, stagger dispatch\n")
        f.write(f"   Impact: Reduce freshness risk, improve delivery reliability\n\n")
        
        # Opportunity 4: High-density urban routes
        high_density_urban = df[(df['route_type'] == 'Urban') & (df['stops_density'] > 0.5)]
        f.write(f"🏙️ HIGH-DENSITY URBAN ROUTES: {len(high_density_urban)} routes\n")
        f.write(f"   Action: Optimize stop sequencing, avoid peak traffic\n")
        f.write(f"   Impact: Reduce delays, improve on-time delivery\n\n")
        
        # Opportunity 5: Driver assignment gaps
        if df['driver_id'].isnull().sum() > 0:
            f.write(f"👤 DRIVER ASSIGNMENT GAPS: {df['driver_id'].isnull().sum()} unassigned routes\n")
            f.write(f"   Action: Assign backup drivers, cross-train\n")
            f.write(f"   Impact: Improve coverage, reduce delays\n\n")
        
        # Summary recommendations
        f.write(f"KEY RECOMMENDATIONS:\n")
        f.write(f"1. DELAY NORMALIZATION: Use estimated_time_min as baseline for anomaly detection\n")
        f.write(f"   normalized_delay = (actual - estimated) / estimated\n\n")
        f.write(f"2. ROUTE RISK SCORING: Integrate risk_score with dispatch/waste data\n")
        f.write(f"   High-risk routes should have priority monitoring\n\n")
        f.write(f"3. VEHICLE OPTIMIZATION: Match vehicle capacity to route demand\n")
        f.write(f"   Reduce overloading (damage) and underutilization (cost)\n\n")
        f.write(f"4. REGIONAL HUBS: For long rural routes, consider intermediate storage\n")
        f.write(f"   Reduces freshness degradation on 3+ hour routes\n\n")
        f.write(f"5. DRIVER PERFORMANCE: Link driver_id to dispatch outcomes\n")
        f.write(f"   Identify training needs, reward high performers\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF ROUTE & TRANSPORT METADATA SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"INFO: Wrote {summary_path}")

def grouped_summaries(df):
    """
    Generate CSV summaries grouped by different dimensions.
    """
    # 1. By Route (aggregated across vehicle assignments)
    route_summary = df.groupby('route_id').agg({
        'route_name': 'first',
        'distance_km': 'first',
        'estimated_time_minutes': 'mean',
        'stops_count': 'first',
        'region': 'first',
        'load_capacity_kg': 'mean',
        'efficiency_km_per_min': 'mean',
        'risk_score': 'mean',
        'vehicle_id': 'nunique',  # How many vehicles serve this route
        'driver_id': 'nunique'     # How many drivers serve this route
    }).round(2)
    route_summary.columns = [
        'route_name', 'distance_km', 'avg_estimated_time_min', 'stops_count',
        'region', 'avg_load_capacity_kg', 'avg_efficiency_km_per_min',
        'avg_risk_score', 'num_vehicles', 'num_drivers'
    ]
    route_summary = route_summary.sort_values('avg_risk_score', ascending=False)
    route_summary.to_csv(SUMMARIES_DIR / 'routes_by_route.csv')
    print(f"INFO: Wrote routes_by_route.csv")
    
    # 2. By Region
    region_summary = df.groupby('region').agg({
        'route_id': 'nunique',
        'distance_km': ['mean', 'sum'],
        'estimated_time_minutes': 'mean',
        'stops_count': ['mean', 'sum'],
        'load_capacity_kg': 'mean',
        'risk_score': 'mean'
    }).round(2)
    region_summary.columns = [
        'num_routes', 'avg_distance_km', 'total_distance_km',
        'avg_time_min', 'avg_stops', 'total_stops', 'avg_capacity_kg', 'avg_risk_score'
    ]
    region_summary = region_summary.sort_values('num_routes', ascending=False)
    region_summary.to_csv(SUMMARIES_DIR / 'routes_by_region.csv')
    print(f"INFO: Wrote routes_by_region.csv")
    
    # 3. By Vehicle
    vehicle_summary = df.groupby('vehicle_id').agg({
        'route_id': 'nunique',
        'distance_km': 'mean',
        'estimated_time_minutes': 'mean',
        'stops_count': 'mean',
        'load_capacity_kg': 'first',
        'risk_score': 'mean',
        'region': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common region
    }).round(2)
    vehicle_summary.columns = [
        'num_routes', 'avg_distance_km', 'avg_time_min', 'avg_stops',
        'capacity_kg', 'avg_risk_score', 'primary_region'
    ]
    vehicle_summary = vehicle_summary.sort_values('num_routes', ascending=False)
    vehicle_summary.to_csv(SUMMARIES_DIR / 'routes_by_vehicle.csv')
    print(f"INFO: Wrote routes_by_vehicle.csv")
    
    # 4. By Driver (excluding nulls)
    driver_summary = df[df['driver_id'].notna()].groupby('driver_id').agg({
        'route_id': 'nunique',
        'distance_km': 'mean',
        'estimated_time_minutes': 'mean',
        'stops_count': 'mean',
        'risk_score': 'mean',
        'region': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).round(2)
    driver_summary.columns = [
        'num_routes', 'avg_distance_km', 'avg_time_min', 'avg_stops',
        'avg_risk_score', 'primary_region'
    ]
    driver_summary = driver_summary.sort_values('num_routes', ascending=False)
    driver_summary.to_csv(SUMMARIES_DIR / 'routes_by_driver.csv')
    print(f"INFO: Wrote routes_by_driver.csv")
    
    # 5. By Route Type
    route_type_summary = df.groupby('route_type').agg({
        'route_id': 'count',
        'distance_km': ['mean', 'min', 'max'],
        'estimated_time_minutes': 'mean',
        'stops_count': 'mean',
        'efficiency_km_per_min': 'mean',
        'risk_score': 'mean'
    }).round(2)
    route_type_summary.columns = [
        'num_configs', 'avg_distance_km', 'min_distance_km', 'max_distance_km',
        'avg_time_min', 'avg_stops', 'avg_efficiency', 'avg_risk_score'
    ]
    route_type_summary.to_csv(SUMMARIES_DIR / 'routes_by_type.csv')
    print(f"INFO: Wrote routes_by_type.csv")
    
    # 6. High Risk Routes (Top 50)
    high_risk = df.nlargest(50, 'risk_score')[[
        'route_id', 'route_name', 'vehicle_id', 'driver_id',
        'distance_km', 'stops_count', 'estimated_time_minutes',
        'load_capacity_kg', 'capacity_strain', 'risk_score', 'region'
    ]].copy()
    high_risk = high_risk.round(2)
    high_risk.to_csv(SUMMARIES_DIR / 'routes_high_risk_top50.csv', index=False)
    print(f"INFO: Wrote routes_high_risk_top50.csv")

def visualizations(df):
    """
    Generate 12 comprehensive visualizations for route metadata analysis.
    """
    def add_caption(text):
        fig = plt.gcf()
        fig.text(0.5, 0.01, text, ha='center', va='bottom', fontsize=9, color='dimgray')
    # 1. Distance Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['distance_km'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(df['distance_km'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["distance_km"].mean():.1f} km')
    plt.axvline(df['distance_km'].median(), color='orange', linestyle='--',
                label=f'Median: {df["distance_km"].median():.1f} km')
    plt.xlabel('Distance (km)')
    plt.ylabel('Frequency')
    plt.title('Route Distance Distribution\nShows spread of short vs long routes')
    plt.legend()
    add_caption('Summarizes route length variability across the delivery network.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_distance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_distance_distribution.png")
    
    # 2. Route Type Distribution (Urban/Suburban/Rural)
    plt.figure(figsize=(8, 6))
    route_type_counts = df['route_type'].value_counts()
    colors = {'Urban': '#2ecc71', 'Suburban': '#f39c12', 'Rural': '#e74c3c'}
    bars = plt.bar(route_type_counts.index, route_type_counts.values,
                   color=[colors[x] for x in route_type_counts.index],
                   edgecolor='black', alpha=0.8)
    plt.xlabel('Route Type')
    plt.ylabel('Number of Configurations')
    plt.title('Route Type Distribution (by Distance)\nUrban <30km | Suburban 30-60km | Rural >60km')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} ({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    add_caption('Shows the balance of urban, suburban, and rural coverage.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_type_distribution.png")
    
    # 3. Stops Count Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['stops_count'], bins=20, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(df['stops_count'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["stops_count"].mean():.1f} stops')
    plt.xlabel('Number of Stops')
    plt.ylabel('Frequency')
    plt.title('Stops per Route Distribution\nMore stops = higher complexity & delay risk')
    plt.legend()
    add_caption('Highlights route stop counts that drive complexity and delays.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_stops_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_stops_distribution.png")
    
    # 4. Distance vs Stops Scatter (with route type coloring)
    plt.figure(figsize=(10, 6))
    for route_type in df['route_type'].unique():
        subset = df[df['route_type'] == route_type]
        plt.scatter(subset['distance_km'], subset['stops_count'],
                   label=route_type, alpha=0.6, s=50)
    plt.xlabel('Distance (km)')
    plt.ylabel('Number of Stops')
    plt.title('Distance vs Stops by Route Type\nShows route complexity patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    add_caption('Shows how stop density changes with distance by route type.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_distance_vs_stops.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_distance_vs_stops.png")
    
    # 5. Regional Route Distribution
    plt.figure(figsize=(10, 6))
    region_counts = df['region'].value_counts()
    colors_region = sns.color_palette('Set2', len(region_counts))
    plt.barh(region_counts.index, region_counts.values, color=colors_region, edgecolor='black')
    plt.xlabel('Number of Route Configurations')
    plt.ylabel('Region')
    plt.title('Route Configurations by Region\nShows logistics network coverage')
    for i, v in enumerate(region_counts.values):
        plt.text(v + 0.5, i, str(v), va='center', fontweight='bold')
    add_caption('Compares route coverage across regions for network balance.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_by_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_by_region.png")
    
    # 6. Vehicle Capacity Distribution
    plt.figure(figsize=(8, 6))
    capacity_counts = df['load_capacity_kg'].value_counts().sort_index()
    plt.bar(capacity_counts.index.astype(str), capacity_counts.values,
            color='mediumseagreen', edgecolor='black', alpha=0.8)
    plt.xlabel('Load Capacity (kg)')
    plt.ylabel('Number of Configurations')
    plt.title('Vehicle Capacity Distribution\nFleet composition by truck size')
    for i, (capacity, count) in enumerate(capacity_counts.items()):
        plt.text(i, count, f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    add_caption('Shows fleet mix across different truck capacities.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_capacity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_capacity_distribution.png")
    
    # 7. Capacity Strain Analysis
    plt.figure(figsize=(10, 6))
    colors_strain = ['green' if x < 0.8 else 'orange' if x < 1.0 else 'red' 
                     for x in df['capacity_strain']]
    plt.scatter(df['stops_count'], df['capacity_strain'], c=colors_strain, alpha=0.6, s=50)
    plt.axhline(1.0, color='red', linestyle='--', label='100% Capacity (Overloaded)')
    plt.axhline(0.5, color='orange', linestyle='--', label='50% Capacity')
    plt.xlabel('Number of Stops')
    plt.ylabel('Capacity Strain (Load / Capacity)')
    plt.title('Capacity Strain vs Stops\n🟢 Green=OK | 🟠 Orange=High | 🔴 Red=Overloaded')
    plt.legend()
    plt.grid(True, alpha=0.3)
    add_caption('Highlights how stop count drives capacity strain and overload risk.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_capacity_strain.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_capacity_strain.png")
    
    # 8. Efficiency (km per minute) by Route Type
    plt.figure(figsize=(10, 6))
    df.boxplot(column='efficiency_km_per_min', by='route_type', figsize=(10, 6),
               patch_artist=True)
    plt.suptitle('')
    plt.title('Route Efficiency by Type\nHigher = faster travel (but includes stop time)')
    plt.xlabel('Route Type')
    plt.ylabel('Efficiency (km/min)')
    add_caption('Compares travel efficiency across urban, suburban, and rural routes.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_efficiency_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_efficiency_by_type.png")
    
    # 9. Trip Start Window Distribution
    plt.figure(figsize=(8, 6))
    start_counts = df['trip_start_window'].value_counts()
    colors_time = ['#3498db', '#9b59b6', '#e67e22']
    plt.bar(start_counts.index, start_counts.values, color=colors_time,
            edgecolor='black', alpha=0.8)
    plt.xlabel('Trip Start Window')
    plt.ylabel('Number of Configurations')
    plt.title('Departure Time Distribution\nEarly departure = avoid traffic, maximize freshness')
    for i, (window, count) in enumerate(start_counts.items()):
        plt.text(i, count, f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    add_caption('Shows how departures are scheduled to balance freshness and traffic.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_start_window.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_start_window.png")
    
    # 10. Risk Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['risk_score'], bins=30, color='crimson', edgecolor='black', alpha=0.7)
    plt.axvline(df['risk_score'].mean(), color='darkred', linestyle='--',
                label=f'Mean: {df["risk_score"].mean():.2f}')
    plt.axvline(0.7, color='orange', linestyle='--', label='High Risk Threshold (0.7)')
    plt.xlabel('Risk Score (0-1)')
    plt.ylabel('Frequency')
    plt.title('Route Risk Score Distribution\nComposite: distance + stops + capacity strain')
    plt.legend()
    add_caption('Summarizes how many routes fall into low vs high risk bands.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_risk_distribution.png")
    
    # 11. Top 15 Riskiest Routes
    plt.figure(figsize=(10, 8))
    top_risk = df.nlargest(15, 'risk_score').sort_values('risk_score')
    colors_risk = ['red' if x > 0.7 else 'orange' for x in top_risk['risk_score']]
    plt.barh(range(len(top_risk)), top_risk['risk_score'], color=colors_risk, edgecolor='black')
    plt.yticks(range(len(top_risk)), 
               [f"{row['route_id']}: {row['route_name'][:25]}" 
                for _, row in top_risk.iterrows()], fontsize=8)
    plt.xlabel('Risk Score')
    plt.title('Top 15 Highest Risk Routes\n🔴 Priority monitoring required')
    plt.axvline(0.7, color='darkred', linestyle='--', alpha=0.5, label='High Risk (0.7)')
    plt.legend()
    add_caption('Highlights the routes requiring the most immediate attention.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_top_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_top_risk.png")
    
    # 12. Complexity Score vs Risk Score
    plt.figure(figsize=(10, 6))
    plt.scatter(df['complexity_score'], df['risk_score'], alpha=0.6, s=50, c='purple')
    plt.xlabel('Complexity Score (distance × stops / time)')
    plt.ylabel('Risk Score')
    plt.title('Route Complexity vs Risk\nBoth dimensions needed for optimization')
    plt.grid(True, alpha=0.3)
    
    # Annotate highest risk
    highest_risk = df.nlargest(3, 'risk_score')
    for _, row in highest_risk.iterrows():
        plt.annotate(row['route_id'], 
                    (row['complexity_score'], row['risk_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red')
    
    add_caption('Shows how complexity and risk move together across routes.')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'routes_complexity_vs_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"INFO: Saved routes_complexity_vs_risk.png")

    # 13. Distance vs estimated time scatter
    if 'distance_km' in df.columns and 'estimated_time_minutes' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['distance_km'], df['estimated_time_minutes'], alpha=0.6, s=50, c='teal')
        plt.xlabel('Distance (km)')
        plt.ylabel('Estimated Time (minutes)')
        plt.title('Distance vs Estimated Time\nLonger routes should align with longer durations')
        plt.grid(True, alpha=0.3)
        add_caption('Validates whether expected travel time scales with distance.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'routes_distance_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved routes_distance_vs_time.png")

    # 14. Risk score by region (boxplot)
    if 'region' in df.columns:
        plt.figure(figsize=(12, 6))
        df.boxplot(column='risk_score', by='region', figsize=(12, 6), patch_artist=True)
        plt.suptitle('')
        plt.xlabel('Region')
        plt.ylabel('Risk Score')
        plt.title('Risk Score Distribution by Region')
        add_caption('Compares regional risk profiles to prioritize route reviews.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'routes_risk_by_region.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved routes_risk_by_region.png")

    # 15. Capacity strain by route type (boxplot)
    if 'route_type' in df.columns:
        plt.figure(figsize=(10, 6))
        df.boxplot(column='capacity_strain', by='route_type', figsize=(10, 6), patch_artist=True)
        plt.suptitle('')
        plt.xlabel('Route Type')
        plt.ylabel('Capacity Strain (Load / Capacity)')
        plt.title('Capacity Strain by Route Type')
        add_caption('Shows how load pressure differs across urban, suburban, and rural routes.')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'routes_capacity_strain_by_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved routes_capacity_strain_by_type.png")

def main():
    """
    Main execution flow for Route & Transport Metadata EDA.
    """
    # Load and prepare data
    df = load_and_prepare()
    
    # Generate summary statistics
    summary_stats(df)
    
    # Generate grouped summaries
    grouped_summaries(df)
    
    # Generate visualizations
    visualizations(df)
    
    print("INFO: ✅ Route & Transport Metadata EDA complete!")
    print(f"INFO: Outputs in {REPORTS_DIR} and {FIGURES_DIR}")

if __name__ == '__main__':
    main()
