from pathlib import Path

# Authoritative constants per DATASET_GUIDE_FOR_COPILOT.md

SHEPPERTON_PLANT_ID = "Shepperton_Plant"

ALLOWED_SKUS = [
    "Soft white",
    "High Energy Brown",
    "Whole grain loaf",
    "Low GI Seed loaf",
]

# Standardized column-safe SKU names (after cleaning)
STD_SKU_COLS = [
    "soft_white",
    "high_energy_brown",
    "whole_grain_loaf",
    "low_gi_seed_loaf",
]

ALLOWED_DEPOTS = [
    "Harare",
    "Hatcliff",
    "Chinhoyi",
    "Chegutu",
    "Bindura",
    "Murehwa",
    "Mutare",
]

# Critical fields expected across datasets (used for null checks)
CRITICAL_FIELDS = {
    "production_dataset": ["batch_id", "timestamp", "plant_id", "sku", "quantity_produced"],
    "quality_control_dataset": ["qc_id", "timestamp", "batch_id", "sku", "parameter", "value", "pass_fail"],
    "dispatch_dataset": ["dispatch_id", "timestamp", "plant_id", "depot_id", "vehicle_id", "total_quantity"],
    "route_transport_multivehicle": ["route_id", "route_name", "vehicle_id", "driver_id", "estimated_time_min", "distance_km", "depot_id", "load_capacity"],
    "sales_dataset": ["sale_id", "date", "plant_id", "depot_id", "sku", "quantity_sold"],
    "sales_pos_dataset": ["sale_id", "timestamp", "retailer_id", "sku", "quantity_sold"],
    "returns_dataset": ["return_id", "timestamp", "route_id", "retailer_id", "sku", "qty_returned"],
    "waste_dataset": ["waste_id", "timestamp", "plant_id", "sku", "quantity_wasted"],
    "inventory_stock_movements": ["record_id", "timestamp", "plant_id", "sku", "qty_in", "qty_out", "balance"],
    "holiday_production_sales": ["record_id", "holiday_name", "depot_id", "sku", "qty_produced", "qty_sold"],
}

# Paths
DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
SUMMARIES_DIR = REPORTS_DIR / "summaries"
FIGURES_DIR = REPORTS_DIR / "figures"
