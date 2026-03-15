"""
Master Anomalies Aggregator

Aggregates and standardizes anomalies from all EDA outputs into a single table for modeling and reporting.
"""
import pandas as pd
from pathlib import Path

# Paths to anomaly sources
REPORTS_DIR = Path(__file__).parents[2] / "reports"
SUMMARIES_DIR = REPORTS_DIR / "summaries"
MODELS_DIR = REPORTS_DIR / "models"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

# Output path
MASTER_OUT = MODELS_DIR / "master_anomalies.csv"

# Load anomaly sources
def load_dispatch_anomalies():
    p = FEATURES_DIR / "dispatch_anomalies.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["anomaly_type"] = "dispatch_delay"
        df["entity_id"] = df["route_id"].fillna("") + ":" + df["plant_id"].fillna("")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["severity"] = df["route_id_zscore"].abs().fillna(0)
        df["description"] = df.apply(lambda r: f"Delay {r['dispatch_delay_minutes']} min (z={r['route_id_zscore']:.2f})", axis=1)
        return df[["timestamp","anomaly_type","entity_id","severity","description"]]
    return pd.DataFrame()

def load_inventory_anomalies():
    p = SUMMARIES_DIR / "inventory_anomalies_top50.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["anomaly_type"] = "inventory_negative_balance"
        df["entity_id"] = df["Location_Type"].astype(str) + ":" + df["SKU"].astype(str)
        df["timestamp"] = pd.NaT
        df["severity"] = df["Min_Balance"].abs()
        df["description"] = df.apply(lambda r: f"Negative balance {r['Min_Balance']} (count={r['Negative_Balance_Count']})", axis=1)
        return df[["timestamp","anomaly_type","entity_id","severity","description"]]
    return pd.DataFrame()

def load_flagged_dispatch_anomalies():
    p = MODELS_DIR / "flagged_anomalies.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["anomaly_type"] = "dispatch_uniqueness"
        df["entity_id"] = df["dispatch_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.NaT
        df["severity"] = 1
        df["description"] = "Duplicate/uniqueness anomaly"
        # remove exact duplicate rows coming from the flagged file itself
        df = df.drop_duplicates(subset=["timestamp","anomaly_type","entity_id","severity","description"]) 

        # Filter flagged uniqueness anomalies to only those that are currently duplicated
        # in the processed dispatch dataset (avoids re-emitting historical issues).
        proc_p = Path(__file__).parents[2] / "data" / "processed" / "dispatch_dataset.parquet"
        if proc_p.exists():
            try:
                proc = pd.read_parquet(proc_p)
                if "dispatch_id" in proc.columns:
                    dup_ids = set(proc.loc[proc["dispatch_id"].duplicated(keep=False), "dispatch_id"].unique())
                    if len(dup_ids) > 0:
                        df = df[df["entity_id"].isin(dup_ids)]
                    else:
                        # no duplicates in processed => no uniqueness anomalies
                        df = df.iloc[0:0]
            except Exception:
                # if processed read fails, keep flagged anomalies as-is
                pass

        return df[["timestamp","anomaly_type","entity_id","severity","description"]]
    return pd.DataFrame()

def main():
    dfs = [
        load_dispatch_anomalies(),
        load_inventory_anomalies(),
        load_flagged_dispatch_anomalies(),
    ]
    master = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    master = master.sort_values("timestamp", na_position="last")
    # drop exact duplicate rows that may have arisen from multiple sources
    master = master.drop_duplicates()
    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_OUT, index=False)
    print(f"Wrote master anomalies: {MASTER_OUT} ({len(master)})")

if __name__ == "__main__":
    main()
