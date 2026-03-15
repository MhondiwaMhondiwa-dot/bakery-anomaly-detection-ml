"""
Review anomaly type distribution in master_anomalies.csv
"""
import pandas as pd
from pathlib import Path

MASTER_PATH = Path(__file__).parents[2] / "reports" / "models" / "master_anomalies.csv"

def main():
    df = pd.read_csv(MASTER_PATH)
    print("Anomaly type counts:")
    print(df['anomaly_type'].value_counts())
    print("\nSample records for each type:")
    for t in df['anomaly_type'].unique():
        print(f"\nType: {t}")
        print(df[df['anomaly_type'] == t].head(3))

if __name__ == "__main__":
    main()
