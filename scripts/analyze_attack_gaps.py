import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/cic_ids2018_clean")

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]


for filename in FILES:

    path = DATA_DIR / filename

    print("\n" + "=" * 100)
    print(f"FILE: {filename}")
    print("=" * 100)

    df = pd.read_csv(
        path,
        usecols=["Timestamp", "Label"],
        parse_dates=["Timestamp"],
    )

    df["Label"] = df["Label"].astype(str).str.strip()

    df = df[df["Label"] != "Benign"]

    # Collapse all flows occurring at the same timestamp
    attack_times = (
        df.groupby("Timestamp")["Label"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
    )

    attack_times["GapSeconds"] = (
        attack_times["Timestamp"]
        .diff()
        .dt.total_seconds()
    )

    attack_times = attack_times.dropna()

    print(f"Unique attack timestamps: {len(attack_times):,}")

    if len(attack_times) == 0:
        continue

    print("\nATTACK TIMESTAMP GAP DISTRIBUTION")
    print("-" * 100)

    print(
        attack_times["GapSeconds"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print("\nLargest gaps:")
    print(
        attack_times[
            ["Timestamp", "GapSeconds", "Label"]
        ]
        .sort_values("GapSeconds", ascending=False)
        .head(30)
        .to_string(index=False)
    )