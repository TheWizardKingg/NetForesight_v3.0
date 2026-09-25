import pandas as pd
from pathlib import Path


BASE_DIR = Path(".")

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]

EXPECTED_DATES = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": pd.Timestamp("2018-02-14"),
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv": pd.Timestamp("2018-02-15"),
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": pd.Timestamp("2018-02-16"),
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv": pd.Timestamp("2018-02-21"),
}


for filename in FILES:

    matches = list(BASE_DIR.rglob(filename))

    if not matches:
        print(f"\nFILE NOT FOUND: {filename}")
        continue

    path = matches[0]
    expected_date = EXPECTED_DATES[filename]

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    df = pd.read_csv(
        path,
        usecols=["Timestamp"],
        dtype={"Timestamp": "string"},
    )

    raw = df["Timestamp"].fillna("").str.strip()

    parsed = pd.to_datetime(
        raw,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    bad = (
        parsed.isna()
        | (parsed.dt.normalize() != expected_date)
    )

    bad_indices = df.index[bad]

    if len(bad_indices) == 0:
        print("No anomalous timestamps found.")
        continue

    print(f"Anomalous rows: {len(bad_indices)}\n")

    for idx in bad_indices:
        print(f"Row index: {idx}")
        print(f"Raw timestamp: {repr(raw.iloc[idx])}")
        print(f"Parsed value: {parsed.iloc[idx]}")
        print("-" * 50)