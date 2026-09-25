import pandas as pd
from pathlib import Path

BASE_DIR = Path(".")

files = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]

for filename in files:

    matches = list(BASE_DIR.rglob(filename))

    if not matches:
        print(f"\nFILE NOT FOUND: {filename}")
        continue

    path = matches[0]

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print(f"PATH: {path}")
    print("=" * 80)

    # Read only the timestamp column
    df = pd.read_csv(
        path,
        usecols=["Timestamp"],
        dtype={"Timestamp": "string"}
    )

    raw = df["Timestamp"].str.strip()

    parsed = pd.to_datetime(
        raw,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    invalid = parsed.isna()

    print(f"Total rows:                 {len(df):,}")
    print(f"Invalid timestamps:         {invalid.sum():,}")

    if invalid.any():
        print("\nInvalid timestamp examples:")
        print(raw[invalid].head(20).to_string(index=False))

    print(f"\nEarliest valid timestamp:   {parsed.min()}")
    print(f"Latest valid timestamp:     {parsed.max()}")

    print("\nDate distribution:")
    print(parsed.dt.date.value_counts().sort_index().to_string())

    print("\nYear distribution:")
    print(parsed.dt.year.value_counts().sort_index().to_string())

    # Check chronological order after correct parsing
    out_of_order = (parsed.diff() < pd.Timedelta(0)).sum()

    print(f"\nOut-of-order rows:          {out_of_order:,}")
    print(
        "Timestamp order:            "
        + ("SORTED" if out_of_order == 0 else "NOT SORTED")
    )

    # Check duplicate timestamps
    duplicates = parsed.duplicated().sum()
    print(f"Duplicate timestamps:       {duplicates:,}")