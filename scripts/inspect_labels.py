import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/cic_ids2018_clean")

files = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]

total_counts = {}

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    counts = {}

    for chunk in pd.read_csv(
        path,
        usecols=["Label"],
        dtype={"Label": "string"},
        chunksize=100_000
    ):
        labels = chunk["Label"].str.strip()

        for label, count in labels.value_counts(dropna=False).items():
            label = "<MISSING>" if pd.isna(label) else label
            counts[label] = counts.get(label, 0) + int(count)
            total_counts[label] = total_counts.get(label, 0) + int(count)

    total = sum(counts.values())

    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        print(f"{label:<40} {count:>12,}  ({percentage:>7.3f}%)")

    print(f"\nTotal rows: {total:,}")


print("\n" + "=" * 80)
print("COMBINED LABEL DISTRIBUTION")
print("=" * 80)

grand_total = sum(total_counts.values())

for label, count in sorted(total_counts.items(), key=lambda x: -x[1]):
    percentage = count / grand_total * 100
    print(f"{label:<40} {count:>12,}  ({percentage:>7.3f}%)")

print(f"\nGrand total: {grand_total:,}")
print("=" * 80)