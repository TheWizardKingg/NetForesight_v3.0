import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/cic_ids2018")
CHUNK_SIZE = 100_000

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("FULL DATASET LABEL INSPECTION")
print("=" * 70)

for file in csv_files:
    print(f"\n{'=' * 70}")
    print(f"FILE: {file.name}")
    print(f"{'=' * 70}")

    label_counts = {}
    total_rows = 0

    for chunk in pd.read_csv(
        file,
        usecols=["Label"],
        chunksize=CHUNK_SIZE
    ):
        total_rows += len(chunk)

        counts = chunk["Label"].value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

    print(f"\nTotal rows: {total_rows:,}")

    print("\nLabel distribution:")

    sorted_counts = sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for label, count in sorted_counts:
        percentage = count / total_rows * 100

        print(
            f"{label:<35} "
            f"{count:>12,} "
            f"({percentage:>6.2f}%)"
        )

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)