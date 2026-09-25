import pandas as pd
from pathlib import Path


BASE_DIR = Path(".")
INPUT_DIR = BASE_DIR / "data" / "cic_ids2018"
OUTPUT_DIR = BASE_DIR / "data" / "cic_ids2018_clean"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-14",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-15",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-16",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-21",
}


for filename, expected_date in FILES.items():

    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    print("\n" + "=" * 80)
    print(f"PROCESSING: {filename}")
    print("=" * 80)

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        continue

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    print("Loading CSV...")

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    original_rows = len(df)

    print(f"Original rows: {original_rows:,}")

    # ------------------------------------------------------------------
    # Remove embedded/repeated header rows
    # ------------------------------------------------------------------

    header_mask = (
        df["Timestamp"].astype(str).str.strip().eq("Timestamp")
        | df["Dst Port"].astype(str).str.strip().eq("Dst Port")
    )

    header_rows = header_mask.sum()

    if header_rows:
        print(f"Embedded header rows removed: {header_rows:,}")

        df = df.loc[~header_mask].copy()

    # ------------------------------------------------------------------
    # Parse timestamp explicitly
    # ------------------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    invalid_timestamp_mask = df["Timestamp"].isna()

    invalid_timestamp_rows = invalid_timestamp_mask.sum()

    if invalid_timestamp_rows:
        print(f"Invalid timestamps removed: {invalid_timestamp_rows:,}")

        df = df.loc[~invalid_timestamp_mask].copy()

    # ------------------------------------------------------------------
    # Remove timestamps belonging to the wrong date
    # ------------------------------------------------------------------

    expected_date = pd.Timestamp(expected_date)

    wrong_date_mask = (
        df["Timestamp"].dt.normalize() != expected_date
    )

    wrong_date_rows = wrong_date_mask.sum()

    if wrong_date_rows:
        print(f"Wrong-date rows removed: {wrong_date_rows:,}")

        df = df.loc[~wrong_date_mask].copy()

    # ------------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------------

    print("Sorting chronologically...")

    df = df.sort_values(
        by="Timestamp",
        kind="mergesort"
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    out_of_order = (
        df["Timestamp"].diff() < pd.Timedelta(0)
    ).sum()

    duplicate_timestamps = df["Timestamp"].duplicated().sum()

    print("\nFinal validation:")
    print(f"Rows remaining:              {len(df):,}")
    print(f"Earliest timestamp:           {df['Timestamp'].min()}")
    print(f"Latest timestamp:             {df['Timestamp'].max()}")
    print(f"Out-of-order rows:            {out_of_order:,}")
    print(f"Duplicate timestamps:         {duplicate_timestamps:,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    print(f"\nSaving cleaned file:")
    print(output_path)

    df.to_csv(
        output_path,
        index=False
    )

    print("DONE.")

print("\n" + "=" * 80)
print("TEMPORAL CLEANING COMPLETE")
print("=" * 80)