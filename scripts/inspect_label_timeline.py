import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/cic_ids2018_clean")

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]


BENIGN = "Benign"


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

    df = df.sort_values(
        "Timestamp",
        kind="mergesort"
    ).reset_index(drop=True)

    # ================================================================
    # 1. LABEL TIME RANGES
    # ================================================================

    print("\nLABEL TIME RANGES")
    print("-" * 100)

    summary = (
        df.groupby("Label")["Timestamp"]
        .agg(["min", "max", "count"])
        .sort_values("min")
    )

    for label, row in summary.iterrows():
        print(
            f"{label:<35} "
            f"{row['count']:>10,.0f} rows   "
            f"{row['min']}  ->  {row['max']}"
        )

    # ================================================================
    # 2. GROUP ALL FLOWS BY EXACT TIMESTAMP
    # ================================================================

    timestamp_groups = (
        df.groupby("Timestamp")["Label"]
        .agg(list)
        .reset_index()
    )

    timestamp_groups["FlowCount"] = (
        timestamp_groups["Label"].apply(len)
    )

    # ================================================================
    # 3. DETERMINE A TIME-LEVEL STATE
    #
    # BENIGN
    # ONE ATTACK LABEL
    # MULTI_ATTACK
    # ================================================================

    def get_state(labels):

        attack_labels = sorted(
            set(label for label in labels if label != BENIGN)
        )

        if len(attack_labels) == 0:
            return BENIGN

        if len(attack_labels) == 1:
            return attack_labels[0]

        return "MULTI_ATTACK"

    timestamp_groups["State"] = (
        timestamp_groups["Label"].apply(get_state)
    )

    # ================================================================
    # 4. TIMESTAMP STATISTICS
    # ================================================================

    total_timestamps = len(timestamp_groups)

    benign_timestamps = (
        timestamp_groups["State"] == BENIGN
    ).sum()

    attack_timestamps = (
        timestamp_groups["State"] != BENIGN
    ).sum()

    mixed_attack_timestamps = (
        timestamp_groups["State"] == "MULTI_ATTACK"
    ).sum()

    print("\nTIMESTAMP-LEVEL SUMMARY")
    print("-" * 100)

    print(f"Unique timestamps:                    {total_timestamps:,}")
    print(f"Benign-only timestamps:               {benign_timestamps:,}")
    print(f"Attack-present timestamps:            {attack_timestamps:,}")
    print(f"Multi-attack timestamps:              {mixed_attack_timestamps:,}")

    # ================================================================
    # 5. TIME-LEVEL STATE TRANSITIONS
    # ================================================================

    timestamp_groups["PreviousState"] = (
        timestamp_groups["State"].shift(1)
    )

    transitions = timestamp_groups.iloc[1:].copy()

    transition_counts = (
        transitions
        .groupby(["PreviousState", "State"])
        .size()
        .sort_values(ascending=False)
    )

    print("\nTIME-LEVEL STATE TRANSITIONS")
    print("-" * 100)

    for (previous_state, current_state), count in transition_counts.head(30).items():

        print(
            f"{previous_state:<32} -> "
            f"{current_state:<32} "
            f"{count:>8,}"
        )

    # ================================================================
    # 6. CONTIGUOUS TIME-LEVEL STATES
    # ================================================================

    run_id = (
        timestamp_groups["State"]
        != timestamp_groups["State"].shift()
    ).cumsum().rename("run_id")

    runs = (
        timestamp_groups
        .groupby(run_id)
        .agg(
            State=("State", "first"),
            Start=("Timestamp", "min"),
            End=("Timestamp", "max"),
            Seconds=("Timestamp", "size"),
            Flows=("FlowCount", "sum"),
        )
        .reset_index(drop=True)
    )

    print("\nTIME-LEVEL RUN SUMMARY")
    print("-" * 100)

    run_summary = (
        runs.groupby("State")["Seconds"]
        .agg(
            Runs="count",
            Min="min",
            Median="median",
            Mean="mean",
            Max="max",
        )
        .sort_values("Runs", ascending=False)
    )

    print(run_summary.to_string())

    # ================================================================
    # 7. FIRST 30 REAL TIME-LEVEL TRANSITIONS
    # ================================================================

    print("\nFIRST 30 TIME-LEVEL TRANSITIONS")
    print("-" * 100)

    changed = (
        timestamp_groups["State"]
        != timestamp_groups["State"].shift()
    )

    changed_rows = timestamp_groups.loc[
        changed,
        ["Timestamp", "State"]
    ].copy()

    changed_rows["Previous"] = (
        timestamp_groups["State"]
        .shift(1)
        .loc[changed_rows.index]
        .values
    )

    changed_rows = changed_rows.iloc[1:]

    for _, row in changed_rows.head(30).iterrows():

        print(
            f"{row['Timestamp']} | "
            f"{row['Previous']} -> {row['State']}"
        )

    # ================================================================
    # 8. ATTACK LABEL OVERLAP
    # ================================================================

    print("\nATTACK LABEL OVERLAP")
    print("-" * 100)

    overlap_counter = {}

    for labels in timestamp_groups["Label"]:

        attack_labels = sorted(
            set(label for label in labels if label != BENIGN)
        )

        if len(attack_labels) >= 2:

            key = tuple(attack_labels)

            overlap_counter[key] = (
                overlap_counter.get(key, 0) + 1
            )

    if overlap_counter:

        for labels, count in sorted(
            overlap_counter.items(),
            key=lambda x: -x[1]
        ):
            print(
                f"{' + '.join(labels):<60} "
                f"{count:>8,} timestamps"
            )

    else:
        print("No timestamps contained multiple attack labels.")