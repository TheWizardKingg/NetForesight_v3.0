import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/cic_ids2018_clean")

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]

THRESHOLDS = [1, 5, 10, 30, 60, 120, 300, 600]


for filename in FILES:

    path = DATA_DIR / filename

    print("\n" + "=" * 100)
    print(filename)
    print("=" * 100)

    df = pd.read_csv(
        path,
        usecols=["Timestamp", "Label"],
        parse_dates=["Timestamp"],
    )

    df["Label"] = df["Label"].astype(str).str.strip()

    # Only attack observations
    df = df[df["Label"] != "Benign"]

    # One timestamp = one attack observation
    attack_times = (
        df.groupby("Timestamp")["Label"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    attack_times["Gap"] = (
        attack_times["Timestamp"]
        .diff()
        .dt.total_seconds()
    )

    print(f"\nUnique attack timestamps: {len(attack_times):,}")

    print("\nEPISODE COUNTS BY GAP THRESHOLD")
    print("-" * 100)

    print(
        f"{'Threshold':>12} "
        f"{'Episodes':>12}"
    )

    for threshold in THRESHOLDS:

        # First attack timestamp starts an episode.
        # A new episode begins whenever the gap exceeds threshold.
        starts = (
            attack_times["Gap"].isna()
            | (attack_times["Gap"] > threshold)
        )

        episodes = int(starts.sum())

        print(
            f"{threshold:>9} sec "
            f"{episodes:>12,}"
        )

    # ------------------------------------------------------------
    # Per-label counts at 60 seconds
    # ------------------------------------------------------------

    threshold = 60

    attack_times["NewEpisode"] = (
        attack_times["Gap"].isna()
        | (attack_times["Gap"] > threshold)
    )

    attack_times["EpisodeID"] = (
        attack_times["NewEpisode"].cumsum()
    )

    episode_labels = (
        attack_times
        .groupby("EpisodeID")["Label"]
        .first()
    )

    print("\n60-SECOND EPISODES BY LABEL")
    print("-" * 100)

    print(
        episode_labels
        .value_counts()
        .to_string()
    )