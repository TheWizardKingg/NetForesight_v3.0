import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/cic_ids2018_clean")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]

BUCKET_SECONDS = 10
HISTORY_BUCKETS = 6       # 60 seconds
FUTURE_BUCKETS = 3        # 30 seconds


USECOLS = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "ACK Flag Cnt",
    "FIN Flag Cnt",
    "PSH Flag Cnt",
    "Label",
]


# ============================================================
# BUILD 10-SECOND WINDOWS
# ============================================================

def build_time_windows(path: Path):

    print(f"\nLoading: {path.name}")

    df = pd.read_csv(
        path,
        usecols=USECOLS,
        low_memory=False,
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    df = df.dropna(subset=["Timestamp"])

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    numeric_cols = [
        c for c in USECOLS
        if c not in ["Timestamp", "Label"]
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # Derived protocol features
    # --------------------------------------------------------

    df["TCPFlows"] = (df["Protocol"] == 6).astype(np.int32)
    df["UDPFlows"] = (df["Protocol"] == 17).astype(np.int32)

    # --------------------------------------------------------
    # 10-second bucket
    # --------------------------------------------------------

    df["Bucket"] = df["Timestamp"].dt.floor(
        f"{BUCKET_SECONDS}s"
    )

    # --------------------------------------------------------
    # Numeric aggregation
    # --------------------------------------------------------

    grouped = df.groupby("Bucket")

    windows = grouped.agg(
        FlowCount=("Label", "size"),

        FlowDurationMean=("Flow Duration", "mean"),

        FwdPktsSum=("Tot Fwd Pkts", "sum"),
        BwdPktsSum=("Tot Bwd Pkts", "sum"),

        FwdBytesSum=("TotLen Fwd Pkts", "sum"),
        BwdBytesSum=("TotLen Bwd Pkts", "sum"),

        PktLenMean=("Pkt Len Mean", "mean"),

        FlowBytesRateMean=("Flow Byts/s", "mean"),
        FlowPktsRateMean=("Flow Pkts/s", "mean"),

        SYNCount=("SYN Flag Cnt", "sum"),
        RSTCount=("RST Flag Cnt", "sum"),
        ACKCount=("ACK Flag Cnt", "sum"),
        FINCount=("FIN Flag Cnt", "sum"),
        PSHCount=("PSH Flag Cnt", "sum"),

        TCPFlows=("TCPFlows", "sum"),
        UDPFlows=("UDPFlows", "sum"),

        UniqueDstPorts=("Dst Port", "nunique"),
        UniqueProtocols=("Protocol", "nunique"),
    )

    # --------------------------------------------------------
    # Determine attack label for each bucket
    # --------------------------------------------------------

    attacks = df[df["Label"] != "Benign"]

    if len(attacks) > 0:

        attack_counts = (
            attacks
            .groupby(["Bucket", "Label"])
            .size()
            .reset_index(name="Count")
        )

        dominant_attack = (
            attack_counts
            .sort_values(
                ["Bucket", "Count"],
                ascending=[True, False],
            )
            .drop_duplicates("Bucket")
            .set_index("Bucket")["Label"]
        )

        windows["AttackLabel"] = (
            dominant_attack
            .reindex(windows.index)
            .fillna("Benign")
        )

    else:
        windows["AttackLabel"] = "Benign"

    windows = windows.sort_index()

    print(
        f"Created {len(windows):,} "
        f"{BUCKET_SECONDS}-second windows"
    )

    return windows


# ============================================================
# CONVERT WINDOWS INTO:
#
# past 60 sec -> next 30 sec label
# ============================================================

FEATURE_COLUMNS = [
    "FlowCount",
    "FlowDurationMean",
    "FwdPktsSum",
    "BwdPktsSum",
    "FwdBytesSum",
    "BwdBytesSum",
    "PktLenMean",
    "FlowBytesRateMean",
    "FlowPktsRateMean",
    "SYNCount",
    "RSTCount",
    "ACKCount",
    "FINCount",
    "PSHCount",
    "TCPFlows",
    "UDPFlows",
    "UniqueDstPorts",
    "UniqueProtocols",
]


def build_examples(windows):

    values = (
        windows[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    labels = windows["AttackLabel"].to_numpy()

    timestamps = windows.index.to_numpy()

    X = []
    y = []
    times = []

    max_i = len(windows) - FUTURE_BUCKETS

    for i in range(HISTORY_BUCKETS, max_i):

        history = values[
            i - HISTORY_BUCKETS:i
        ]

        future_labels = labels[
            i:i + FUTURE_BUCKETS
        ]

        # ----------------------------------------------------
        # "NEXT ACTIVITY" = first attack appearing
        # in the future window.
        # ----------------------------------------------------

        target = "Benign"

        for label in future_labels:

            if label != "Benign":
                target = label
                break

        X.append(
            history.reshape(-1)
        )

        y.append(target)

        times.append(timestamps[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        np.asarray(times),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    all_X = []
    all_y = []
    all_times = []

    for filename in FILES:

        path = DATA_DIR / filename

        if not path.exists():
            print(f"WARNING: Missing {path}")
            continue

        windows = build_time_windows(path)

        X, y, times = build_examples(windows)

        all_X.append(X)
        all_y.append(y)
        all_times.append(times)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    times = np.concatenate(all_times)

    print("\n" + "=" * 80)
    print("DEMO DATASET")
    print("=" * 80)

    print(f"Samples:       {len(X):,}")
    print(f"Features:      {X.shape[1]:,}")

    label_counts = pd.Series(y).value_counts()

    print("\nTarget distribution:")
    print(label_counts.to_string())

    # ========================================================
    # ENCODE LABELS
    # ========================================================

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    print("\nClasses:")
    for idx, label in enumerate(encoder.classes_):
        print(f"{idx}: {label}")

    # ========================================================
    # TRAIN / TEST
    #
    # This is a DEMO split, not final research evaluation.
    # ========================================================

    X_train, X_test, y_train, y_test, t_train, t_test = (
        train_test_split(
            X,
            y_encoded,
            times,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded,
        )
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples: ", len(X_test))

    # ========================================================
    # MODEL
    # ========================================================

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 80)
    print("DEMO MODEL RESULTS")
    print("=" * 80)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = ARTIFACT_DIR / "demo_model.joblib"
    encoder_path = ARTIFACT_DIR / "demo_label_encoder.joblib"

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        encoder,
        encoder_path,
    )

    metadata = {
        "bucket_seconds": BUCKET_SECONDS,
        "history_buckets": HISTORY_BUCKETS,
        "future_buckets": FUTURE_BUCKETS,
        "feature_columns": FEATURE_COLUMNS,
        "num_input_features": int(X.shape[1]),
        "classes": encoder.classes_.tolist(),
    }

    metadata_path = ARTIFACT_DIR / "demo_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # SAVE A SMALL TEST SET FOR REPLAY
    # ========================================================

    np.savez_compressed(
        ARTIFACT_DIR / "demo_test_data.npz",
        X=X_test,
        y=y_test,
        timestamps=t_test,
    )

    print("\nSaved artifacts:")

    print(model_path)
    print(encoder_path)
    print(metadata_path)
    print(
        ARTIFACT_DIR / "demo_test_data.npz"
    )


if __name__ == "__main__":
    main()