import os
import random
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"

OUTPUT_SEQUENCE_PATH = "data/unsw_sequences.pt"

ARTIFACT_DIR = "backend"

LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.pkl")
FEATURE_SCALER_PATH = os.path.join(ARTIFACT_DIR, "feature_scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(ARTIFACT_DIR, "feature_names.pkl")

SEQUENCE_LENGTH = 5

# Maximum ratio of Normal samples to attack samples.
# Example: 1.0 means approximately equal Normal/Attack.
NORMAL_TO_ATTACK_RATIO = 1.0

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# Utility
# ============================================================

def clean_column_names(df):
    """
    Removes accidental whitespace from column names.
    """
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_target_column(df):
    """
    Creates a clean attack category column.

    UNSW-NB15 normally contains:
        attack_cat

    Normal traffic can contain missing values, so missing
    attack categories are mapped to 'Normal'.
    """

    if "attack_cat" in df.columns:

        df["attack_cat"] = (
            df["attack_cat"]
            .fillna("Normal")
            .astype(str)
            .str.strip()
        )

        # Normalize naming inconsistencies.
        df["attack_cat"] = df["attack_cat"].replace({
            "Backdoors": "Backdoor",
            "backdoors": "Backdoor",
            "normal": "Normal",
            "Normal": "Normal",
        })

        return "attack_cat"

    if "label" in df.columns:
        print(
            "[!] WARNING: 'attack_cat' was not found. "
            "Falling back to binary 'label'."
        )

        df["target_category"] = np.where(
            df["label"].astype(int) == 0,
            "Normal",
            "Attack"
        )

        return "target_category"

    raise ValueError(
        "Dataset does not contain either 'attack_cat' or 'label'."
    )


# ============================================================
# Categorical Feature Encoding
# ============================================================

def encode_categorical_features(train_df, test_df):
    """
    Encodes categorical network features using mappings learned
    from the combined dataset.

    These mappings are stored so inference can later use the
    exact same feature representation.
    """

    categorical_columns = [
        "proto",
        "service",
        "state",
    ]

    encoders = {}

    for column in categorical_columns:

        if column not in train_df.columns:
            continue

        # Convert everything to strings first.
        train_values = train_df[column].fillna("UNKNOWN").astype(str)
        test_values = test_df[column].fillna("UNKNOWN").astype(str)

        # Build mapping using both datasets so that an unseen
        # test category does not crash preprocessing.
        combined_values = pd.concat(
            [train_values, test_values],
            ignore_index=True
        )

        unique_values = sorted(combined_values.unique())

        mapping = {
            value: index
            for index, value in enumerate(unique_values)
        }

        train_df[column] = train_values.map(mapping).astype(np.float32)
        test_df[column] = test_values.map(mapping).astype(np.float32)

        encoders[column] = mapping

        print(
            f"[+] Encoded categorical feature '{column}': "
            f"{len(mapping)} categories"
        )

    return train_df, test_df, encoders


# ============================================================
# Feature Preparation
# ============================================================

def prepare_features(train_df, test_df):
    """
    Creates the final numerical feature matrix.

    Important:
    The exact feature column order is saved and later used by
    the backend during inference.
    """

    ignored_columns = {
        "id",
        "label",
        "attack_cat",
        "target_category",
        "target",
    }

    feature_columns = [
        column
        for column in train_df.columns
        if column not in ignored_columns
    ]

    # Make sure test set contains exactly the same columns.
    missing_columns = [
        column
        for column in feature_columns
        if column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Test dataset is missing features: {missing_columns}"
        )

    train_features = train_df[feature_columns].copy()
    test_features = test_df[feature_columns].copy()

    # Convert everything to numeric.
    for column in feature_columns:

        train_features[column] = pd.to_numeric(
            train_features[column],
            errors="coerce"
        )

        test_features[column] = pd.to_numeric(
            test_features[column],
            errors="coerce"
        )

    # Replace invalid values.
    train_features = train_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    test_features = test_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values using TRAINING statistics only.
    train_medians = train_features.median(numeric_only=True)

    train_features = train_features.fillna(train_medians)
    test_features = test_features.fillna(train_medians)

    # Any column that still contains NaN gets zero.
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)

    print(
        f"[+] Final feature count: {len(feature_columns)}"
    )

    return (
        train_features,
        test_features,
        feature_columns,
    )


# ============================================================
# Sequence Construction
# ============================================================

def build_sequences(X, y, sequence_length):
    """
    Builds next-event prediction sequences.

    Input:
        [t0, t1, t2, t3, t4]

    Target:
        t5

    This means the model learns:

        previous 5 observations -> next observation's class
    """

    if len(X) <= sequence_length:
        raise ValueError(
            "Not enough samples to construct sequences."
        )

    sequences = []
    targets = []

    for i in range(len(X) - sequence_length):

        sequence = X[
            i:i + sequence_length
        ]

        target = y[
            i + sequence_length
        ]

        sequences.append(sequence)
        targets.append(target)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
    )


# ============================================================
# Class Balancing
# ============================================================

def balance_sequences(X, y, label_encoder):
    """
    Reduces the extreme Normal-class dominance.

    We do NOT randomly throw away attack samples.

    Instead:
        - Keep every attack sequence.
        - Keep a controlled number of Normal sequences.

    This prevents the model from simply learning:

        'Everything is Normal.'
    """

    normal_indices = np.where(
        y == label_encoder.transform(["Normal"])[0]
    )[0]

    attack_indices = np.where(
        y != label_encoder.transform(["Normal"])[0]
    )[0]

    print("\n[+] Before balancing:")
    print(f"    Normal sequences : {len(normal_indices):,}")
    print(f"    Attack sequences : {len(attack_indices):,}")

    if len(attack_indices) == 0:
        raise ValueError(
            "No attack sequences were found. "
            "Cannot build a useful attack forecasting dataset."
        )

    max_normal = int(
        len(attack_indices) * NORMAL_TO_ATTACK_RATIO
    )

    if len(normal_indices) > max_normal:

        normal_indices = np.random.choice(
            normal_indices,
            size=max_normal,
            replace=False
        )

    selected_indices = np.concatenate(
        [
            attack_indices,
            normal_indices,
        ]
    )

    np.random.shuffle(selected_indices)

    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]

    print("\n[+] After balancing:")
    print(f"    Total sequences  : {len(y_balanced):,}")
    print(
        f"    Normal sequences : "
        f"{np.sum(y_balanced == label_encoder.transform(['Normal'])[0]):,}"
    )
    print(
        f"    Attack sequences : "
        f"{np.sum(y_balanced != label_encoder.transform(['Normal'])[0]):,}"
    )

    return X_balanced, y_balanced


# ============================================================
# Main Processing Pipeline
# ============================================================

def process_unsw_dataset():

    print("=" * 70)
    print("NetForesight V2 - UNSW-NB15 Sequence Builder")
    print("=" * 70)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\n[1/7] Loading UNSW-NB15 datasets...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_PATH}"
        )

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Testing dataset not found:\n{TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    print(
        f"[+] Training rows : {len(train_df):,}"
    )

    print(
        f"[+] Testing rows  : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Target preparation
    # --------------------------------------------------------

    print("\n[2/7] Preparing target labels...")

    train_target_column = clean_target_column(train_df)
    test_target_column = clean_target_column(test_df)

    if train_target_column != test_target_column:
        raise ValueError(
            "Training and testing target columns do not match."
        )

    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    combined_targets = pd.concat(
        [
            train_df[train_target_column],
            test_df[test_target_column],
        ],
        ignore_index=True
    )

    label_encoder.fit(combined_targets)

    train_df["target"] = label_encoder.transform(
        train_df[train_target_column]
    )

    test_df["target"] = label_encoder.transform(
        test_df[test_target_column]
    )

    print(
        f"[+] Classes ({len(label_encoder.classes_)}):"
    )

    for index, class_name in enumerate(
        label_encoder.classes_
    ):
        print(
            f"    {index:2d} -> {class_name}"
        )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH
    )

    print(
        f"[✔] Saved label encoder -> "
        f"{LABEL_ENCODER_PATH}"
    )

    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    print("\n[3/7] Encoding categorical features...")

    train_df, test_df, categorical_encoders = (
        encode_categorical_features(
            train_df,
            test_df
        )
    )

    # --------------------------------------------------------
    # Feature preparation
    # --------------------------------------------------------

    print("\n[4/7] Preparing numerical feature matrix...")

    train_features, test_features, feature_names = (
        prepare_features(
            train_df,
            test_df
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # --------------------------------------------------------

    print("\n[5/7] Fitting feature scaler...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_features.values
    )

    X_test = scaler.transform(
        test_features.values
    )

    joblib.dump(
        scaler,
        FEATURE_SCALER_PATH
    )

    joblib.dump(
        feature_names,
        FEATURE_NAMES_PATH
    )

    print(
        f"[✔] Saved scaler -> "
        f"{FEATURE_SCALER_PATH}"
    )

    print(
        f"[✔] Saved feature schema -> "
        f"{FEATURE_NAMES_PATH}"
    )

    # --------------------------------------------------------
    # Build sequences separately
    # --------------------------------------------------------

    print("\n[6/7] Building temporal sequences...")

    y_train = train_df["target"].to_numpy(
        dtype=np.int64
    )

    y_test = test_df["target"].to_numpy(
        dtype=np.int64
    )

    X_train_seq, y_train_seq = build_sequences(
        X_train,
        y_train,
        SEQUENCE_LENGTH
    )

    X_test_seq, y_test_seq = build_sequences(
        X_test,
        y_test,
        SEQUENCE_LENGTH
    )

    print(
        f"[+] Training sequences: "
        f"{X_train_seq.shape}"
    )

    print(
        f"[+] Testing sequences : "
        f"{X_test_seq.shape}"
    )

    # --------------------------------------------------------
    # Balance TRAINING DATA ONLY
    # --------------------------------------------------------

    print("\n[7/7] Balancing training sequences...")

    X_train_seq, y_train_seq = balance_sequences(
        X_train_seq,
        y_train_seq,
        label_encoder
    )

    # Do NOT balance the test set.
    #
    # The test set should represent the original distribution.
    # Otherwise evaluation becomes artificially optimistic.

    # --------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_seq,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_seq,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test_seq,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test_seq,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    dataset = {
        "X_train": X_train_tensor,
        "y_train": y_train_tensor,
        "X_test": X_test_tensor,
        "y_test": y_test_tensor,

        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": len(feature_names),
        "num_classes": len(label_encoder.classes_),

        "feature_names": feature_names,
        "class_names": list(label_encoder.classes_),
    }

    torch.save(
        dataset,
        OUTPUT_SEQUENCE_PATH
    )

    print(
        f"\n[✔] Saved processed dataset -> "
        f"{OUTPUT_SEQUENCE_PATH}"
    )

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Training shape : "
        f"{tuple(X_train_tensor.shape)}"
    )

    print(
        f"Testing shape  : "
        f"{tuple(X_test_tensor.shape)}"
    )

    print(
        f"Features       : "
        f"{len(feature_names)}"
    )

    print(
        f"Classes        : "
        f"{len(label_encoder.classes_)}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    process_unsw_dataset()