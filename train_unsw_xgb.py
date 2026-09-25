import os
import joblib
import torch
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/unsw_sequences.pt"
MODEL_PATH = "backend/xgboost_model.pkl"

RANDOM_SEED = 42


# Keep this reasonable for an 8 GB RAM CPU machine.
N_ESTIMATORS = 200
MAX_DEPTH = 5
LEARNING_RATE = 0.05

SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

N_JOBS = 2


# ============================================================
# Load data
# ============================================================

print("\n" + "=" * 70)
print("NetForesight V2 - XGBoost Training")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}\n"
        "Run build_unsw_sequences.py first."
    )

print(f"\n[+] Loading dataset from {DATA_PATH}")

data = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False
)

X_train_seq = data["X_train"].float().numpy()
y_train = data["y_train"].numpy()

X_test_seq = data["X_test"].float().numpy()
y_test = data["y_test"].numpy()

class_names = data["class_names"]

print(f"[+] Training sequences : {len(X_train_seq)}")
print(f"[+] Testing sequences  : {len(X_test_seq)}")
print(f"[+] Sequence length    : {X_train_seq.shape[1]}")
print(f"[+] Feature dimension  : {X_train_seq.shape[2]}")
print(f"[+] Number of classes  : {len(class_names)}")


# ============================================================
# XGBoost uses the latest frame
# ============================================================
#
# The Transformer sees all 5 timesteps.
#
# XGBoost is being used as a fast single-frame baseline:
#
#     [t-4, t-3, t-2, t-1, t] -> use only t
#
# This also makes it useful later as a fallback / comparison
# model in the backend.
# ============================================================

print("\n[+] Extracting latest timestep from every sequence...")

X_train = X_train_seq[:, -1, :]
X_test = X_test_seq[:, -1, :]

print(f"[+] X_train shape: {X_train.shape}")
print(f"[+] X_test shape : {X_test.shape}")


# ============================================================
# Class distribution
# ============================================================

print("\n[+] Training class distribution:")

for class_id, class_name in enumerate(class_names):
    count = int((y_train == class_id).sum())

    print(
        f"    {class_id:2d} | "
        f"{class_name:<20} | "
        f"{count}"
    )


# ============================================================
# Create XGBoost model
# ============================================================

print("\n[+] Creating XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,

    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,

    objective="multi:softprob",
    eval_metric="mlogloss",

    random_state=RANDOM_SEED,

    tree_method="hist",

    n_jobs=N_JOBS,

    reg_lambda=1.0,
)


# ============================================================
# Train
# ============================================================

print("\n[+] Training XGBoost...")
print("[+] CPU-friendly configuration enabled.\n")

model.fit(
    X_train,
    y_train,
)

print("\n[✔] XGBoost training completed.")


# ============================================================
# Evaluate
# ============================================================

print("\n" + "=" * 70)
print("Evaluation")
print("=" * 70)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names)))
)

print("Confusion Matrix:")
print("-" * 70)

print(cm)


# ============================================================
# Save model
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 70)
print(f"[✔] Saved XGBoost model to:")
print(f"    {MODEL_PATH}")
print("=" * 70)

print("\n[+] XGBoost training finished successfully.")