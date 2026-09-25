import os
import random
import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/unsw_sequences.pt"

LABEL_ENCODER_PATH = "backend/label_encoder.pkl"

MODEL_PATH = "backend/transformer_forecaster.pt"

SEQUENCE_LENGTH = 5

BATCH_SIZE = 128

EPOCHS = 15

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 1e-4

D_MODEL = 64

N_HEADS = 4

N_LAYERS = 2

DROPOUT = 0.15

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# Transformer Model
# ============================================================

class AttackForecasterTransformer(nn.Module):

    def __init__(
        self,
        feature_dim,
        seq_len=5,
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.15,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Convert raw network feature vectors into the
        # transformer's internal representation.
        self.embedding = nn.Linear(
            feature_dim,
            d_model
        )

        # Learnable positional information.
        #
        # The transformer itself does not inherently know that
        # position 0 happened before position 4.
        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                seq_len,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # Instead of flattening all transformer outputs,
        # use the final timestep as the representation of
        # the observed sequence.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):

        # x:
        # [batch, sequence_length, feature_dim]

        x = self.embedding(x)

        x = x + self.positional_embedding

        x = self.transformer(x)

        # Normalize transformer output.
        x = self.norm(x)

        # The final timestep represents the latest observed
        # network activity.
        x = x[:, -1, :]

        return self.classifier(x)


# ============================================================
# Class Weight Calculation
# ============================================================

def calculate_class_weights(y, num_classes):
    """
    Calculates inverse-frequency class weights.

    Rare attack categories receive larger weights.
    Common categories receive smaller weights.

    This prevents the loss function from treating:

        50,000 Normal samples

    and

        500 rare attack samples

    as equally informative.
    """

    counts = np.bincount(
        y,
        minlength=num_classes
    ).astype(np.float64)

    total = counts.sum()

    weights = np.zeros(
        num_classes,
        dtype=np.float32
    )

    for class_id in range(num_classes):

        if counts[class_id] > 0:

            weights[class_id] = (
                total /
                (num_classes * counts[class_id])
            )

    # Normalize the weights so that the average weight
    # of existing classes is approximately 1.
    non_zero = weights > 0

    if np.any(non_zero):

        weights[non_zero] /= weights[non_zero].mean()

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    X,
    y,
    class_names,
):
    """
    Evaluates the model on the untouched test dataset.
    """

    model.eval()

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(DEVICE)

            outputs = model(batch_x)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.numpy()
            )

    all_predictions = np.asarray(
        all_predictions
    )

    all_targets = np.asarray(
        all_targets
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy          : {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1          : {macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1       : {weighted_f1 * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            all_targets,
            all_predictions,
            labels=list(range(len(class_names))),
        )
    )

    print("=" * 70)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# Training
# ============================================================

def train():

    print("=" * 70)
    print("NetForesight V2 - Transformer Training")
    print("=" * 70)

    os.makedirs(
        "backend",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load processed dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}\n\n"
            "Run build_unsw_sequences.py first."
        )

    print("\n[1/6] Loading processed sequences...")

    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=False,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    sequence_length = int(
        data.get(
            "sequence_length",
            SEQUENCE_LENGTH
        )
    )

    feature_dim = int(
        data["feature_dim"]
    )

    num_classes = int(
        data["num_classes"]
    )

    class_names = data["class_names"]

    print(
        f"[+] Training sequences : {len(X_train):,}"
    )

    print(
        f"[+] Testing sequences  : {len(X_test):,}"
    )

    print(
        f"[+] Features            : {feature_dim}"
    )

    print(
        f"[+] Sequence length     : {sequence_length}"
    )

    print(
        f"[+] Classes             : {num_classes}"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if X_train.ndim != 3:

        raise ValueError(
            f"Expected training data with 3 dimensions, "
            f"got {X_train.ndim}."
        )

    if X_test.ndim != 3:

        raise ValueError(
            f"Expected testing data with 3 dimensions, "
            f"got {X_test.ndim}."
        )

    if X_train.shape[1] != sequence_length:

        raise ValueError(
            "Training sequence length does not match metadata."
        )

    if X_train.shape[2] != feature_dim:

        raise ValueError(
            "Training feature dimension does not match metadata."
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n[2/6] Checking class distribution...")

    unique, counts = torch.unique(
        y_train,
        return_counts=True
    )

    for class_id, count in zip(
        unique.tolist(),
        counts.tolist()
    ):

        class_name = (
            class_names[class_id]
            if class_id < len(class_names)
            else str(class_id)
        )

        print(
            f"    {class_id:2d} | "
            f"{class_name:<20} | "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------------

    print("\n[3/6] Calculating class weights...")

    class_weights = calculate_class_weights(
        y_train.numpy(),
        num_classes,
    ).to(DEVICE)

    print(
        "[+] Class weights:"
    )

    for index, weight in enumerate(
        class_weights.tolist()
    ):

        print(
            f"    {index:2d} | "
            f"{class_names[index]:<20} | "
            f"{weight:.4f}"
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n[4/6] Creating Transformer...")

    model = AttackForecasterTransformer(
        feature_dim=feature_dim,
        seq_len=sequence_length,
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[+] Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n[5/6] Training model...")
    print(
        f"[+] Epochs       : {EPOCHS}"
    )

    print(
        f"[+] Batch size   : {BATCH_SIZE}"
    )

    print(
        f"[+] Learning rate: {LEARNING_RATE}"
    )

    best_loss = float("inf")

    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            # Prevent occasional exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += (
                loss.item() *
                batch_y.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

        epoch_loss = (
            running_loss /
            max(total, 1)
        )

        epoch_accuracy = (
            correct /
            max(total, 1)
        )

        scheduler.step(
            epoch_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_accuracy * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best training state.
        if epoch_loss < best_loss:

            best_loss = epoch_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print("\n[6/6] Evaluating final model...")

    metrics = evaluate(
        model,
        X_test,
        y_test,
        class_names,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "feature_dim":
            feature_dim,

        "sequence_length":
            sequence_length,

        "num_classes":
            num_classes,

        "class_names":
            class_names,

        "d_model":
            D_MODEL,

        "nhead":
            N_HEADS,

        "num_layers":
            N_LAYERS,

        "dropout":
            DROPOUT,

        "metrics":
            metrics,
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print(
        f"\n[✔] Transformer saved to:"
        f"\n    {MODEL_PATH}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    train()