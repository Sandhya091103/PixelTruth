"""
PixelTruth - model evaluation on the held-out test set.

How to run:
    python -m src.evaluate --model efficientnet     # main model
    python -m src.evaluate --model custom           # baseline CNN

Produces:
  - Console: accuracy, precision, recall, F1 (classification report), AUC
  - outputs/confusion_matrix_<model>.png
  - outputs/roc_curve_<model>.png

Label convention (see data.py): real = 0, fake = 1.
The positive class is "fake" (deepfake detected).
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

from . import config as cfg
from . import data as data_mod
from . import models as models_mod
from . import utils


# ----------------------------------------------------------------------
# Load a trained model from its checkpoint
# ----------------------------------------------------------------------
def load_model(which):
    """Build the architecture and load the saved best weights."""
    if which == "custom":
        model = models_mod.build_custom_cnn()
        ckpt = cfg.CUSTOM_CNN_PATH
    else:
        # freeze_base flag does not change the parameter keys, only
        # requires_grad - so the saved state_dict loads fine either way.
        model = models_mod.build_efficientnet(freeze_base=False)
        ckpt = cfg.EFFICIENTNET_PATH

    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Train the model first.")

    model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded weights from {ckpt.name}")
    return model


# ----------------------------------------------------------------------
# Run the model over the whole test set, collect labels + probabilities
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader):
    """Return (y_true, y_prob) numpy arrays. y_prob = P(fake)."""
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(cfg.DEVICE, non_blocking=True)
        with torch.autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP):
            logits = model(images)
        probs = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)            # rows=true, cols=pred
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {save_path}")


def plot_roc_curve(y_true, y_prob, auc, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve -> {save_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a PixelTruth model")
    parser.add_argument("--model", choices=["custom", "efficientnet"],
                        default="efficientnet", help="which model to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability threshold for the 'fake' class")
    args = parser.parse_args()

    utils.set_seed()
    utils.device_info()

    print("\nLoading test data...")
    _, _, test_loader = data_mod.get_dataloaders()
    print(f"Test images: {len(test_loader.dataset)}")

    model = load_model(args.model)

    print("\nRunning inference on the test set...")
    y_true, y_prob = predict(model, test_loader)
    y_pred = (y_prob >= args.threshold).astype(int)

    # --- Metrics ---
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("\n================ TEST RESULTS ================")
    print(f"Model:     {args.model}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification report (positive class = fake):")
    print(classification_report(
        y_true, y_pred, target_names=cfg.CLASS_NAMES, digits=4))

    # --- Plots ---
    cm_path  = cfg.OUTPUT_DIR / f"confusion_matrix_{args.model}.png"
    roc_path = cfg.OUTPUT_DIR / f"roc_curve_{args.model}.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)
    plot_roc_curve(y_true, y_prob, auc, roc_path)

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
