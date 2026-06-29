"""
PixelTruth - small helpers (seed, device info, plotting).
"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from . import config as cfg


def set_seed(seed=cfg.SEED):
    """Reproducibility - same result on every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_info():
    """Print GPU/CPU info."""
    print(f"Device: {cfg.DEVICE}")
    if cfg.DEVICE.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"AMP (mixed precision): {cfg.USE_AMP}")


def plot_history(history, title, save_path):
    """
    Training curves - loss + accuracy (train vs valid).
    history: dict with keys train_loss, val_loss, train_acc, val_acc (lists).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"],   "o-", label="valid")
    ax1.set_title("Loss");     ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["val_acc"],   "o-", label="valid")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves -> {save_path}")
