"""
PixelTruth - misclassification analysis.

Find the test images the model got wrong and show them, split into:
  - "Fakes that fooled the model"  (true: fake, predicted: real)
  - "Reals flagged as fake"        (true: real, predicted: fake)

We sort each group by the model's confidence in its WRONG answer, so the
most confident mistakes (the most interesting failures) show first.

How to run:
    python -m src.misclassify
    python -m src.misclassify --per-group 5

Output:
    outputs/misclassified.png

Label convention: real = 0, fake = 1.
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from . import config as cfg
from . import data as data_mod
from . import models as models_mod
from . import utils


# ----------------------------------------------------------------------
# Collect predictions together with each image's file path
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_with_paths(model, loader):
    """Return y_true, y_prob (P(fake)), and the list of file paths.
    Test loader uses shuffle=False, so order matches dataset.samples."""
    paths = [p for (p, _) in loader.dataset.samples]

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
    return y_true, y_prob, paths


def show_image(ax, path, title):
    img = Image.open(path).convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Misclassification analysis")
    parser.add_argument("--per-group", type=int, default=5,
                        help="how many wrong images to show per group")
    args = parser.parse_args()

    utils.set_seed()
    utils.device_info()

    print("\nLoading test data...")
    _, _, test_loader = data_mod.get_dataloaders()
    print(f"Test images: {len(test_loader.dataset)}")

    model = models_mod.build_efficientnet(freeze_base=False)
    model.load_state_dict(torch.load(cfg.EFFICIENTNET_PATH, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded {cfg.EFFICIENTNET_PATH.name}")

    print("\nRunning inference on the test set...")
    y_true, y_prob, paths = predict_with_paths(model, test_loader)
    y_pred = (y_prob >= 0.5).astype(int)

    wrong = np.where(y_pred != y_true)[0]
    print(f"Total misclassified: {len(wrong)} / {len(y_true)} "
          f"({len(wrong) / len(y_true) * 100:.2f}%)")

    # Group 1: true fake (1), predicted real -> low prob_fake, fooled the model
    fooled = [i for i in wrong if y_true[i] == 1]
    fooled.sort(key=lambda i: y_prob[i])            # most "confidently real" first

    # Group 2: true real (0), predicted fake -> high prob_fake, false alarm
    false_alarm = [i for i in wrong if y_true[i] == 0]
    false_alarm.sort(key=lambda i: -y_prob[i])      # most "confidently fake" first

    print(f"  Fakes that fooled the model: {len(fooled)}")
    print(f"  Reals flagged as fake:       {len(false_alarm)}")

    k = args.per_group
    groups = [
        ("Fakes that FOOLED the model (true: fake -> pred: real)", fooled[:k]),
        ("Reals flagged as FAKE (true: real -> pred: fake)", false_alarm[:k]),
    ]

    fig, axes = plt.subplots(2, k, figsize=(2.4 * k, 5.4))
    if k == 1:
        axes = axes.reshape(2, 1)

    for row, (title, idxs) in enumerate(groups):
        for col in range(k):
            ax = axes[row, col]
            if col < len(idxs):
                i = idxs[col]
                conf_fake = y_prob[i] * 100
                show_image(ax, paths[i],
                           f"P(fake)={conf_fake:.1f}%")
            else:
                ax.axis("off")
        # row label on the left-most axis
        axes[row, 0].set_ylabel(title, fontsize=9)

    # Put group titles as text above each row band
    fig.suptitle("Misclassified test images (most confident mistakes first)",
                 fontsize=12)
    for row, (title, _) in enumerate(groups):
        axes[row, 0].annotate(
            title, xy=(0, 1.18), xycoords="axes fraction",
            fontsize=10, fontweight="bold", ha="left", va="bottom")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = cfg.OUTPUT_DIR / "misclassified.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved misclassified grid -> {out}")
    print("[DONE] Misclassification analysis complete.")


if __name__ == "__main__":
    main()
