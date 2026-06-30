"""
PixelTruth - Grad-CAM visualization.

Grad-CAM (Gradient-weighted Class Activation Mapping) shows WHICH regions of
a face the model focuses on when deciding real vs fake. Useful for
interpretability - e.g. does the model look at skin texture, eyes, or
background artifacts?

How to run:
    python -m src.gradcam                 # default: a few real + a few fake
    python -m src.gradcam --per-class 4   # 4 images per class

Output:
    outputs/gradcam_examples.png  (rows of: original | Grad-CAM overlay)

Target layer: EfficientNetB0's last convolutional block (model.features[-1]),
the deepest feature map - standard choice for Grad-CAM on CNNs.
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from . import config as cfg
from . import models as models_mod
from . import utils


# A target that returns the model's single logit for one sample.
# Higher logit -> more "fake", so the CAM highlights regions driving the
# fake decision. (Standard ClassifierOutputTarget expects multi-class logits;
# our model has one output neuron, so we use this small custom target.)
class FakeLogitTarget:
    def __call__(self, model_output):
        # model_output is shape (1,) for a single sample
        return model_output[0]


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------
_to_tensor = transforms.ToTensor()
_normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)


def load_image(path):
    """Return (rgb_float, input_tensor).
    rgb_float : HxWx3 in [0,1] for display / overlay
    input_tensor : 1x3xHxW normalized, on the device
    """
    img = Image.open(path).convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
    rgb_float = np.asarray(img, dtype=np.float32) / 255.0
    tensor = _normalize(_to_tensor(img)).unsqueeze(0).to(cfg.DEVICE)
    return rgb_float, tensor


def pick_samples(per_class):
    """Pick `per_class` image paths from each of test/real and test/fake."""
    samples = []
    for cls in ["real", "fake"]:                     # display order
        folder = cfg.TEST_DIR / cls
        files = sorted(folder.glob("*.jpg"))
        # spread picks across the folder instead of just the first few
        step = max(1, len(files) // (per_class + 1))
        chosen = [files[step * (i + 1)] for i in range(per_class)]
        for f in chosen:
            samples.append((f, cls))
    return samples


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for PixelTruth")
    parser.add_argument("--per-class", type=int, default=3,
                        help="how many images per class (real/fake)")
    args = parser.parse_args()

    utils.set_seed()
    utils.device_info()

    # --- Load the trained EfficientNet ---
    model = models_mod.build_efficientnet(freeze_base=False)
    model.load_state_dict(torch.load(cfg.EFFICIENTNET_PATH, map_location=cfg.DEVICE))
    model.eval()
    print(f"Loaded {cfg.EFFICIENTNET_PATH.name}")

    # Last conv block - the layer Grad-CAM hooks into
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    samples = pick_samples(args.per_class)
    n = len(samples)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, (path, true_cls) in enumerate(samples):
        rgb_float, input_tensor = load_image(path)

        # Predicted probability of "fake"
        with torch.no_grad():
            logit = model(input_tensor)
            prob_fake = torch.sigmoid(logit).item()
        pred_cls = "fake" if prob_fake >= 0.5 else "real"
        conf = prob_fake if pred_cls == "fake" else 1 - prob_fake

        # Grad-CAM heatmap (full precision - it needs gradients)
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[FakeLogitTarget()])[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        ok = "OK" if pred_cls == true_cls else "WRONG"
        axes[row, 0].imshow(rgb_float)
        axes[row, 0].set_title(f"Original (true: {true_cls})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"Grad-CAM  pred: {pred_cls} ({conf*100:.1f}%) [{ok}]")
        axes[row, 1].axis("off")

    fig.suptitle("Grad-CAM - where the model looks (red = high attention)",
                 fontsize=13)
    fig.tight_layout()
    out = cfg.OUTPUT_DIR / "gradcam_examples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM examples -> {out}")
    print("[DONE] Grad-CAM complete.")


if __name__ == "__main__":
    main()
