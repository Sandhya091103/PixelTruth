"""
PixelTruth - model architectures.

Two models:
  1. build_custom_cnn()    -> CNN from scratch (baseline)
  2. build_efficientnet()  -> EfficientNetB0 transfer learning (production)

Both output a single logit (for BCEWithLogitsLoss - sigmoid is applied inside
the loss, which is numerically stable). For a probability use torch.sigmoid(logit).
"""
import torch
import torch.nn as nn
from torchvision import models

from . import config as cfg


# ======================================================================
# 1. Custom CNN  (baseline - from scratch)
# ======================================================================
class CustomCNN(nn.Module):
    """
    Input (3 x 224 x 224)
      -> Conv(32)  + BN + ReLU + MaxPool
      -> Conv(64)  + BN + ReLU + MaxPool
      -> Conv(128) + BN + ReLU + MaxPool
      -> GlobalAvgPool
      -> Dense(256) + ReLU + Dropout(0.5)
      -> Dense(1)                          (logit)
    """
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),           # halves H, W
            )

        self.features = nn.Sequential(
            block(3,   32),    # 224 -> 112
            block(32,  64),    # 112 -> 56
            block(64, 128),    # 56  -> 28
        )
        self.gap = nn.AdaptiveAvgPool2d(1)     # GlobalAveragePooling -> (B,128,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # (B, 128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),                 # single logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_custom_cnn():
    """Return a Custom CNN instance on the configured device."""
    model = CustomCNN().to(cfg.DEVICE)
    return model


# ======================================================================
# 2. EfficientNetB0  (transfer learning)
# ======================================================================
def build_efficientnet(freeze_base=True):
    """
    EfficientNetB0 (ImageNet pretrained) + our binary head.

    freeze_base=True  -> Phase 1 (feature extraction): base frozen,
                         only the new head trains.
    For fine-tuning later, use unfreeze_top().
    """
    # Load the base with ImageNet weights
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # --- Phase 1: freeze the whole base ---
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    # --- Drop the old 1000-class head, attach our binary head ---
    # efficientnet_b0.classifier = Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features          # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(128, 1),                                 # single logit
    )

    return model.to(cfg.DEVICE)


def unfreeze_top(model, last_n=cfg.UNFREEZE_LAST_N):
    """
    Phase 2 (fine-tuning): unfreeze the base's last `last_n` layers.
    Keep BatchNorm layers frozen + in eval mode (transfer-learning best practice).
    """
    # features is an nn.Sequential - unfreeze its last children
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-last_n:]:
        for param in block.parameters():
            param.requires_grad = True

    # Keep BatchNorm layers frozen (so running stats don't get corrupted)
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    return model


# ----------------------------------------------------------------------
# Helper: how many params are trainable (for a sanity check)
# ----------------------------------------------------------------------
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------
# Quick check:  python -m src.models
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(cfg.DEVICE)

    print("=== Custom CNN ===")
    cnn = build_custom_cnn()
    out = cnn(dummy)
    t, tr = count_params(cnn)
    print(f"output shape: {tuple(out.shape)}   params: {t:,}  trainable: {tr:,}")

    print("\n=== EfficientNetB0 (Phase 1: base frozen) ===")
    eff = build_efficientnet(freeze_base=True)
    out = eff(dummy)
    t, tr = count_params(eff)
    print(f"output shape: {tuple(out.shape)}   params: {t:,}  trainable: {tr:,}")

    print("\n=== EfficientNetB0 (Phase 2: top unfrozen) ===")
    eff = unfreeze_top(eff)
    t, tr = count_params(eff)
    print(f"params: {t:,}  trainable: {tr:,}")
