"""
PixelTruth — model architectures.

Do models:
  1. build_custom_cnn()    -> scratch se CNN (baseline)
  2. build_efficientnet()  -> EfficientNetB0 transfer learning (production)

Dono ka output: 1 logit (BCEWithLogitsLoss ke liye — sigmoid loss ke andar
lagta hai, numerically stable). Probability chahiye to torch.sigmoid(logit).
"""
import torch
import torch.nn as nn
from torchvision import models

from . import config as cfg


# ======================================================================
# 1. Custom CNN  (baseline — scratch se)
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
                nn.MaxPool2d(2),           # H,W aadha
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
    """Custom CNN instance device pe return karta hai."""
    model = CustomCNN().to(cfg.DEVICE)
    return model


# ======================================================================
# 2. EfficientNetB0  (transfer learning)
# ======================================================================
def build_efficientnet(freeze_base=True):
    """
    EfficientNetB0 (ImageNet pretrained) + apna binary head.

    freeze_base=True  -> Phase 1 (feature extraction): base frozen,
                         sirf naya head train hoga.
    Baad me fine-tuning ke liye unfreeze_top() use karna.
    """
    # ImageNet weights ke saath base load karo
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # --- Phase 1: poora base freeze ---
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    # --- Purana 1000-class head hatao, apna binary head lagao ---
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
    Phase 2 (fine-tuning): base ke aakhri `last_n` layers unfreeze karo.
    BatchNorm layers ko frozen + eval mode me rakho (transfer learning best practice).
    """
    # features ek nn.Sequential hai — uske children unfreeze karo
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-last_n:]:
        for param in block.parameters():
            param.requires_grad = True

    # BatchNorm layers frozen rakho (running stats na bigden)
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    return model


# ----------------------------------------------------------------------
# Helper: kitne params trainable hain (sanity check ke liye)
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
