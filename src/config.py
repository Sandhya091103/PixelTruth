"""
PixelTruth — central configuration.

Saari settings ek hi jagah. Kahin aur kuch hardcode nahi karna —
path/size/batch/lr badalna ho to sirf yahan badlo.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths  (project root ke relative — kahin se bhi chalao, kaam karega)
# ----------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent.parent      # PixelTruth/
DATA_DIR    = ROOT_DIR / "datasets" / "real-vs-fake"      # actual dataset path
TRAIN_DIR   = DATA_DIR / "train"
VALID_DIR   = DATA_DIR / "valid"
TEST_DIR    = DATA_DIR / "test"

MODEL_DIR   = ROOT_DIR / "model"                          # saved models
OUTPUT_DIR  = ROOT_DIR / "outputs"                        # plots, heatmaps

# Banao agar exist nahi karte (datasets/ ko nahi chhedte — wo already hai)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Classes  (folder name -> label)
#   real = 0,  fake = 1   ->  positive class = "deepfake detected"
# ----------------------------------------------------------------------
CLASS_NAMES = ["real", "fake"]      # index = label  (real=0, fake=1)
NUM_CLASSES = 1                     # binary -> single sigmoid output

# ----------------------------------------------------------------------
# Image settings
# ----------------------------------------------------------------------
IMG_SIZE = 224                      # EfficientNetB0 ka native input (224x224)

# ImageNet normalization stats (pretrained models inhi pe trained hain)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------
BATCH_SIZE   = 32
NUM_WORKERS  = 4                    # DataLoader parallel loading (Windows-safe)

# Phase 1 — feature extraction (base frozen, sirf head train)
LR_HEAD      = 1e-3
EPOCHS_HEAD  = 5

# Phase 2 — fine-tuning (top layers unfrozen, low lr)
LR_FINETUNE      = 1e-5
EPOCHS_FINETUNE  = 10
UNFREEZE_LAST_N  = 30              # base ke aakhri kitne blocks unfreeze karne hain

# Callbacks / training control
EARLY_STOP_PATIENCE  = 5           # itne epoch improve na ho to ruk jao
LR_PATIENCE          = 3           # itne epoch baad lr aadha karo
LR_FACTOR            = 0.5

# ----------------------------------------------------------------------
# Reproducibility & device
# ----------------------------------------------------------------------
SEED = 42

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixed precision (GPU pe fast training) — GPU available ho to hi on
USE_AMP = torch.cuda.is_available()

# ----------------------------------------------------------------------
# Saved model paths
# ----------------------------------------------------------------------
CUSTOM_CNN_PATH   = MODEL_DIR / "pixeltruth_custom_cnn.pt"
EFFICIENTNET_PATH = MODEL_DIR / "pixeltruth_efficientnet.pt"
