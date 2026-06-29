"""
PixelTruth - central configuration.

All settings live here. Don't hardcode anything elsewhere - to change a
path/size/batch/lr, change it only here.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths  (relative to project root - works from anywhere)
# ----------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent.parent      # PixelTruth/
DATA_DIR    = ROOT_DIR / "datasets" / "real-vs-fake"      # actual dataset path
TRAIN_DIR   = DATA_DIR / "train"
VALID_DIR   = DATA_DIR / "valid"
TEST_DIR    = DATA_DIR / "test"

MODEL_DIR   = ROOT_DIR / "model"                          # saved models
OUTPUT_DIR  = ROOT_DIR / "outputs"                        # plots, heatmaps

# Create if missing (we never touch datasets/ - it already exists)
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
IMG_SIZE = 224                      # EfficientNetB0 native input (224x224)

# ImageNet normalization stats (pretrained models were trained on these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------
BATCH_SIZE   = 32
NUM_WORKERS  = 4                    # DataLoader parallel loading (Windows-safe)

# Phase 1 - feature extraction (base frozen, train head only)
LR_HEAD      = 1e-3
EPOCHS_HEAD  = 5

# Phase 2 - fine-tuning (top layers unfrozen, low lr)
LR_FINETUNE      = 1e-5
EPOCHS_FINETUNE  = 10
UNFREEZE_LAST_N  = 30              # how many of the base's last blocks to unfreeze

# Callbacks / training control
EARLY_STOP_PATIENCE  = 5           # stop if no improvement for this many epochs
LR_PATIENCE          = 3           # halve lr after this many stagnant epochs
LR_FACTOR            = 0.5

# ----------------------------------------------------------------------
# Reproducibility & device
# ----------------------------------------------------------------------
SEED = 42

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mixed precision (faster training on GPU) - only on if GPU is available
USE_AMP = torch.cuda.is_available()

# ----------------------------------------------------------------------
# Saved model paths
# ----------------------------------------------------------------------
CUSTOM_CNN_PATH   = MODEL_DIR / "pixeltruth_custom_cnn.pt"
EFFICIENTNET_PATH = MODEL_DIR / "pixeltruth_efficientnet.pt"
