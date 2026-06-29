"""
PixelTruth - data pipeline.

Load images from folders -> transform (resize/normalize/augment) ->
build DataLoaders (train/valid/test).

Dataset structure (ImageFolder understands this layout):
    datasets/real-vs-fake/
        train/  fake/  real/
        valid/  fake/  real/
        test/   fake/  real/

NOTE on label mapping:
    ImageFolder labels folders in alphabetical order:
        fake -> 0 ,  real -> 1
    We want  real=0, fake=1  (fake = positive/deepfake).
    So we apply a remap (target_transform) below.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from . import config as cfg


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
# Common normalization (ImageNet stats - required for pretrained EfficientNet)
_normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)

# TRAIN - with augmentation (train only, to reduce overfitting).
# NOTE: deepfake clues live in texture/noise, so we avoid aggressive
# distortions (heavy blur/denoise) - only safe augmentations.
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),                 # PIL [0,255] -> tensor [0,1] (C,H,W)
    _normalize,
])

# VALID / TEST - no augmentation, just resize + normalize
eval_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    _normalize,
])


# ----------------------------------------------------------------------
# Label remap:  ImageFolder(fake=0, real=1)  ->  we want real=0, fake=1
# ----------------------------------------------------------------------
def _remap_label(y):
    # ImageFolder: fake=0, real=1  ->  flip to real=0, fake=1
    return 1 - y


# ----------------------------------------------------------------------
# Dataset builders
# ----------------------------------------------------------------------
def _make_dataset(root, transform):
    return datasets.ImageFolder(
        root=str(root),
        transform=transform,
        target_transform=_remap_label,
    )


def get_dataloaders(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS):
    """Return 3 DataLoaders for train/valid/test."""
    train_ds = _make_dataset(cfg.TRAIN_DIR, train_transform)
    valid_ds = _make_dataset(cfg.VALID_DIR, eval_transform)
    test_ds  = _make_dataset(cfg.TEST_DIR,  eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, valid_loader, test_loader


# ----------------------------------------------------------------------
# Quick sanity check:  python -m src.data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders()
    print("Class mapping (ours): real=0, fake=1")
    print(f"Train batches: {len(train_loader)}  ({len(train_loader.dataset)} images)")
    print(f"Valid batches: {len(valid_loader)}  ({len(valid_loader.dataset)} images)")
    print(f"Test  batches: {len(test_loader)}  ({len(test_loader.dataset)} images)")

    # Grab one batch and verify the shape
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {tuple(images.shape)}   (B, C, H, W)")
    print(f"Batch labels shape:       {tuple(labels.shape)}")
    print(f"Label sample (0=real,1=fake): {labels[:8].tolist()}")
    print(f"Pixel range after normalize: [{images.min():.2f}, {images.max():.2f}]")
