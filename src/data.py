"""
PixelTruth — data pipeline.

Folder se images load karo -> transform (resize/normalize/augment) ->
DataLoader banao (train/valid/test).

Dataset structure (ImageFolder isi format ko samajhta hai):
    datasets/real-vs-fake/
        train/  fake/  real/
        valid/  fake/  real/
        test/   fake/  real/

NOTE label mapping:
    ImageFolder folders ko alphabetical order me label karta hai:
        fake -> 0 ,  real -> 1
    Hum chahte hain  real=0, fake=1  (fake = positive/deepfake).
    Isliye neeche ek remap (target_transform) lagate hain.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from . import config as cfg


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
# Common normalization (ImageNet stats — pretrained EfficientNet ke liye zaruri)
_normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)

# TRAIN — augmentation ke saath (sirf training pe, taaki model overfit na ho)
# NOTE: deepfake ke clues texture/noise me hote hain, isliye aggressive
# distortions (heavy blur/denoise) NAHI lagate — sirf safe augmentations.
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),                 # [0,255] PIL -> [0,1] tensor (C,H,W)
    _normalize,
])

# VALID / TEST — koi augmentation nahi, sirf resize + normalize
eval_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    _normalize,
])


# ----------------------------------------------------------------------
# Label remap:  ImageFolder(fake=0, real=1)  ->  hum chahte real=0, fake=1
# ----------------------------------------------------------------------
def _remap_label(y):
    # ImageFolder: fake=0, real=1  ->  flip karke real=0, fake=1
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
    """train/valid/test ke liye 3 DataLoaders return karta hai."""
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
    print("Class mapping (humara): real=0, fake=1")
    print(f"Train batches: {len(train_loader)}  ({len(train_loader.dataset)} images)")
    print(f"Valid batches: {len(valid_loader)}  ({len(valid_loader.dataset)} images)")
    print(f"Test  batches: {len(test_loader)}  ({len(test_loader.dataset)} images)")

    # ek batch nikaalo, shape verify karo
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {tuple(images.shape)}   (B, C, H, W)")
    print(f"Batch labels shape:       {tuple(labels.shape)}")
    print(f"Label sample (0=real,1=fake): {labels[:8].tolist()}")
    print(f"Pixel range after normalize: [{images.min():.2f}, {images.max():.2f}]")
