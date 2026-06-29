"""
PixelTruth - data exploration (Phase 2).

Run with:
    python -m src.explore

What it does:
  1. Count images per split/class - count table
  2. Real vs Fake sample image grid -> outputs/sample_images.png
  3. Image size / channel / mode check
"""
import random

from PIL import Image
import matplotlib.pyplot as plt

from . import config as cfg


# ----------------------------------------------------------------------
# 1. Class counts
# ----------------------------------------------------------------------
def count_images():
    print("=" * 48)
    print(f"{'split':<8}{'real':>12}{'fake':>12}{'total':>14}")
    print("-" * 48)
    grand = 0
    for split, folder in [("train", cfg.TRAIN_DIR),
                          ("valid", cfg.VALID_DIR),
                          ("test",  cfg.TEST_DIR)]:
        n_real = sum(1 for _ in (folder / "real").glob("*"))
        n_fake = sum(1 for _ in (folder / "fake").glob("*"))
        total = n_real + n_fake
        grand += total
        print(f"{split:<8}{n_real:>12,}{n_fake:>12,}{total:>14,}")
    print("-" * 48)
    print(f"{'GRAND TOTAL':<32}{grand:>16,}")
    print("=" * 48)


# ----------------------------------------------------------------------
# 2. Sample image grid (real vs fake)
# ----------------------------------------------------------------------
def sample_grid(n_per_class=5, seed=cfg.SEED):
    rng = random.Random(seed)

    def pick(folder, k):
        files = list((folder).glob("*"))
        return rng.sample(files, k)

    real_imgs = pick(cfg.TRAIN_DIR / "real", n_per_class)
    fake_imgs = pick(cfg.TRAIN_DIR / "fake", n_per_class)

    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 2.2, 5))
    for j, path in enumerate(real_imgs):
        axes[0, j].imshow(Image.open(path).convert("RGB"))
        axes[0, j].axis("off")
    for j, path in enumerate(fake_imgs):
        axes[1, j].imshow(Image.open(path).convert("RGB"))
        axes[1, j].axis("off")

    # Row labels (axis is off, so draw them as figure text)
    fig.text(0.02, 0.74, "REAL", fontsize=14, fontweight="bold",
             color="green", rotation=90, va="center")
    fig.text(0.02, 0.30, "FAKE", fontsize=14, fontweight="bold",
             color="red", rotation=90, va="center")

    fig.suptitle("PixelTruth - Real vs AI-generated (StyleGAN) faces",
                 fontsize=13)
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])

    save_path = cfg.OUTPUT_DIR / "sample_images.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample grid -> {save_path}")


# ----------------------------------------------------------------------
# 3. Image properties check
# ----------------------------------------------------------------------
def check_image_props(n=200, seed=cfg.SEED):
    rng = random.Random(seed)
    sizes, modes = set(), set()
    for cls in ["real", "fake"]:
        files = list((cfg.TRAIN_DIR / cls).glob("*"))
        for path in rng.sample(files, min(n, len(files))):
            with Image.open(path) as im:
                sizes.add(im.size)       # (W, H)
                modes.add(im.mode)       # RGB / L / ...
    print(f"\nImage properties (sampled {n} per class):")
    print(f"  unique sizes : {sorted(sizes)}")
    print(f"  unique modes : {sorted(modes)}  (RGB = 3 channels)")


if __name__ == "__main__":
    count_images()
    check_image_props()
    sample_grid()
    print("\n[DONE] Phase 2 exploration complete.")
