"""
PixelTruth — training engine.

Chalane ke tarike:
    python -m src.train --model custom            # baseline CNN
    python -m src.train --model efficientnet      # transfer learning (2 phase)
    python -m src.train --model efficientnet --epochs-head 3 --epochs-ft 8

Features:
  - Mixed precision (AMP) GPU pe — fast training
  - EarlyStopping (val_loss improve na ho to ruk jao)
  - ReduceLROnPlateau (atak gaye to lr aadha)
  - Best model checkpoint (sabse achha val_loss wala save)
"""
import argparse
import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config as cfg
from . import data as data_mod
from . import models as models_mod
from . import utils


# ----------------------------------------------------------------------
# Ek epoch train karo
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch_desc,
                    max_batches=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(tqdm(loader, desc=epoch_desc, leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


# ----------------------------------------------------------------------
# Validation / evaluation (no grad)
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, max_batches=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(cfg.DEVICE, non_blocking=True)

        with torch.autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


# ----------------------------------------------------------------------
# fit() — poora training loop with callbacks
# ----------------------------------------------------------------------
def fit(model, train_loader, valid_loader, optimizer, epochs,
        save_path, phase_name="train", history=None, max_batches=None):
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=cfg.USE_AMP)

    # ReduceLROnPlateau — val_loss atak jaye to lr aadha
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_FACTOR, patience=cfg.LR_PATIENCE,
    )

    if history is None:
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        desc = f"[{phase_name}] epoch {epoch}/{epochs}"
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, desc, max_batches)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, max_batches)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{desc}  |  train_loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}  |  lr {lr_now:.1e}")

        # --- Checkpoint best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"    [BEST] saved to {save_path.name}")
        else:
            epochs_no_improve += 1

        # --- EarlyStopping ---
        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"    [STOP] early stopping (no improve for "
                  f"{cfg.EARLY_STOP_PATIENCE} epochs)")
            break

    # best weights restore karo (restore_best_weights=True jaisa)
    model.load_state_dict(best_weights)
    return history


# ----------------------------------------------------------------------
# Model-specific training routines
# ----------------------------------------------------------------------
def train_custom(train_loader, valid_loader, epochs, max_batches=None):
    model = models_mod.build_custom_cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR_HEAD)
    history = fit(model, train_loader, valid_loader, optimizer,
                  epochs=epochs, save_path=cfg.CUSTOM_CNN_PATH,
                  phase_name="custom-cnn", max_batches=max_batches)
    utils.plot_history(history, "Custom CNN",
                       cfg.OUTPUT_DIR / "training_curves_custom.png")
    return model


def train_efficientnet(train_loader, valid_loader, epochs_head, epochs_ft,
                       max_batches=None):
    # --- Phase 1: feature extraction (base frozen, head train) ---
    print("\n========== PHASE 1: Feature Extraction (base frozen) ==========")
    model = models_mod.build_efficientnet(freeze_base=True)
    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR_HEAD)
    history = fit(model, train_loader, valid_loader, opt1,
                  epochs=epochs_head, save_path=cfg.EFFICIENTNET_PATH,
                  phase_name="effnet-head", max_batches=max_batches)

    # --- Phase 2: fine-tuning (top unfrozen, low lr) ---
    print("\n========== PHASE 2: Fine-Tuning (top layers unfrozen) ==========")
    model = models_mod.unfreeze_top(model)
    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR_FINETUNE)
    history = fit(model, train_loader, valid_loader, opt2,
                  epochs=epochs_ft, save_path=cfg.EFFICIENTNET_PATH,
                  phase_name="effnet-finetune", history=history,
                  max_batches=max_batches)

    utils.plot_history(history, "EfficientNetB0 (transfer learning)",
                       cfg.OUTPUT_DIR / "training_curves_efficientnet.png")
    return model


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PixelTruth models")
    parser.add_argument("--model", choices=["custom", "efficientnet"],
                        required=True, help="kaunsa model train karna hai")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS_HEAD,
                        help="custom CNN ke liye epochs")
    parser.add_argument("--epochs-head", type=int, default=cfg.EPOCHS_HEAD,
                        help="efficientnet phase 1 epochs")
    parser.add_argument("--epochs-ft", type=int, default=cfg.EPOCHS_FINETUNE,
                        help="efficientnet phase 2 (fine-tune) epochs")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="per epoch sirf itne batches (smoke test ke liye)")
    args = parser.parse_args()

    utils.set_seed()
    utils.device_info()

    print("\nLoading data...")
    train_loader, valid_loader, _ = data_mod.get_dataloaders()
    print(f"Train: {len(train_loader.dataset)}  Valid: {len(valid_loader.dataset)}")

    if args.model == "custom":
        train_custom(train_loader, valid_loader, epochs=args.epochs,
                     max_batches=args.max_batches)
    else:
        train_efficientnet(train_loader, valid_loader,
                           epochs_head=args.epochs_head, epochs_ft=args.epochs_ft,
                           max_batches=args.max_batches)

    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
