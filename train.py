"""
train.py

Training loop for pedestrian intention prediction model.

Key decisions:
- AdamW optimizer (correct weight decay for Transformers)
- Cosine annealing LR schedule (1e-4 → 1e-6 over 50 epochs)
- Gradient clipping max_norm=1.0
- Early stopping patience=10 on val F1@2s
- Checkpoint saving on best val F1
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import time

from graph_construction import EDGE_INDEX
from model   import PedestrianIntentModel
from losses  import CombinedLoss
from dataset import build_datasets

# ── Config ────────────────────────────────────────────────────────────────────
JAAD_ROOT      = '/Users/payas/JAAD'
PIE_ANNOT_ROOT = '/Users/payas/PIE_annotations/annotations/annotations'
NPY_ROOT       = '/Users/payas/pedestrian_project/npy_data'
CHECKPOINT_DIR = '/Users/payas/pedestrian_project/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE     = 16
NUM_EPOCHS     = 50
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 10
HORIZON_IDX    = 2   # F1@2s used for early stopping (index 2 = 2s horizon)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── F1 computation ────────────────────────────────────────────────────────────

def compute_f1(all_preds, all_targets, threshold=0.5):
    """
    Computes F1 at all 4 horizons.
    all_preds:   [N, 4] crossing probabilities
    all_targets: [N, 4] binary labels
    Returns list of 4 F1 scores.
    """
    f1_scores = []
    for h in range(4):
        preds   = (all_preds[:, h] > threshold).astype(int)
        targets = all_targets[:, h].astype(int)
        f1      = f1_score(targets, preds, zero_division=0)
        f1_scores.append(f1)
    return f1_scores


# ── Validation loop ───────────────────────────────────────────────────────────

def validate(model, val_loader, loss_fn, epoch, edge_index):
    model.eval()
    all_preds   = []
    all_targets = []
    all_flags   = []
    total_loss  = 0.0
    n_batches   = 0

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch['skeleton'].to(DEVICE)
            headpose = batch['headpose'].to(DEVICE)
            targets  = batch['crossing_labels'].to(DEVICE)

            # Move batch labels to device
            batch_device = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(skeleton, headpose, edge_index)
            losses  = loss_fn(outputs, batch_device, epoch)

            total_loss  += losses['total'].item()
            n_batches   += 1

            all_preds.append(outputs['crossing_probs'].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_flags.append(batch['dataset_flag'].cpu().numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_flags   = np.concatenate(all_flags,   axis=0)

    # F1 per dataset (0=JAAD, 1=PIE)
    jaad_mask = all_flags == 0
    pie_mask  = all_flags == 1

    f1_jaad = compute_f1(all_preds[jaad_mask], all_targets[jaad_mask]) if jaad_mask.sum() > 0 else [0.0]*4
    f1_pie  = compute_f1(all_preds[pie_mask],  all_targets[pie_mask])  if pie_mask.sum()  > 0 else [0.0]*4

    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, f1_jaad, f1_pie


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    # Build datasets
    print("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(
        jaad_root      = JAAD_ROOT,
        pie_annot_root = PIE_ANNOT_ROOT,
        npy_root       = NPY_ROOT,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=False)

    # Model, optimizer, scheduler, loss
    model      = PedestrianIntentModel().to(DEVICE)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_fn    = CombinedLoss()
    edge_index = EDGE_INDEX.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Early stopping state
    best_val_f1  = 0.0
    patience_ctr = 0
    best_epoch   = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        epoch_losses = {'total': 0, 'focal': 0, 'distraction': 0,
                        'edl': 0, 'mono': 0}
        n_batches    = 0
        t_start      = time.time()

        for batch in train_loader:
            skeleton = batch['skeleton'].to(DEVICE)
            headpose = batch['headpose'].to(DEVICE)
            batch_device = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(skeleton, headpose, edge_index)
            losses  = loss_fn(outputs, batch_device, epoch)

            losses['total'].backward()

            # Gradient clipping — prevents instability in attention layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k] if isinstance(losses[k], float) \
                                             else losses[k].item()
            n_batches += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        t_epoch = time.time() - t_start

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, f1_jaad, f1_pie = validate(model, val_loader, loss_fn, epoch, edge_index)

        # Early stopping on average F1@2s across both datasets
        val_f1_2s = (f1_jaad[HORIZON_IDX] + f1_pie[HORIZON_IDX]) / 2.0

        # Print epoch summary
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {epoch_losses['total']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {t_epoch:.1f}s")
        print(f"  JAAD → F1@0.5s: {f1_jaad[0]:.3f} | F1@1s: {f1_jaad[1]:.3f} | "
              f"F1@2s: {f1_jaad[2]:.3f} | F1@4s: {f1_jaad[3]:.3f}")
        print(f"  PIE  → F1@0.5s: {f1_pie[0]:.3f}  | F1@1s: {f1_pie[1]:.3f}  | "
              f"F1@2s: {f1_pie[2]:.3f}  | F1@4s: {f1_pie[3]:.3f}")

        # ── Checkpoint saving ─────────────────────────────────────────────────
        if val_f1_2s > best_val_f1:
            best_val_f1  = val_f1_2s
            best_epoch   = epoch + 1
            patience_ctr = 0

            checkpoint = {
                'epoch':      epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_f1_2s':  val_f1_2s,
                'val_f1_jaad': f1_jaad,
                'val_f1_pie':  f1_pie,
            }
            path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.save(checkpoint, path)
            print(f"  ★ New best model saved (F1@2s: {val_f1_2s:.3f})")

        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}. "
                      f"Best F1@2s: {best_val_f1:.3f} at epoch {best_epoch}")
                break

    print(f"\nTraining complete. Best F1@2s: {best_val_f1:.3f} at epoch {best_epoch}")
    return model


if __name__ == '__main__':
    train()