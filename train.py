"""
train.py

Training loop for pedestrian intention prediction model.

Key decisions:
- AdamW optimizer (correct weight decay for Transformers)
- Linear warmup (3 epochs) then cosine annealing (5e-4 → 1e-6)
- Gradient clipping max_norm=1.0
- Early stopping patience=10 on val F1@2s
- Checkpoint saving on best val F1
- Optimal threshold search per horizon (floor 0.30)
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
NUM_EPOCHS     = 20        # shorter for stage 2
LR             = 1e-4      # lower LR — fine-tuning, not training from scratch
WARMUP_EPOCHS  = 1         # shorter warmup
WEIGHT_DECAY   = 1e-4
PATIENCE       = 10
HORIZON_IDX    = 2

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ── F1 computation ────────────────────────────────────────────────────────────

def compute_f1(all_preds, all_targets, threshold=None):
    """
    Computes F1 at all 4 horizons.
    Threshold search range: [0.30, 0.70] with step 0.05.
    """
    f1_scores  = []
    thresholds = []
    
    for h in range(4):
        targets = all_targets[:, h].astype(int)
        
        if threshold is not None:
            preds = (all_preds[:, h] > threshold).astype(int)
            f1 = f1_score(targets, preds, zero_division=0)
            f1_scores.append(f1)
            thresholds.append(threshold)
        else:
            best_f1 = 0.0
            best_t  = 0.5
            for t in np.arange(0.30, 0.71, 0.05):
                preds = (all_preds[:, h] > t).astype(int)
                f = f1_score(targets, preds, zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_t  = t
            f1_scores.append(best_f1)
            thresholds.append(best_t)
    
    return f1_scores, thresholds


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

            batch_device = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(skeleton, headpose, edge_index)
            losses  = loss_fn(outputs, batch_device, epoch)

            total_loss  += losses['total'].item()
            n_batches   += 1

            # Use crossing_probs (post-sigmoid) for evaluation
            all_preds.append(outputs['crossing_probs'].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_flags.append(batch['dataset_flag'].cpu().numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_flags   = np.concatenate(all_flags,   axis=0)

    jaad_mask = all_flags == 0
    pie_mask  = all_flags == 1

    f1_jaad, t_jaad = compute_f1(all_preds[jaad_mask], all_targets[jaad_mask]) if jaad_mask.sum() > 0 else ([0.0]*4, [0.5]*4)
    f1_pie,  t_pie  = compute_f1(all_preds[pie_mask],  all_targets[pie_mask])  if pie_mask.sum()  > 0 else ([0.0]*4, [0.5]*4)

    avg_loss = total_loss / max(n_batches, 1)

    # Log prediction distribution every 5 epochs for debugging
    if (epoch + 1) % 5 == 0 or epoch == 0:
        for name, mask in [('JAAD', jaad_mask), ('PIE', pie_mask)]:
            preds_h2 = all_preds[mask, 2]
            tgts_h2  = all_targets[mask, 2]
            pos_preds = preds_h2[tgts_h2 == 1]
            neg_preds = preds_h2[tgts_h2 == 0]
            if len(pos_preds) > 0 and len(neg_preds) > 0:
                print(f"  [{name} pred dist @2s] "
                      f"pos: mean={pos_preds.mean():.3f} std={pos_preds.std():.3f} | "
                      f"neg: mean={neg_preds.mean():.3f} std={neg_preds.std():.3f}")

    return avg_loss, f1_jaad, f1_pie, t_jaad, t_pie


# ── Learning rate schedule with warmup ────────────────────────────────────────

def get_lr(epoch, warmup_epochs, max_lr, min_lr, total_epochs):
    """Linear warmup for warmup_epochs, then cosine decay."""
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print(f"Using device: {DEVICE}")
    
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

    model      = PedestrianIntentModel().to(DEVICE)
    # Load Stage 1 checkpoint
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'stage2_best.pt'),
                       map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded Stage 1 checkpoint (epoch {ckpt['epoch']}, F1@2s: {ckpt['val_f1_2s']:.3f})")
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    loss_fn    = CombinedLoss()
    edge_index = EDGE_INDEX.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    best_val_f1  = 0.0
    patience_ctr = 0
    best_epoch   = 0

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"LR: {LR} with {WARMUP_EPOCHS}-epoch warmup then cosine decay")
    print(f"Loss: BCEWithLogitsLoss (pos_weight=2.0) — stronger gradients than focal")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        current_lr = get_lr(epoch, WARMUP_EPOCHS, LR, 1e-6, NUM_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        epoch_losses = {'total': 0, 'crossing': 0, 'distraction': 0,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k] if isinstance(losses[k], float) \
                                             else losses[k].item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        t_epoch = time.time() - t_start

        val_loss, f1_jaad, f1_pie, t_jaad, t_pie = validate(
            model, val_loader, loss_fn, epoch, edge_index
        )

        val_f1_2s = (f1_jaad[HORIZON_IDX] + f1_pie[HORIZON_IDX]) / 2.0

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {epoch_losses['total']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {t_epoch:.1f}s")
        print(f"  JAAD → F1@0.5s: {f1_jaad[0]:.3f} | F1@1s: {f1_jaad[1]:.3f} | "
              f"F1@2s: {f1_jaad[2]:.3f} | F1@4s: {f1_jaad[3]:.3f}  "
              f"(t: {t_jaad[2]:.2f})")
        print(f"  PIE  → F1@0.5s: {f1_pie[0]:.3f}  | F1@1s: {f1_pie[1]:.3f}  | "
              f"F1@2s: {f1_pie[2]:.3f}  | F1@4s: {f1_pie[3]:.3f}   "
              f"(t: {t_pie[2]:.2f})")

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
                'thresholds_jaad': t_jaad,
                'thresholds_pie':  t_pie,
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