"""
evaluate.py

Complete evaluation script for the pedestrian intention prediction model.
Runs all stage checkpoints and produces:
- Ablation table (F1 per stage)
- Uncertainty analysis (ECE, monotonicity)
- Per-dataset breakdown
- Confusion matrices

Usage:
    cd ~/pedestrian_project && python evaluate.py
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from graph_construction import EDGE_INDEX
from model import PedestrianIntentModel
from dataset import build_datasets

# ── Config ────────────────────────────────────────────────────────────────────
JAAD_ROOT      = '/Users/payas/JAAD'
PIE_ANNOT_ROOT = '/Users/payas/PIE_annotations/annotations/annotations'
NPY_ROOT       = '/Users/payas/pedestrian_project/npy_data'
CHECKPOINT_DIR = '/Users/payas/pedestrian_project/checkpoints'
DEVICE         = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
THRESHOLD      = 0.50  # fixed threshold for fair comparison with baselines


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    model = PedestrianIntentModel().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    val_f1 = ckpt.get('val_f1_2s', 0)
    print(f"  Loaded checkpoint: epoch {epoch}, val F1@2s={val_f1:.3f}")
    return model


def run_inference(model, dataloader, edge_index):
    """Run model on all data, return predictions and labels."""
    all_preds, all_targets, all_flags = [], [], []
    all_uncertainty, all_alpha, all_beta = [], [], []
    all_distraction_pred, all_distraction_label = [], []

    with torch.no_grad():
        for batch in dataloader:
            sk = batch['skeleton'].to(DEVICE)
            hp = batch['headpose'].to(DEVICE)
            out = model(sk, hp, edge_index)

            all_preds.append(out['crossing_probs'].cpu().numpy())
            all_targets.append(batch['crossing_labels'].numpy())
            all_flags.append(batch['dataset_flag'].numpy())
            all_uncertainty.append(out['uncertainty'].cpu().numpy())
            all_alpha.append(out['alpha'].cpu().numpy())
            all_beta.append(out['beta'].cpu().numpy())
            all_distraction_pred.append(out['distraction_logits'].argmax(dim=1).cpu().numpy())
            all_distraction_label.append(batch['distraction_label'].numpy())

    return {
        'preds':       np.concatenate(all_preds),
        'targets':     np.concatenate(all_targets),
        'flags':       np.concatenate(all_flags),
        'uncertainty': np.concatenate(all_uncertainty),
        'alpha':       np.concatenate(all_alpha),
        'beta':        np.concatenate(all_beta),
        'dist_pred':   np.concatenate(all_distraction_pred),
        'dist_label':  np.concatenate(all_distraction_label),
    }


def compute_ece(preds, targets, n_bins=10):
    """Expected Calibration Error — measures if predicted probabilities match actual accuracy."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = targets[mask].mean()
        bin_conf = preds[mask].mean()
        ece += mask.sum() / len(preds) * abs(bin_acc - bin_conf)
    return ece


def evaluate_stage(model, dataloader, edge_index, stage_name):
    """Full evaluation for one stage checkpoint."""
    results = run_inference(model, dataloader, edge_index)
    preds   = results['preds']
    targets = results['targets']
    flags   = results['flags']
    unc     = results['uncertainty']

    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")

    # ── F1 per dataset at fixed threshold ─────────────────────────────────
    horizons = ['0.5s', '1s', '2s', '4s']
    
    for name, mask in [('JAAD', flags == 0), ('PIE', flags == 1)]:
        if mask.sum() == 0:
            continue
        print(f"\n  {name} ({mask.sum()} samples) — threshold={THRESHOLD}")
        for h, hz in enumerate(horizons):
            t = targets[mask, h].astype(int)
            p = (preds[mask, h] > THRESHOLD).astype(int)
            f1   = f1_score(t, p, zero_division=0)
            prec = precision_score(t, p, zero_division=0)
            rec  = recall_score(t, p, zero_division=0)
            acc  = accuracy_score(t, p)
            print(f"    {hz}: F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  Acc={acc:.3f}")

    # ── Uncertainty analysis ──────────────────────────────────────────────
    print(f"\n  Uncertainty (EDL)")
    print(f"    Mean per horizon: ", end="")
    for h, hz in enumerate(horizons):
        print(f"{hz}={unc[:, h].mean():.4f}  ", end="")
    print()

    # Monotonicity check
    violations = 0
    total_checks = len(unc) * 3
    for t in range(3):
        violations += (unc[:, t] > unc[:, t + 1]).sum()
    print(f"    Monotonicity violations: {violations}/{total_checks} ({100*violations/total_checks:.1f}%)")

    # ECE per horizon
    print(f"    ECE per horizon: ", end="")
    for h, hz in enumerate(horizons):
        ece = compute_ece(preds[:, h], targets[:, h])
        print(f"{hz}={ece:.4f}  ", end="")
    print()

    # ── Prediction distribution ───────────────────────────────────────────
    print(f"\n  Prediction Distribution @2s")
    for name, mask in [('JAAD', flags == 0), ('PIE', flags == 1)]:
        if mask.sum() == 0:
            continue
        pos = preds[mask, 2][targets[mask, 2] == 1]
        neg = preds[mask, 2][targets[mask, 2] == 0]
        if len(pos) > 0 and len(neg) > 0:
            print(f"    {name}: pos={pos.mean():.3f}±{pos.std():.3f}  "
                  f"neg={neg.mean():.3f}±{neg.std():.3f}  "
                  f"gap={pos.mean()-neg.mean():.3f}")

    # Return summary for ablation table
    jaad_mask = flags == 0
    pie_mask  = flags == 1
    jaad_f1 = f1_score(targets[jaad_mask, 2].astype(int),
                       (preds[jaad_mask, 2] > THRESHOLD).astype(int),
                       zero_division=0) if jaad_mask.sum() > 0 else 0
    pie_f1  = f1_score(targets[pie_mask, 2].astype(int),
                       (preds[pie_mask, 2] > THRESHOLD).astype(int),
                       zero_division=0) if pie_mask.sum() > 0 else 0
    return jaad_f1, pie_f1


def main():
    print(f"Using device: {DEVICE}")
    print("Loading test dataset...")

    _, _, test_ds = build_datasets(
        jaad_root=JAAD_ROOT,
        pie_annot_root=PIE_ANNOT_ROOT,
        npy_root=NPY_ROOT,
    )

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    edge_index  = EDGE_INDEX.to(DEVICE)

    # ── Run all stage checkpoints ─────────────────────────────────────────
    stages = [
        ('Stage 1: Crossing Loss Only',         'stage1_best.pt'),
        ('Stage 2: + Distraction (w=0.05)',      'stage2_best.pt'),
        ('Stage 3: + EDL (w=0.05)',              'stage3_best.pt'),
        ('Stage 4: + Monotonicity (w=0.05)',     'stage4_best.pt'),
    ]

    ablation = []
    for stage_name, ckpt_file in stages:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"\n  SKIPPED {stage_name} — {ckpt_file} not found")
            continue
        print(f"\nLoading {ckpt_file}...")
        model = load_model(ckpt_path)
        jaad_f1, pie_f1 = evaluate_stage(model, test_loader, edge_index, stage_name)
        ablation.append((stage_name, jaad_f1, pie_f1))

    # ── Print ablation summary table ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY (Test Set, threshold={THRESHOLD})")
    print(f"{'='*60}")
    print(f"  {'Stage':<40} {'JAAD F1@2s':>10} {'PIE F1@2s':>10} {'Avg':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8}")
    for name, jf, pf in ablation:
        avg = (jf + pf) / 2
        print(f"  {name:<40} {jf:>10.3f} {pf:>10.3f} {avg:>8.3f}")

    # ── Comparison with baselines ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON WITH BASELINES")
    print(f"{'='*60}")
    print(f"  {'Method':<40} {'JAAD F1@2s':>10} {'PIE F1@2s':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    print(f"  {'PCPA (Kotseruba 2021)':<40} {'~0.73':>10} {'~0.73':>10}")
    print(f"  {'PedGraph (Cadena 2022)':<40} {'~0.70':>10} {'—':>10}")
    if ablation:
        best = ablation[-1]  # Stage 4 = full model
        print(f"  {'Ours (Full Model)':<40} {best[1]:>10.3f} {best[2]:>10.3f}")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()