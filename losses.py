"""
losses.py

All loss functions for the pedestrian intention prediction model.

Four losses combined:
    1. BCEWithLogits      — crossing head, with pos_weight for class imbalance
    2. Weighted CE         — distraction head, masked for non-JAAD/PIE
    3. EDL Loss            — uncertainty head, with KL annealing
    4. Monotonicity Loss   — our novel extension, uncertainty grows with horizon

CHANGE LOG:
- Replaced focal_loss(pred_probs) with BCEWithLogitsLoss(raw_logits).
  Focal loss caused vanishing gradients when predictions collapsed to ~0.5.
  BCEWithLogitsLoss operates on raw logits, giving much stronger gradients
  and numerical stability via the log-sum-exp trick.
- pos_weight=2.0 handles class imbalance (similar effect to focal loss
  but without the gradient-killing (1-p)^gamma term).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss 1: BCE with logits + pos_weight ──────────────────────────────────────

def crossing_loss(logits, targets, pos_weight=2.0):
    """
    Binary cross-entropy on raw logits with positive class weighting.
    
    pos_weight > 1 increases the loss for false negatives (missed crossings).
    This has a similar effect to focal loss for handling class imbalance,
    but gives much stronger gradients because it operates on raw logits
    instead of post-Sigmoid probabilities.
    
    Args:
        logits:     [batch, 4] raw crossing logits (before Sigmoid)
        targets:    [batch, 4] binary crossing labels
        pos_weight: scalar weight for positive class (crossers)
    """
    pw = torch.tensor([pos_weight], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


# ── Loss 1 alt: Focal Loss (kept for later stages) ───────────────────────────

def focal_loss(pred, target, gamma=2.0, reduction='mean'):
    """
    Focal Loss for binary classification. Operates on probabilities.
    Kept for potential use in later training stages.
    """
    pred   = pred.clamp(1e-6, 1 - 1e-6)
    bce    = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    p_t    = pred * target + (1 - pred) * (1 - target)
    weight = (1 - p_t) ** gamma
    loss   = weight * bce
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()


# ── Loss 2: Weighted Cross Entropy for Distraction ────────────────────────────

class WeightedDistrationLoss(nn.Module):
    """
    Weighted Cross Entropy for distraction classification.
    Class weights [1.0, 3.0] for [attentive, distracted].
    Masked: only computed for samples with distraction labels.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            class_weights = torch.tensor([1.0, 3.0])
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        loss = F.cross_entropy(logits, targets,
                               weight=self.class_weights.to(logits.device),
                               reduction='none')
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)


# ── Loss 3: EDL Loss ──────────────────────────────────────────────────────────

def edl_loss(alpha, beta, targets, epoch, warmup_epochs=30):
    """
    Evidential Deep Learning loss for uncertainty quantification.
    """
    lam = min(1.0, epoch / max(warmup_epochs, 1))
    S = alpha + beta

    fit = targets * (torch.digamma(S) - torch.digamma(alpha)) + \
          (1 - targets) * (torch.digamma(S) - torch.digamma(beta))

    alpha_tilde = targets       + (1 - targets) * alpha
    beta_tilde  = (1 - targets) + targets       * beta
    S_tilde = alpha_tilde + beta_tilde

    kl = (torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde)
          + (alpha_tilde - 1) * torch.digamma(alpha_tilde)
          + (beta_tilde  - 1) * torch.digamma(beta_tilde)
          - (S_tilde     - 2) * torch.digamma(S_tilde))

    loss = (fit + lam * kl).mean()
    return loss


# ── Loss 4: Temporal Monotonicity Constraint ──────────────────────────────────

def monotonicity_loss(uncertainty):
    """
    Uncertainty should never decrease at longer horizons.
    """
    loss = 0.0
    for t in range(uncertainty.shape[1] - 1):
        violation = torch.clamp(uncertainty[:, t] - uncertainty[:, t+1], min=0)
        loss      = loss + (violation ** 2).mean()
    return loss


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Stage 1: Only crossing loss (BCEWithLogits).
    Other losses computed but weighted at 0.0.
    """
    def __init__(self):
        super().__init__()
        self.distraction_loss_fn = WeightedDistrationLoss()

        self.w_crossing    = 1.0
        self.w_distraction = 0.05
        self.w_edl         = 0.05
        self.w_mono        = 0.05

    def forward(self, outputs, batch, epoch):
        targets     = batch['crossing_labels']
        dist_labels = batch['distraction_label']
        has_dist    = batch['has_distraction']

        # Loss 1: BCEWithLogits on crossing head (uses raw logits)
        l_crossing = crossing_loss(
            outputs['crossing_logits'], targets, pos_weight=2.0
        )

        # Loss 2: Distraction CE (masked)
        l_dist = self.distraction_loss_fn(
            outputs['distraction_logits'], dist_labels, has_dist
        )

        # Loss 3: EDL loss
        l_edl = edl_loss(
            outputs['alpha'], outputs['beta'], targets, epoch
        )

        # Loss 4: Temporal monotonicity
        l_mono = monotonicity_loss(outputs['uncertainty'])

        # Combined
        total = (self.w_crossing    * l_crossing +
                 self.w_distraction * l_dist     +
                 self.w_edl         * l_edl      +
                 self.w_mono        * l_mono)

        return {
            'total':       total,
            'crossing':    l_crossing.item(),
            'distraction': l_dist.item(),
            'edl':         l_edl.item(),
            'mono':        l_mono.item(),
        }


if __name__ == '__main__':
    batch_size = 4

    outputs = {
        'crossing_logits':    torch.randn(batch_size, 4),    # raw logits now
        'crossing_probs':     torch.rand(batch_size, 4),
        'distraction_logits': torch.rand(batch_size, 2),
        'alpha':              torch.rand(batch_size, 4) + 1,
        'beta':               torch.rand(batch_size, 4) + 1,
        'edl_probs':          torch.rand(batch_size, 4),
        'uncertainty':        torch.rand(batch_size, 4),
    }

    batch = {
        'crossing_labels':   torch.randint(0, 2, (batch_size, 4)).float(),
        'distraction_label': torch.randint(0, 2, (batch_size,)),
        'has_distraction':   torch.ones(batch_size),
    }

    loss_fn = CombinedLoss()
    losses  = loss_fn(outputs, batch, epoch=5)

    print("Loss components:")
    for k, v in losses.items():
        print(f"  {k:15s}: {v:.4f}")