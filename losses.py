"""
losses.py

All loss functions for the pedestrian intention prediction model.

Four losses combined:
    1. Focal Loss          — crossing head, handles class imbalance
    2. Weighted CE         — distraction head, masked for non-JAAD/PIE
    3. EDL Loss            — uncertainty head, with KL annealing
    4. Monotonicity Loss   — our novel extension, uncertainty grows with horizon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss 1: Focal Loss ────────────────────────────────────────────────────────

def focal_loss(pred, target, gamma=2.0, reduction='mean'):
    """
    Focal Loss for binary classification.
    
    Standard BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Focal Loss:   L = -[y*(1-p)^gamma*log(p) + (1-y)*p^gamma*log(1-p)]
    
    The (1-p)^gamma term down-weights easy confident predictions.
    When the model is very confident and correct, the loss is near zero.
    When the model is wrong or uncertain, the loss is high.
    
    gamma=2 is the standard choice from Lin et al. (2017).
    
    Args:
        pred:   [batch, 4] crossing probabilities (after Sigmoid)
        target: [batch, 4] binary crossing labels
        gamma:  focusing parameter (2.0 standard)
    """
    # Clamp predictions to avoid log(0)
    pred   = pred.clamp(1e-6, 1 - 1e-6)
    
    # Standard BCE terms
    bce    = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    
    # Focal weighting
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
    Distracted class gets 3x weight because it's rarer and
    more important to detect correctly.
    
    Masked: only computed for samples that have distraction labels
    (JAAD and PIE). TITAN samples are excluded.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            class_weights = torch.tensor([1.0, 3.0])
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets, mask):
        """
        logits:  [batch, 2] raw distraction predictions
        targets: [batch]    binary distraction labels (0=attentive, 1=distracted)
        mask:    [batch]    1.0 for samples with distraction labels, 0.0 otherwise
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Compute CE loss for all samples
        loss = F.cross_entropy(logits, targets,
                               weight=self.class_weights.to(logits.device),
                               reduction='none')

        # Apply mask — zero out samples without distraction labels
        loss = loss * mask

        # Average only over masked samples
        return loss.sum() / (mask.sum() + 1e-8)


# ── Loss 3: EDL Loss ──────────────────────────────────────────────────────────

def edl_loss(alpha, beta, targets, epoch, warmup_epochs=10):
    """
    Evidential Deep Learning loss for uncertainty quantification.
    
    Two components:
    
    1. Fit term: penalises wrong predictions
       Uses digamma function (derivative of log-gamma).
       Forces alpha to be high when target=1, beta when target=0.
    
    2. KL regularisation: penalises overconfident wrong predictions
       KL[Dirichlet(alpha_tilde, beta_tilde) || Dirichlet(1,1)]
       alpha_tilde removes the correct class evidence so we only
       penalise evidence that isn't supported by the label.
       
       Lambda is annealed from 0 to 1 over warmup_epochs.
       This lets the model learn to predict correctly first,
       before being penalised for claiming false certainty.
    
    Args:
        alpha:   [batch, 4] evidence for crossing
        beta:    [batch, 4] evidence against crossing
        targets: [batch, 4] binary crossing labels
        epoch:   current training epoch (for annealing)
    """
    # Annealing lambda: 0 at epoch 0, 1 at epoch warmup_epochs
    lam = min(1.0, epoch / max(warmup_epochs, 1))

    S = alpha + beta  # total evidence [batch, 4]

    # ── Fit term ──────────────────────────────────────────────────────────────
    # For samples where target=1: want alpha to be high
    # For samples where target=0: want beta to be high
    fit = targets * (torch.digamma(S) - torch.digamma(alpha)) + \
          (1 - targets) * (torch.digamma(S) - torch.digamma(beta))

    # ── KL regularisation term ────────────────────────────────────────────────
    # Remove correct class evidence before KL
    alpha_tilde = targets       + (1 - targets) * alpha
    beta_tilde  = (1 - targets) + targets       * beta

    S_tilde = alpha_tilde + beta_tilde

    # KL divergence between Dirichlet(alpha_tilde, beta_tilde) and Dirichlet(1,1)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde)
          + (alpha_tilde - 1) * torch.digamma(alpha_tilde)
          + (beta_tilde  - 1) * torch.digamma(beta_tilde)
          - (S_tilde     - 2) * torch.digamma(S_tilde))

    loss = (fit + lam * kl).mean()
    return loss


# ── Loss 4: Temporal Monotonicity Constraint ──────────────────────────────────

def monotonicity_loss(uncertainty):
    """
    Our novel extension to EDL — Contribution 2.
    
    Uncertainty should never decrease at longer horizons.
    Predicting 4 seconds ahead should always be at least as uncertain
    as predicting 0.5 seconds ahead.
    
    Physically motivated: longer horizon = less information = more uncertainty.
    
    Penalty: max(0, u_t - u_{t+1})^2 for each consecutive horizon pair.
    Only penalises violations — no penalty when uncertainty correctly increases.
    
    Args:
        uncertainty: [batch, 4] vacuity at each horizon
                     columns: [u@0.5s, u@1s, u@2s, u@4s]
    """
    loss = 0.0
    for t in range(uncertainty.shape[1] - 1):
        # Penalise when u_t > u_{t+1} (uncertainty decreasing — violation)
        violation = torch.clamp(uncertainty[:, t] - uncertainty[:, t+1], min=0)
        loss      = loss + (violation ** 2).mean()
    return loss


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combines all four losses with tuned weights.
    
    L_total = 1.0 * focal
            + 0.5 * distraction
            + 0.5 * edl
            + 0.1 * monotonicity
    
    Weights reflect task importance:
    - Crossing prediction is the primary task → 1.0
    - Distraction and uncertainty are auxiliary → 0.5
    - Monotonicity is a regulariser → 0.1
    """
    def __init__(self):
        super().__init__()
        self.distraction_loss_fn = WeightedDistrationLoss()

        self.w_focal       = 1.0
        self.w_distraction = 0.5
        self.w_edl         = 0.5
        self.w_mono        = 0.1

    def forward(self, outputs, batch, epoch):
        """
        outputs: dict from model.forward()
        batch:   dict from dataset.__getitem__()
        epoch:   current epoch for EDL annealing
        """
        targets     = batch['crossing_labels']       # [batch, 4]
        dist_labels = batch['distraction_label']     # [batch]
        has_dist    = batch['has_distraction']        # [batch] mask

        # Loss 1: Focal loss on crossing head
        l_focal = focal_loss(
            outputs['crossing_probs'], targets, gamma=2.0
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
        total = (self.w_focal       * l_focal +
                 self.w_distraction * l_dist  +
                 self.w_edl         * l_edl   +
                 self.w_mono        * l_mono)

        return {
            'total':       total,
            'focal':       l_focal.item(),
            'distraction': l_dist.item(),
            'edl':         l_edl.item(),
            'mono':        l_mono.item(),
        }


if __name__ == '__main__':
    # Test all losses with dummy data
    batch_size = 4

    # Dummy model outputs
    outputs = {
        'crossing_probs':     torch.rand(batch_size, 4),
        'distraction_logits': torch.rand(batch_size, 2),
        'alpha':              torch.rand(batch_size, 4) + 1,
        'beta':               torch.rand(batch_size, 4) + 1,
        'edl_probs':          torch.rand(batch_size, 4),
        'uncertainty':        torch.rand(batch_size, 4),
    }

    # Dummy batch
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