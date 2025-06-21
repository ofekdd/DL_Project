# ── losses/asymmetric_loss.py ────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Ridnik et al., ICCV 2021 – Asymmetric Loss for Multi-Label Classification.
    gamma_pos: focus parameter for positive targets (typically 0-1)
    gamma_neg: focus parameter for negative targets (typically 4-5)
    clip     : optional floor for negative probabilities (∈ [0, 0.5])
    """
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gamma_pos, self.gamma_neg, self.clip = gamma_pos, gamma_neg, clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs

        # optional clip to prevent easy negatives from dominating
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # logits for BCE
        log_pos = torch.log(probs_pos.clamp(min=1e-8))
        log_neg = torch.log(probs_neg.clamp(min=1e-8))

        # asymmetric focusing
        loss_pos = (1 - probs_pos) ** self.gamma_pos * targets * log_pos
        loss_neg = probs_pos ** self.gamma_neg * (1 - targets) * log_neg
        loss = -(loss_pos + loss_neg).mean()

        return loss
