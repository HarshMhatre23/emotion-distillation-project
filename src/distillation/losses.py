import torch, torch.nn as nn, torch.nn.functional as F

class SoftLabelKDLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super().__init__(); self.T = temperature
    def forward(self, s_logits, t_probs):
        s = F.log_softmax(s_logits / self.T, dim=-1)
        # Clamp target probs to avoid log(0)
        t = t_probs.clamp(min=1e-8, max=1.0)
        t = t / t.sum(dim=-1, keepdim=True)
        return F.kl_div(s, t, reduction="batchmean") * (self.T ** 2)

class HiddenStateKDLoss(nn.Module):
    def __init__(self, sd, td):
        super().__init__()
        self.proj = nn.Linear(sd, td)
        self.mse = nn.MSELoss()
    def forward(self, s, t):
        return self.mse(self.proj(s), t.detach())

class CompositeKDLoss(nn.Module):
    def __init__(self, sd, td, temperature=3.0,
                 soft_weight=1.0, hidden_weight=0.1, ce_weight=0.5):
        super().__init__()
        self.soft = SoftLabelKDLoss(temperature)
        self.hidden = HiddenStateKDLoss(sd, td)
        self.w_soft = soft_weight
        self.w_hidden = hidden_weight
        self.w_ce = ce_weight
    def forward(self, s_logits, t_probs, s_hidden, t_hidden, hard):
        # Guard against NaN inputs
        s_logits = torch.nan_to_num(s_logits, nan=0.0)
        loss_soft = self.soft(s_logits, t_probs)
        loss_hidden = self.hidden(s_hidden, t_hidden)
        loss_ce = F.binary_cross_entropy_with_logits(s_logits, hard)
        total = self.w_soft * loss_soft + self.w_hidden * loss_hidden + self.w_ce * loss_ce
        return {"total": total, "soft": loss_soft, "hidden": loss_hidden, "ce": loss_ce}
