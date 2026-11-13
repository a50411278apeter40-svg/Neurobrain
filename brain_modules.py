"""Brain-inspired modules (each can be switched on/off via config for ablation).

1. KWinnerMLP        lateral inhibition / sparse coding: only the top-k% most active units fire
2. DendriticMLP      two-compartment neuron: basal (feed-forward) drives, apical (top-down context)
                     multiplicatively gates -> "coincidence detection" of pyramidal neurons
3. FastWeightMemory  Hebbian synaptic plasticity: delta-rule outer-product memory updated online
                     (hippocampus-like fast learner next to the slow cortical weights)
4. Homeostasis       firing-rate homeostasis: running mean activity pulled toward a target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .math_ops_neurobrain import SynapticFire
except ImportError:
    from math_ops_neurobrain import SynapticFire


class KWinnerMLP(nn.Module):
    """Sparse coding: after SynapticFire keep only the top `k_frac` units per token (lateral inhibition)."""
    def __init__(self, d, ratio, k_frac=0.25):
        super().__init__()
        h = d * ratio
        self.up, self.down, self.act = nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False), SynapticFire(h)
        self.k = max(1, int(h * k_frac))

    def forward(self, x, context=None):
        a = self.act(self.up(x))
        thr = a.topk(self.k, dim=-1).values[..., -1:]
        a = a * (a >= thr).to(a.dtype)                       # winners keep activity, losers are silenced
        return self.down(a)


class DendriticMLP(nn.Module):
    """Pyramidal-neuron style: basal path = feed-forward drive, apical path = top-down modulation.
    out = down( SF(basal(x)) * (1 + tanh(apical(context))) ). Without context apical term is 0 -> plain MLP."""
    def __init__(self, d, ratio, k_frac=0.25, sparse=True):
        super().__init__()
        h = d * ratio
        self.basal, self.apical = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False)
        self.down, self.act = nn.Linear(h, d, bias=False), SynapticFire(h)
        self.k = max(1, int(h * k_frac)) if sparse else None
        nn.init.zeros_(self.apical.weight)

    def forward(self, x, context=None):
        a = self.act(self.basal(x))
        if context is not None:
            a = a * (1 + torch.tanh(self.apical(context)))
        if self.k:
            thr = a.topk(self.k, dim=-1).values[..., -1:]
            a = a * (a >= thr).to(a.dtype)
        return self.down(a)


class FastWeightMemory(nn.Module):
    """Hebbian fast weights (delta rule): W_t = W_{t-1} + beta_t (v_t - W_{t-1}k_t) k_t^T ; out_t = W_t q_t.
    Parallel causal form for training via chunk-free scan (fine for L<=~2k); O(L*d^2) state update."""
    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.beta = nn.Linear(d, heads)
        self.gate = nn.Parameter(torch.tensor(-2.0))           # starts almost closed -> safe to add to a trained net

    def forward(self, x, state=None):
        B, L, D = x.shape
        sh = lambda t: t.view(B, L, self.h, self.hd).transpose(1, 2)
        q, k, v = sh(self.q(x)), sh(self.k(x)), sh(self.v(x))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2).unsqueeze(-1)        # (B,H,L,1)
        W = state if state is not None else x.new_zeros(B, self.h, self.hd, self.hd)
        outs = []
        for t in range(L):
            kt, vt, qt, bt = k[:, :, t], v[:, :, t], q[:, :, t], beta[:, :, t]
            pred = torch.einsum("bhij,bhj->bhi", W, kt)
            W = W + bt.unsqueeze(-1) * torch.einsum("bhi,bhj->bhij", vt - pred, kt)   # Hebbian / delta update
            outs.append(torch.einsum("bhij,bhj->bhi", W, qt))
        y = torch.stack(outs, 2).transpose(1, 2).reshape(B, L, D)
        return torch.sigmoid(self.gate) * self.o(y), W


def homeostasis_loss(acts, target=0.1):
    """Firing-rate homeostasis: penalise deviation of mean |activity| from a target set-point."""
    return sum(((a.abs().mean() - target) ** 2) for a in acts) / max(len(acts), 1)
