"""
Novel math components.

1) SynapticFire activation (SFA)
   f(x) = x * sigmoid(b*x) * (1 + a * tanh(g*x) * exp(-x^2 / (2 s^2)))
   - b, a, g, s are learnable per-channel.
   - Far from 0: behaves like SiLU (smooth, linear for x>>0, suppressed for x<<0).
   - Near 0: an oscillatory "threshold-crossing" correction modelled on the
     brief overshoot of a neuron's membrane potential around firing threshold.

2) Theta-Gamma Oscillatory Encoding (TGOE)
   PE(p, 2i)   = sin(w_i * p + m * sin(theta * p))
   PE(p, 2i+1) = cos(w_i * p + m * sin(theta * p))
   - w_i : log-spaced "gamma" carrier frequencies
   - theta: slow "theta" modulator, m: coupling depth (phase-amplitude coupling)
   Works for 1-D (time/audio), 2-D (image) and 3-D (video) positions by summing
   independent axis encodings over disjoint channel slices.
"""
import math
import torch
import torch.nn as nn


class SynapticFire(nn.Module):
    def __init__(self, channels=None, beta=1.0, alpha=0.25, gamma=1.5, sigma=1.0):
        super().__init__()
        shape = (channels,) if channels else (1,)
        self.beta = nn.Parameter(torch.full(shape, float(beta)))
        self.alpha = nn.Parameter(torch.full(shape, float(alpha)))
        self.gamma = nn.Parameter(torch.full(shape, float(gamma)))
        self.log_sigma = nn.Parameter(torch.full(shape, math.log(sigma)))

    def forward(self, x):
        s = self.log_sigma.exp()
        bump = torch.tanh(self.gamma * x) * torch.exp(-(x * x) / (2 * s * s + 1e-6))
        return x * torch.sigmoid(self.beta * x) * (1 + self.alpha * bump)


class ThetaGammaEncoding(nn.Module):
    def __init__(self, dim, theta=0.37, coupling=0.8, min_freq=1.0, max_freq=1e-4):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        w = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), half))
        self.register_buffer("w", w)
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.m = nn.Parameter(torch.tensor(float(coupling)))
        self.dim = dim

    def forward(self, positions):
        """positions: (...,) float tensor -> (..., dim)"""
        p = positions.float().unsqueeze(-1)
        phase = self.w * p + self.m * torch.sin(self.theta * p)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class MultiAxisTGOE(nn.Module):
    """TGOE over (t, y, x) with disjoint channel slices."""
    def __init__(self, dim):
        super().__init__()
        d = dim // 3 // 2 * 2
        self.dt, self.dy, self.dx = d, d, dim - 2 * d
        if self.dx % 2:
            self.dx -= 1
            self.pad = 1
        else:
            self.pad = 0
        self.et = ThetaGammaEncoding(self.dt, theta=0.21)
        self.ey = ThetaGammaEncoding(self.dy, theta=0.37)
        self.ex = ThetaGammaEncoding(self.dx, theta=0.53)
        self.dim = dim

    def forward(self, t, y, x):
        out = torch.cat([self.et(t), self.ey(y), self.ex(x)], dim=-1)
        if out.shape[-1] < self.dim:
            out = nn.functional.pad(out, (0, self.dim - out.shape[-1]))
        return out
