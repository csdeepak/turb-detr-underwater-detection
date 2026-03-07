"""SimAM — Simple, Parameter-Free Attention Module for CNNs.

Paper:
    Yang, L. et al. "SimAM: A Simple, Parameter-Free Attention Module for
    Convolutional Neural Networks." ICML 2021.

Key idea:
    SimAM derives per-neuron importance weights *without any learnable
    parameters* by measuring the linear separability of each neuron's
    activation from the rest of the spatial feature map.  The energy
    function is:

        e_t(w, b, y, x_i) =  1/(M-1) × Σ_{i=1}^{M-1} (-1 - (w·x_i + b))²
                             + (1 - (w·t + b))²

    where t is the target neuron, x_i are all other neurons in the same
    channel, and M = H × W.  Minimising this energy yields the closed-form
    importance:

        e_t* = 4 × (σ̂² + λ) / ((t − μ̂)² + 2σ̂² + 2λ)

    where μ̂ and σ̂² are the spatial mean and variance of the channel.
    Lower energy ⟹ the neuron is *more* distinct from neighbours ⟹ it
    should receive *higher* attention.  Therefore the final attention
    weight is  1 / e_t*  (inverse energy), applied element-wise and
    normalised by the sigmoid function.

Complexity:
    - 0 learnable parameters
    - O(B × C) spatial reductions  (mean & variance)
    - Negligible FLOPs compared to any conv layer

Usage:
    >>> import torch
    >>> from models.simam import SimAM
    >>> attn = SimAM(lambda_param=1e-4)
    >>> x = torch.randn(2, 64, 32, 32)
    >>> out = attn(x)          # same shape (2, 64, 32, 32)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Parameter-free SimAM attention module.

    Computes an energy-based, per-neuron attention weight from spatial
    statistics alone — no learnable parameters, no channel reduction,
    no extra convolutions.

    Parameters
    ----------
    lambda_param : float
        Small positive constant (λ) for numerical stability in the
        energy denominator.  The paper uses 1e-4.

    Shape
    -----
    - Input :  ``(B, C, H, W)``
    - Output:  ``(B, C, H, W)``   (same shape, attention-weighted)

    Examples
    --------
    >>> m = SimAM()
    >>> x = torch.randn(4, 256, 16, 16)
    >>> y = m(x)
    >>> y.shape
    torch.Size([4, 256, 16, 16])
    """

    def __init__(self, lambda_param: float = 1e-4) -> None:
        super().__init__()
        # λ is a fixed scalar — register as buffer so it moves with
        # .to(device) / .half() but is never in .parameters().
        self.register_buffer(
            "lambda_param",
            torch.tensor(lambda_param, dtype=torch.float32),
        )

    # ── forward ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SimAM attention and re-weight the input.

        Steps
        -----
        1. Spatial mean  μ̂  and variance  σ̂²  per channel.
        2. Per-neuron energy:
               e_t* = 4(σ̂² + λ) / ((t − μ̂)² + 2σ̂² + 2λ)
        3. Attention = sigmoid(1 / e_t*)  — high-energy (bland)
           neurons are suppressed, low-energy (salient) neurons
           are amplified.
        4. Output = x ⊙ attention

        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Attention-weighted feature map, same shape as input.
        """
        # ── spatial statistics (keep C dimension) ────────────
        # mean & var over H, W  →  shape (B, C, 1, 1)
        mu = x.mean(dim=(2, 3), keepdim=True)       # μ̂
        var = x.var(dim=(2, 3), keepdim=True)        # σ̂²  (unbiased by default)

        # ── per-neuron energy (closed-form) ──────────────────
        #   e_t* = 4(σ̂² + λ) / ((x - μ̂)² + 2σ̂² + 2λ)
        diff_sq = (x - mu).pow(2)                    # (t − μ̂)²
        numerator = 4.0 * (var + self.lambda_param)  # (B,C,1,1)
        denominator = diff_sq + 2.0 * var + 2.0 * self.lambda_param

        energy = numerator / denominator             # (B, C, H, W)

        # ── attention weights ────────────────────────────────
        # Lower energy → more salient → higher weight
        attention = torch.sigmoid(1.0 / energy)      # (B, C, H, W)

        return x * attention

    # ── repr ─────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return f"lambda_param={self.lambda_param.item():.1e}"


# ─────────────────────────────────────────────────────────────
# Convenience wrapper for sequential insertion
# ─────────────────────────────────────────────────────────────
def simam_attention(
    x: torch.Tensor,
    lambda_param: float = 1e-4,
) -> torch.Tensor:
    """Functional interface — apply SimAM attention in one call.

    Parameters
    ----------
    x : torch.Tensor
        Feature map ``(B, C, H, W)``.
    lambda_param : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Attention-weighted feature map.
    """
    mu = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True)
    diff_sq = (x - mu).pow(2)
    energy = 4.0 * (var + lambda_param) / (diff_sq + 2.0 * var + 2.0 * lambda_param)
    return x * torch.sigmoid(1.0 / energy)


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Smoke test with random input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = SimAM(lambda_param=1e-4).to(device)
    dummy = torch.randn(2, 128, 32, 32, device=device)
    out = module(dummy)

    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} vs {dummy.shape}"
    assert list(module.parameters()) == [], "SimAM should have 0 learnable parameters"
    print(f"SimAM self-test passed ✓")
    print(f"  Input : {dummy.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in module.parameters())} (should be 0)")
    print(f"  Buffers: {sum(b.numel() for b in module.buffers())}")
    print(f"  Device: {device}")
    print(f"  Module: {module}")
