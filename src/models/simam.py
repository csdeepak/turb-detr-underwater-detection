"""
SimAM: Simple Parameter-Free Attention Module
Reference: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
for Convolutional Neural Networks", ICML 2021.

Injected between CNN intra-scale encoder and Transformer cross-scale encoder
in RT-DETR to suppress turbidity noise before global self-attention.

Why this works for turbidity:
- Turbidity creates high-variance, spatially incoherent noise in feature maps
- SimAM's energy function assigns LOW energy to neurons that deviate meaningfully
  from spatial mean (high-information neurons)
- Backscatter noise deviates randomly → gets suppressed by sigmoid gating
- Zero additional parameters → zero computational overhead
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    Parameter-free 3D attention using energy-based neuron importance estimation.

    Args:
        lambda_val (float): Regularization constant to prevent division by zero.
                           Default 1e-4 works for most cases.
                           Ablation recommended: test 1e-3, 1e-4, 1e-5.
    """

    def __init__(self, lambda_val: float = 1e-4):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SimAM attention weighting.

        Args:
            x: Input feature map of shape (B, C, H, W)

        Returns:
            Attention-weighted feature map of same shape (B, C, H, W)
        """
        b, c, h, w = x.size()
        n = h * w - 1  # Unbiased variance denominator

        # Channel-wise spatial mean
        mean = x.mean(dim=[2, 3], keepdim=True)

        # Spatial variance (unbiased estimate)
        var = ((x - mean) ** 2).sum(dim=[2, 3], keepdim=True) / n

        # Energy function: high-information neurons get HIGH energy
        # The +0.5 bias prevents complete neuron suppression
        energy = (x - mean) ** 2 / (4 * (var + self.lambda_val)) + 0.5

        # Sigmoid gating: convert energy to [0, 1] attention weights
        attention = torch.sigmoid(energy)

        return x * attention

    def extra_repr(self) -> str:
        return f"lambda_val={self.lambda_val}"


def verify_zero_params(model_with_simam: nn.Module, model_without_simam: nn.Module):
    """
    Verify that SimAM injection adds zero trainable parameters.

    Args:
        model_with_simam: RT-DETR model with SimAM injected
        model_without_simam: Vanilla RT-DETR model

    Raises:
        AssertionError if parameter count differs
    """
    params_with = sum(p.numel() for p in model_with_simam.parameters())
    params_without = sum(p.numel() for p in model_without_simam.parameters())

    assert params_with == params_without, (
        f"Parameter count mismatch! "
        f"With SimAM: {params_with:,}, Without: {params_without:,}. "
        f"SimAM should add ZERO parameters. Check injection point."
    )
    print(f"✓ Parameter count verified: {params_with:,} (unchanged)")
