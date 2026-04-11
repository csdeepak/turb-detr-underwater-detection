"""Tests for SimAM module."""

import torch
import pytest
from src.models.simam import SimAM


def test_simam_output_shape():
    """SimAM output must match input shape exactly."""
    module = SimAM()
    x = torch.randn(2, 256, 20, 20)
    out = module(x)
    assert out.shape == x.shape


def test_simam_zero_parameters():
    """SimAM must add exactly zero trainable parameters."""
    module = SimAM()
    n_params = sum(p.numel() for p in module.parameters())
    assert n_params == 0, f"Expected 0 params, got {n_params}"


def test_simam_gradient_flow():
    """Gradients must flow through SimAM without NaN."""
    module = SimAM()
    x = torch.randn(1, 64, 10, 10, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_simam_lambda_values():
    """Test different lambda values don't break forward pass."""
    for lam in [1e-3, 1e-4, 1e-5]:
        module = SimAM(lambda_val=lam)
        x = torch.randn(1, 128, 16, 16)
        out = module(x)
        assert not torch.isnan(out).any()


def test_simam_output_bounded():
    """Output values should be finite and reasonably bounded."""
    module = SimAM()
    x = torch.randn(1, 256, 20, 20) * 10
    out = module(x)
    assert torch.isfinite(out).all()
