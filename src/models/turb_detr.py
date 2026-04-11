"""
Turb-DETR: SimAM-Injected RT-DETR

Injection point: Between CNN intra-scale encoder (AIFI) output
and Transformer cross-scale encoder (CCFM) input in HybridEncoder.

Using Ultralytics RT-DETR implementation:
- File: ultralytics/nn/modules/block.py
- Class: HybridEncoder
- Method: forward()
- Insert: features = [self.simam(f) for f in features]
  after AIFI output, before CCFM input

This module provides utilities for the injection.
The actual modification is done to the Ultralytics source.
"""

from .simam import SimAM


def get_injection_instructions() -> str:
    """Print exact instructions for injecting SimAM into Ultralytics RT-DETR."""
    return """
    === SimAM Injection into Ultralytics RT-DETR ===

    1. Locate file: ultralytics/nn/modules/block.py
    2. Find class: HybridEncoder
    3. In __init__(), add:
         self.simam = SimAM(lambda_val=1e-4)

    4. In forward(), after the AIFI block processes features
       and before CCFM fusion, add:
         features = [self.simam(f) for f in features]

    5. Verify parameter count is unchanged:
         python -c "
         from ultralytics import RTDETR
         model = RTDETR('rtdetr-l.pt')
         params = sum(p.numel() for p in model.parameters())
         print(f'Parameters: {params:,}')
         "

    Expected: Parameter count should be IDENTICAL to vanilla RT-DETR.
    If it increased, your injection point is wrong.
    """


if __name__ == "__main__":
    print(get_injection_instructions())
