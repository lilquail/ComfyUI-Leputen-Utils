"""
Core PBR utilities for ComfyUI-Leputen-Utils.

Contains helper functions for PBR map processing, specifically for
converting between image data (0..1) and vector data (-1..1).
"""

import torch


def image_to_normal(image: torch.Tensor) -> torch.Tensor:
    """
    Converts an image tensor with values in [0, 1] to a normal vector tensor with values in [-1, 1].

    Args:
        image: Input image tensor (B, H, W, C) range [0, 1]

    Returns:
        Normal tensor (B, H, W, C) range [-1, 1]
    """
    return (image * 2.0) - 1.0


def normal_to_image(normal: torch.Tensor) -> torch.Tensor:
    """
    Converts a normal vector tensor with values in [-1, 1] to an image tensor with values in [0, 1].

    Args:
        normal: Input normal tensor (B, H, W, C) range [-1, 1]

    Returns:
        Image tensor (B, H, W, C) range [0, 1]
    """
    return (normal + 1.0) / 2.0
