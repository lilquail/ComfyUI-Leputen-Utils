"""
Z-Stack node - statistical stacking (median/mean) for noise reduction.

Uses dynamic expanding inputs - a new input slot appears when the current one is connected.
"""

import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class AnyType(str):
    """A special type that matches any other type for wildcard inputs."""

    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


# Wildcard type for flexible image inputs
any_type = AnyType("*")


class ZStack:
    """
    Statistical stacking (median/mean blending) of multiple images.

    Takes multiple images via dynamic expanding inputs and computes the per-pixel
    median or mean across all connected images. This technique is commonly used
    in photography and image processing to reduce noise, remove transient objects,
    and improve overall image quality.
    """

    MODES = ["Median", "Mean"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    cls.MODES,
                    {
                        "default": "Median",
                        "tooltip": "Median: More robust to outliers, removes transient objects. Mean: Simple averaging, faster computation.",
                    },
                ),
            },
            "optional": {
                "image_1": (
                    "IMAGE",
                    {"tooltip": "First image to stack. Connect more images below."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stacked_image",)
    FUNCTION = "stack"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = """Statistical stacking for noise reduction with dynamic inputs.

• Median: Robust to outliers, removes transient objects. Best for noise reduction.
• Mean: Simple averaging, preserves all data. Faster but less robust.

Connect multiple images - new inputs appear automatically when you connect."""

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Accept any dynamically created inputs."""
        return True

    def stack(self, mode: str, **kwargs) -> tuple[torch.Tensor]:
        """
        Stack multiple images using the specified statistical method.

        Args:
            mode: Stacking method - "Median" or "Mean".
            **kwargs: Dynamically created image inputs (image_1, image_2, etc.)

        Returns:
            Single stacked image tensor of shape [1, H, W, C].
        """
        # Collect all connected image inputs
        images = []
        for key, value in kwargs.items():
            if key.startswith("image_") and value is not None:
                if isinstance(value, torch.Tensor):
                    images.append(value)

        if not images:
            # No images connected, return empty
            raise ValueError("No images connected. Please connect at least one image.")

        if len(images) == 1:
            # Single image, nothing to stack
            return (images[0],)

        # Get shapes for validation
        first_shape = images[0].shape
        for i, img in enumerate(images[1:], start=2):
            if img.shape != first_shape:
                raise ValueError(
                    f"Image {i} has shape {img.shape}, but expected {first_shape}. "
                    "All images must have the same dimensions."
                )

        # Stack all images along a new dimension (dim=0)
        # Each image may already be a batch, so we need to handle that
        # For simplicity, we'll take the first frame of each batch
        frames = [img[0] for img in images]  # Take first frame from each batch
        stacked_batch = torch.stack(frames, dim=0)

        # Compute the statistic across the batch dimension (dim=0)
        stacked = torch.median(stacked_batch, dim=0).values if mode == "Median" else torch.mean(stacked_batch, dim=0)

        # Add batch dimension back
        return (stacked.unsqueeze(0),)
