"""
Channel Operations node - per-channel manipulation and normalization.
"""
import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class ChannelOperations:
    """
    Performs per-channel operations on an image, similar to NormalizeNormals but for any image.
    Each channel can be independently normalized, inverted, or set to a fixed value.
    """

    CHANNEL_MODES = [
        "Passthrough",
        "Normalize (0-1)",
        "Invert",
        "Set to 0",
        "Set to 0.5",
        "Set to 1",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to process."}),
                "red_channel": (cls.CHANNEL_MODES, {
                    "default": "Passthrough",
                    "tooltip": "Operation to apply to the Red channel."
                }),
                "green_channel": (cls.CHANNEL_MODES, {
                    "default": "Passthrough",
                    "tooltip": "Operation to apply to the Green channel."
                }),
                "blue_channel": (cls.CHANNEL_MODES, {
                    "default": "Passthrough",
                    "tooltip": "Operation to apply to the Blue channel."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_channels"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = """Per-channel image manipulation.

• Passthrough: No change
• Normalize (0-1): Stretches channel to use full 0-1 range (auto-levels)
• Invert: Flips values (1 - value)
• Set to 0/0.5/1: Forces channel to a fixed value"""

    def _process_channel(self, channel: torch.Tensor, mode: str) -> torch.Tensor:
        """Process a single channel based on the selected mode."""
        if mode == "Passthrough":
            return channel
        elif mode == "Normalize (0-1)":
            # Per-image normalization
            min_val = channel.min()
            max_val = channel.max()
            range_val = max_val - min_val
            if range_val > 0:
                return (channel - min_val) / range_val
            else:
                return channel
        elif mode == "Invert":
            return 1.0 - channel
        elif mode == "Set to 0":
            return torch.zeros_like(channel)
        elif mode == "Set to 0.5":
            return torch.full_like(channel, 0.5)
        elif mode == "Set to 1":
            return torch.ones_like(channel)
        else:
            return channel

    def process_channels(self, image: torch.Tensor, red_channel: str, green_channel: str, blue_channel: str) -> tuple[torch.Tensor]:
        results = []

        for img in image:
            r = self._process_channel(img[..., 0], red_channel)
            g = self._process_channel(img[..., 1], green_channel)
            b = self._process_channel(img[..., 2], blue_channel)

            # Handle alpha if present
            if img.shape[-1] == 4:
                a = img[..., 3]
                processed = torch.stack([r, g, b, a], dim=-1)
            else:
                processed = torch.stack([r, g, b], dim=-1)

            results.append(processed)

        return (torch.stack(results),)
