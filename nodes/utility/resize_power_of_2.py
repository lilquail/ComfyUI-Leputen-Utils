"""
Power of 2 Resize node - resizes images to power of 2 dimensions.
"""

import torch
from PIL import Image

from ...core import LEPUTEN_UTILS_CATEGORY, calc_power_of_2, log_verbose, pil2tensor, tensor2pil


class ResizePowerOf2:
    """
    Resizes or crops image dimensions to the nearest power of 2 for game engine compatibility.

    Many game engines and older graphics APIs require textures with power of 2
    dimensions (e.g., 256, 512, 1024, 2048). This node provides flexible options
    for resizing or cropping images to meet these requirements.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image tensor to resize or crop."}),
                "mode": (
                    ["Nearest", "Up", "Down"],
                    {
                        "default": "Nearest",
                        "tooltip": "Nearest: closest power of 2. Up: next larger. Down: previous smaller.",
                    },
                ),
            },
            "optional": {
                "method": (
                    ["Resize", "Crop"],
                    {
                        "default": "Resize",
                        "tooltip": "Resize: scale image to new dimensions. Crop: cut image to new dimensions.",
                    },
                ),
                "crop_position": (
                    [
                        "Top-Left",
                        "Top-Center",
                        "Top-Right",
                        "Center-Left",
                        "Center",
                        "Center-Right",
                        "Bottom-Left",
                        "Bottom-Center",
                        "Bottom-Right",
                    ],
                    {"default": "Center", "tooltip": "Crop alignment when using Crop method."},
                ),
                "resample": (
                    ["Lanczos", "Bilinear", "Nearest"],
                    {
                        "default": "Lanczos",
                        "tooltip": "Resampling filter for resize. Lanczos: highest quality. Bilinear: balanced. Nearest: pixelated.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Utility"
    DESCRIPTION = "Resizes or crops image dimensions to power of 2 for game engine compatibility."

    def process(
        self,
        image: torch.Tensor,
        mode: str,
        method: str = "Resize",
        crop_position: str = "Center",
        resample: str = "Lanczos",
    ) -> tuple[torch.Tensor]:
        """Resize or crop image dimensions to power of 2."""
        # Map resample names to PIL constants
        resample_map = {
            "Lanczos": Image.LANCZOS,
            "Bilinear": Image.BILINEAR,
            "Nearest": Image.NEAREST,
        }
        resample_filter = resample_map.get(resample, Image.LANCZOS)

        # Process batch
        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            pil_img = tensor2pil(image[i])
            orig_w, orig_h = pil_img.size

            new_w = calc_power_of_2(orig_w, mode)
            new_h = calc_power_of_2(orig_h, mode)

            if (new_w, new_h) != (orig_w, orig_h):
                if method == "Crop":
                    # Calculate crop box based on position
                    left, top = self._calc_crop_offset(orig_w, orig_h, new_w, new_h, crop_position)
                    pil_img = pil_img.crop((left, top, left + new_w, top + new_h))
                    log_verbose("ResizePowerOf2", f"Cropped {orig_w}x{orig_h} -> {new_w}x{new_h} ({crop_position})")
                else:
                    pil_img = pil_img.resize((new_w, new_h), resample_filter)
                    log_verbose("ResizePowerOf2", f"Resized {orig_w}x{orig_h} -> {new_w}x{new_h}")

            results.append(pil2tensor(pil_img))

        return (torch.cat(results, dim=0),)

    @staticmethod
    def _calc_crop_offset(orig_w: int, orig_h: int, new_w: int, new_h: int, position: str) -> tuple[int, int]:
        """Calculate crop offset based on position anchor."""
        # Ensure we don't crop beyond image bounds
        max_left = max(0, orig_w - new_w)
        max_top = max(0, orig_h - new_h)

        if position == "Top-Left":
            return 0, 0
        elif position == "Top-Center":
            return max_left // 2, 0
        elif position == "Top-Right":
            return max_left, 0
        elif position == "Center-Left":
            return 0, max_top // 2
        elif position == "Center":
            return max_left // 2, max_top // 2
        elif position == "Center-Right":
            return max_left, max_top // 2
        elif position == "Bottom-Left":
            return 0, max_top
        elif position == "Bottom-Center":
            return max_left // 2, max_top
        elif position == "Bottom-Right":
            return max_left, max_top
        else:
            return max_left // 2, max_top // 2  # Default to center
