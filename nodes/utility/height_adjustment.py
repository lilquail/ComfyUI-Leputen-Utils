"""
Height Adjustment node - adjusts levels and offset of height maps.
"""
import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class HeightAdjustment:
    """
    Adjusts the levels and offset of a height map to precisely control its
    range, intensity, and overall elevation.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input height map image (grayscale tensor) to adjust."}),
                "levels_in_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Input Black", "tooltip": "Sets the new black point for the height map."}),
                "levels_in_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Input White", "tooltip": "Sets the new white point for the height map."}),
                "offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Shifts the entire range of the height map up or down."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_height"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = "Adjusts the levels and offset of a height map for precise control over elevation and intensity."

    def adjust_height(self, image: torch.Tensor, levels_in_min: float, levels_in_max: float, offset: float) -> tuple[torch.Tensor]:
        if levels_in_min >= levels_in_max:
            levels_in_max = levels_in_min + 1e-6

        adjusted_image = (image - levels_in_min) / (levels_in_max - levels_in_min)
        adjusted_image = adjusted_image + offset
        adjusted_image = torch.clamp(adjusted_image, 0.0, 1.0)

        return (adjusted_image,)
