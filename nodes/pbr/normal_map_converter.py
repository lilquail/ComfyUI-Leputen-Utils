"""
Normal Map Converter node - flips X/Y channels for format conversion.
"""
import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class NormalMapConverter:
    """
    Converts normal maps between different formats with independent X/Y channel flipping.

    DirectX uses Y- (green pointing down), OpenGL uses Y+ (green pointing up).
    Flip Y to convert between them. Flip X for rare engine quirks.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The normal map image to convert."}),
                "flip_x": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the Red (X) channel. Rarely needed - only for engines with non-standard X conventions."
                }),
                "flip_y": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Invert the Green (Y) channel. Enable to convert between DirectX (Y-) and OpenGL (Y+) formats."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/NormalMap"
    DESCRIPTION = "Flips normal map channels. Enable flip_y to convert between DirectX (Y-) and OpenGL (Y+) formats."

    def convert(self, image: torch.Tensor, flip_x: bool, flip_y: bool) -> tuple[torch.Tensor]:
        converted_image = image.clone()

        if flip_x:
            converted_image[:, :, :, 0] = 1.0 - converted_image[:, :, :, 0]
        if flip_y:
            converted_image[:, :, :, 1] = 1.0 - converted_image[:, :, :, 1]

        return (converted_image,)
