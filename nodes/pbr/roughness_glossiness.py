"""
Roughness/Glossiness Converter node.

Converts between roughness and glossiness workflows using simple inversion.
"""
import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class RoughnessGlossinessConverter:
    """
    Converts between roughness and glossiness textures.

    Roughness and glossiness are inverse representations of surface smoothness:
    - Roughness: 1 = rough, 0 = smooth
    - Glossiness: 1 = smooth, 0 = rough

    Conversion is simply: output = 1 - input
    """
    DESCRIPTION = "Converts between roughness and glossiness textures. Simple inversion: roughness = 1 - glossiness."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input roughness or glossiness texture."}),
                "mode": (["Roughness to Glossiness", "Glossiness to Roughness"], {
                    "default": "Roughness to Glossiness",
                    "tooltip": "Conversion direction. Both use the same formula (inversion), but the label helps clarify intent."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"

    def convert(self, image: torch.Tensor, mode: str) -> tuple[torch.Tensor]:
        """
        Convert roughness to glossiness or vice versa.

        Args:
            image: Input texture tensor (B, H, W, C)
            mode: Conversion direction (cosmetic, same formula)

        Returns:
            Inverted texture tensor
        """
        # Simple inversion: output = 1 - input
        # Works for both directions since they're mathematical inverses
        result = 1.0 - image

        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)

        return (result,)
