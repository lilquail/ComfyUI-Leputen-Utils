"""
Normalize Normals node - recalculates the Z channel of normal maps.
"""

import torch

from ...core import LEPUTEN_UTILS_CATEGORY, image_to_normal, normal_to_image


class NormalizeNormals:
    """Normalizes a normal map by recalculating its blue (Z) channel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input normal map to be normalized."}),
                "output_z": (
                    ["Recalculate (Standard)", "Recalculate (0-1 Range)", "Set to 0", "Set to 0.5", "Set to 1"],
                    {
                        "default": "Recalculate (Standard)",
                        "tooltip": "How to set the Blue (Z) channel. 'Recalculate' derives Z from X/Y for valid normals. Fixed values useful for specific shaders.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/NormalMap"
    DESCRIPTION = """Fixes or rebuilds the Blue (Z) channel of a normal map.

• Recalculate (Standard): Derives Z from X/Y using z = sqrt(1 - x² - y²), then maps to 0.5-1.0 range. Use for standard tangent-space normal maps.
• Recalculate (0-1 Range): Same calculation but outputs raw Z (0-1). Use when shader expects unmapped values.
• Set to 0/0.5/1: Forces a fixed value. Useful for debugging or special shaders."""

    def normalize(self, image: torch.Tensor, output_z: str) -> tuple[torch.Tensor]:
        normals = image_to_normal(image)
        x = normals[:, :, :, 0]
        y = normals[:, :, :, 1]

        new_normals = torch.zeros_like(normals)
        new_normals[:, :, :, 0] = x
        new_normals[:, :, :, 1] = y

        z = torch.sqrt(1.0 - torch.clamp(x**2 + y**2, 0.0, 1.0))

        if output_z == "Recalculate (Standard)":
            new_normals[:, :, :, 2] = (z / 2.0) + 0.5
        elif output_z == "Recalculate (0-1 Range)":
            new_normals[:, :, :, 2] = z
        elif output_z == "Set to 0":
            new_normals[:, :, :, 2] = -1.0
        elif output_z == "Set to 0.5":
            new_normals[:, :, :, 2] = 0.0
        elif output_z == "Set to 1":
            new_normals[:, :, :, 2] = 1.0

        output_image = normal_to_image(new_normals)
        return (output_image,)
