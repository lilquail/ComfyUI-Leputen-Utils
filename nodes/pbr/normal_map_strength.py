"""
Normal Map Strength node - adjusts intensity of normal maps.
"""

import torch

from ...core import LEPUTEN_UTILS_CATEGORY, image_to_normal, normal_to_image


class NormalMapStrength:
    """Adjusts the perceived strength or 'bumpiness' of a normal map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input normal map whose strength needs adjustment."}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "The desired strength of the normal map.",
                    },
                ),
                "mode": (
                    ["Partial Derivatives", "Angles"],
                    {"tooltip": "The method used to adjust the normal map strength."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_strength"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/NormalMap"
    DESCRIPTION = "Adjusts the intensity or 'bumpiness' of a normal map using two different methods."

    def adjust_strength(self, image: torch.Tensor, strength: float, mode: str) -> tuple[torch.Tensor]:
        normals = image_to_normal(image)
        ddx = normals[:, :, :, 0]
        ddy = normals[:, :, :, 1]

        if mode == "Partial Derivatives":
            ddx_scaled = ddx * strength
            ddy_scaled = ddy * strength
            z_squared = 1.0 - torch.clamp(ddx_scaled**2 + ddy_scaled**2, 0.0, 1.0)
            z = torch.sqrt(z_squared)
            new_normals = torch.stack([ddx_scaled, ddy_scaled, z], dim=-1)

        elif mode == "Angles":
            angle = torch.atan2(ddy, ddx)
            magnitude = torch.sqrt(ddx**2 + ddy**2)
            new_angle = angle * strength
            ddx_scaled = torch.cos(new_angle) * magnitude
            ddy_scaled = torch.sin(new_angle) * magnitude
            z_squared = 1.0 - torch.clamp(ddx_scaled**2 + ddy_scaled**2, 0.0, 1.0)
            z = torch.sqrt(z_squared)
            new_normals = torch.stack([ddx_scaled, ddy_scaled, z], dim=-1)

        output_image = normal_to_image(new_normals)
        return (output_image,)
