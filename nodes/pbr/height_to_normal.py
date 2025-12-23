"""
Height To Normal node - generates normal maps from height maps using Sobel operators.
"""

import torch

from ...core import LEPUTEN_UTILS_CATEGORY, normal_to_image


class HeightToNormal:
    """Generates a normal map from a height/displacement map using Sobel operators."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_map": ("IMAGE", {"tooltip": "The input grayscale height map (white is high, black is low)."}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Controls the intensity of the normal map.",
                    },
                ),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Invert the height map before processing."}),
            },
            "optional": {
                "output_format": (
                    ["OpenGL (Y+)", "DirectX (Y-)"],
                    {"default": "OpenGL (Y+)", "tooltip": "The coordinate system for the output normal map."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_normal"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/NormalMap"
    DESCRIPTION = (
        "Generates a normal map from a height map using Sobel operators. A fast, non-AI alternative to DeepBump."
    )

    def generate_normal(
        self, height_map: torch.Tensor, strength: float, invert: bool, output_format: str = "OpenGL (Y+)"
    ) -> tuple[torch.Tensor]:
        if height_map.shape[-1] >= 3:
            grayscale = height_map[..., 0] * 0.299 + height_map[..., 1] * 0.587 + height_map[..., 2] * 0.114
        else:
            grayscale = height_map[..., 0]

        if invert:
            grayscale = 1.0 - grayscale

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=height_map.dtype, device=height_map.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=height_map.dtype, device=height_map.device
        ).view(1, 1, 3, 3)

        results = []
        for img in grayscale:
            img_4d = img.unsqueeze(0).unsqueeze(0)
            padded = torch.nn.functional.pad(img_4d, (1, 1, 1, 1), mode="reflect")

            dx = torch.nn.functional.conv2d(padded, sobel_x)
            dy = torch.nn.functional.conv2d(padded, sobel_y)

            dx = dx.squeeze(0).squeeze(0) * strength
            dy = dy.squeeze(0).squeeze(0) * strength

            if output_format == "DirectX (Y-)":
                dy = -dy

            nx = -dx
            ny = -dy
            nz = torch.ones_like(nx)

            length = torch.sqrt(nx**2 + ny**2 + nz**2)
            length = torch.clamp(length, min=1e-6)

            nx = nx / length
            ny = ny / length
            nz = nz / length

            normal_map = torch.stack([nx, ny, nz], dim=-1)
            results.append(normal_to_image(normal_map))

        return (torch.stack(results),)
