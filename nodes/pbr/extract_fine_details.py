"""
Extract Fine Details node - high-pass filter for normal map detail extraction.
"""

import torch
import torchvision.transforms.functional as tf

from ...core import LEPUTEN_UTILS_CATEGORY, normal_to_image


class ExtractFineDetails:
    """Extracts fine-grained surface details from a normal map using a high-pass filter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE", {"tooltip": "The input normal map from which to extract fine details."}),
                "detail_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 50.0,
                        "step": 0.1,
                        "tooltip": "Blur radius for high-pass filter. Lower values extract finer details, higher values extract larger features.",
                    },
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Intensity of extracted details. 0 = flat normal, 1 = original intensity, >1 = exaggerated.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_details"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/NormalMap"
    DESCRIPTION = "Extracts fine details from a normal map using a high-pass filter for detail separation."

    def extract_details(
        self, normal_map: torch.Tensor, detail_scale: float, strength: float = 1.0
    ) -> tuple[torch.Tensor]:
        blur_radius = detail_scale
        kernel_size = int(blur_radius) * 2 + 1
        if kernel_size < 3:
            kernel_size = 3

        results = []
        for img in normal_map:
            img_chw = img.permute(2, 0, 1)
            blurred_img_chw = tf.gaussian_blur(img_chw.unsqueeze(0), kernel_size=[kernel_size, kernel_size]).squeeze(0)

            # High-pass filter: original - blurred
            # This extracts the delta between the original and blurred version
            high_pass_rg = img_chw[0:2] - blurred_img_chw[0:2]

            # Apply strength scaling and re-center back to 0.5 (neutral normal)
            high_pass_rg = high_pass_rg * strength + 0.5

            final_r = torch.clamp(high_pass_rg[0], 0.0, 1.0)
            final_g = torch.clamp(high_pass_rg[1], 0.0, 1.0)

            # Reconstruct Z channel to maintain unit vector length
            # x² + y² + z² = 1  => z = sqrt(1 - x² - y²)
            new_x = (final_r * 2.0) - 1.0
            new_y = (final_g * 2.0) - 1.0
            new_z_squared = 1.0 - torch.clamp(new_x**2 + new_y**2, 0.0, 1.0)
            new_z = torch.sqrt(new_z_squared)
            final_b_correct = normal_to_image(new_z)

            final_img_chw = torch.stack([final_r, final_g, final_b_correct])
            results.append(final_img_chw.permute(1, 2, 0))

        return (torch.stack(results),)
