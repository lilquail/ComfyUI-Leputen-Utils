"""
Color Space Converter node - converts between sRGB and Linear color spaces.
"""
import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class ColorSpaceConverter:
    """
    Converts images between different color spaces, which is crucial for
    maintaining color accuracy in Physically Based Rendering (PBR) and
    linear lighting workflows.
    """
    CONVERSION_MODES = ["sRGB to Linear", "Linear to sRGB"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image tensor to convert between color spaces."}),
                "conversion": (cls.CONVERSION_MODES, {"tooltip": "The type of color space conversion to perform."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = "Converts images between sRGB and Linear color spaces, vital for accurate PBR workflows."

    def convert(self, image: torch.Tensor, conversion: str) -> tuple[torch.Tensor]:
        gamma = 2.2
        image_rgb = image[:, :, :, :3]

        if conversion == "sRGB to Linear":
            converted_rgb = torch.pow(image_rgb, gamma)
        elif conversion == "Linear to sRGB":
            converted_rgb = torch.pow(image_rgb, 1.0 / gamma)

        converted_rgb = torch.clamp(converted_rgb, 0.0, 1.0)

        if image.shape[-1] == 4:
            alpha_channel = image[:, :, :, 3:4]
            output_image = torch.cat((converted_rgb, alpha_channel), dim=-1)
        else:
            output_image = converted_rgb

        return (output_image,)
