"""
Equalize node - histogram equalization for contrast enhancement.
"""
import torch
from PIL import ImageOps

from ...core import (
    LEPUTEN_UTILS_CATEGORY,
    pil2tensor,
    tensor2pil,
)


class Equalize:
    """
    Performs histogram equalization on an image to increase its global contrast
    by redistributing pixel intensities more evenly.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image tensor to apply histogram equalization to."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "equalize"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = "Increases image contrast via histogram equalization, making details more visible."

    def equalize(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        from PIL import Image

        results = []
        for img_tensor in image:
            pil_img = tensor2pil(img_tensor.unsqueeze(0))
            pil_gray = pil_img.convert("L")
            equalized_gray = ImageOps.equalize(pil_gray)

            if pil_img.mode == 'RGBA':
                alpha = pil_img.split()[3]
                equalized_pil = Image.merge("RGBA", (equalized_gray, equalized_gray, equalized_gray, alpha))
            else:
                equalized_pil = equalized_gray.convert("RGB")

            results.append(pil2tensor(equalized_pil))

        return (torch.cat(results, dim=0),)
