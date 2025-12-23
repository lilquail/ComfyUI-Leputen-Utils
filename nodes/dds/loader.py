"""
DDS Loader node - loads a single DDS file from ComfyUI input directory.
"""

import os

import folder_paths
import torch

from ...core import (
    DDS_FORMATS,
    LEPUTEN_UTILS_CATEGORY,
    convert_single_dds_to_pil,
    get_changed_hash,
    get_dds_info,
    get_file_info,
    log_error,
    log_verbose,
    normalize_dds_format,
    pil_to_comfy_tensors,
)


class DDSLoader:
    """
    Loads a single DirectDraw Surface (DDS) file from the ComfyUI input directory.
    It converts the DDS file into a standard image tensor and an optional mask,
    handling alpha un-premultiplication (alpha bleeding) to prevent common
    rendering artifacts in game engines or 3D applications.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, _dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.lower().endswith(".dds"):
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(full_path, input_dir)
                    files.append(relative_path.replace("\\", "/"))
        return {
            "required": {
                "image": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "tooltip": "Select a DDS image file to load. This should be placed in the ComfyUI input directory.",
                    },
                ),
                "alpha_bleed": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable alpha bleeding (un-premultiply alpha) to prevent halo artifacts around transparent areas. Essential for game assets.",
                    },
                ),
            },
            "optional": {
                "blur_radius": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Apply a blur to the RGB channels during alpha bleeding for smoother transitions. Only active if Alpha Bleed is enabled.",
                    },
                ),
                "invert_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the output MASK will be inverted (white becomes black, and vice-versa).",
                    },
                ),
            },
        }

    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DDS"
    DESCRIPTION = """
    Loads a single DDS (DirectDraw Surface) texture, performing alpha bleeding and mask inversion if needed.
    
    Useful for loading game textures or environment maps that are already in DDS format.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", DDS_FORMATS, ["sRGB", "Linear"])
    RETURN_NAMES = ("image", "mask", "filename", "path", "format", "color_space")
    FUNCTION = "load_dds"

    @classmethod
    def VALIDATE_INPUTS(cls, image: str, **kwargs):
        """Validates that the selected DDS file exists."""
        if not image:
            return "Please select a DDS file"
        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.exists(image_path):
            return f"DDS file not found: {image_path}"
        if not image.lower().endswith(".dds"):
            return f"File is not a DDS file: {image}"
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Custom change detection to exclude processing-only inputs from cache key."""
        return get_changed_hash(kwargs, {"alpha_bleed", "blur_radius", "invert_mask"})

    def load_dds(
        self, image: str, alpha_bleed: bool, blur_radius: float = 0.0, invert_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, str, str, str, str]:
        """
        Loads a single DDS file, converts it to a PIL Image, applies optional alpha bleeding,
        and returns it as ComfyUI-compatible image and mask tensors.

        Args:
            image (str): The filename of the DDS image to load, relative to the ComfyUI input directory.
            alpha_bleed (bool): If True, applies alpha bleeding (un-premultiplication) to prevent halo artifacts.
            blur_radius (float, optional): Radius for Gaussian blur during alpha bleeding. Only active if `alpha_bleed` is True.
            invert_mask (bool, optional): If True, inverts the generated alpha mask.

        Returns:
            tuple: (image_rgb, alpha_channel, filename_no_ext, directory_path, format_name, color_space)
        """
        image_path = folder_paths.get_annotated_filepath(image)
        log_verbose("DDSLoader", f"Loading DDS file: {image_path}")
        log_verbose(
            "DDSLoader", f"Alpha bleeding: {alpha_bleed}, Blur radius: {blur_radius}, Invert mask: {invert_mask}"
        )

        try:
            # Get format and color space information using shared utility
            raw_format, color_space = get_dds_info(image_path, "DDSLoader")
            format_name = normalize_dds_format(raw_format)  # Strip _SRGB suffix

            # Convert DDS to PIL image with optional alpha bleeding
            pil_image = convert_single_dds_to_pil(image_path, alpha_bleed, blur_radius)

            # Convert to ComfyUI tensors using shared utility
            image_rgb, alpha_channel = pil_to_comfy_tensors(pil_image, invert_mask)

            if invert_mask:
                log_verbose("DDSLoader", "Mask inverted.")

            # Get file info using shared utility
            filename_no_ext, directory_path = get_file_info(image_path)

            log_verbose("DDSLoader", f"Successfully loaded and processed {image_path}")
            return (image_rgb, alpha_channel, filename_no_ext, directory_path, format_name, color_space)
        except Exception as e:
            log_error("DDSLoader", f"Error loading DDS file {image_path}: {e}")
            raise RuntimeError(f"Failed to load DDS file {image_path}: {e}") from e
