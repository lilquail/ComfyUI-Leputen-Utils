"""
Image Load node - enhanced version of ComfyUI LoadImage with alpha handling.
"""

import os

import folder_paths
import torch

import nodes

from ...core import (
    LEPUTEN_UTILS_CATEGORY,
    log_error,
    log_verbose,
    process_standard_image,
)


class ImageLoadLeputen(nodes.LoadImage):
    """
    An enhanced version of the default ComfyUI `Load Image` node. This node
    provides additional controls for alpha channel handling, including
    alpha bleeding (un-premultiplying) to prevent halo artifacts in
    transparent textures, and an option to invert the output mask.

    Unlike the default LoadImage, this node lists images from subfolders
    and supports EXR files (HDR).
    """

    # Supported image extensions (case-insensitive)
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".exr"}

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, _dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in cls.SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(full_path, input_dir)
                    files.append(relative_path.replace("\\", "/"))
        return {
            "required": {
                "image": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "tooltip": "Select an image file to load. Supports files in subfolders and EXR (HDR) format.",
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
                "exr_tone_map": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "For EXR files: Apply Reinhard tone mapping for HDR content. If disabled, values are clamped to [0,1].",
                    },
                ),
            },
        }

    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = """
    Loads a single image with enhanced alpha handling (alpha bleed, mask inversion) and EXR (HDR) support.
    
    Also preserves the relative path/filename structure in the output string, which is useful for
    batch processing workflows that need to mirror the input folder structure.
    """

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "path")
    FUNCTION = "load_image"

    @classmethod
    def VALIDATE_INPUTS(cls, image: str, **kwargs):
        """Validates that the selected image file exists."""
        if not image:
            return "Please select an image file"
        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.exists(image_path):
            return f"Image file not found: {image_path}"
        return True

    def load_image(
        self,
        image: str,
        alpha_bleed: bool,
        blur_radius: float = 0.0,
        invert_mask: bool = False,
        exr_tone_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Loads a single image file, applies optional alpha bleeding, and returns it
        as ComfyUI-compatible image and mask tensors.
        """
        image_path = folder_paths.get_annotated_filepath(image)
        log_verbose("ImageLoadLeputen", f"Loading image file: {image_path}")
        log_verbose(
            "ImageLoadLeputen", f"Alpha bleeding: {alpha_bleed}, Blur radius: {blur_radius}, Invert mask: {invert_mask}"
        )

        try:
            # Load and process image using shared utility
            image_rgb, alpha_channel, _filename, directory_path = process_standard_image(
                image_path,
                alpha_bleed=alpha_bleed,
                blur_radius=blur_radius,
                invert_mask=invert_mask,
                node_name="ImageLoadLeputen",
                exr_tone_map=exr_tone_map,
            )

            # Use the input image string (relative path) for filename output to preserve subfolders
            # e.g. "subfolder/image.png" -> "subfolder/image"
            filename_for_output = os.path.splitext(image)[0]

            return (image_rgb, alpha_channel, filename_for_output, directory_path)
        except FileNotFoundError:
            # Re-raise with specific message for ComfyUI
            log_error("ImageLoadLeputen", f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}") from None
        except Exception as e:
            # Re-raise with descriptive error
            raise RuntimeError(f"Error loading or processing image {image_path}: {e}") from e

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Determines if the node's output needs to be re-evaluated based on input changes."""
        exclude_keys = {"alpha_bleed", "blur_radius", "invert_mask", "exr_tone_map"}
        return super().IS_CHANGED(**{k: v for k, v in kwargs.items() if k not in exclude_keys})
