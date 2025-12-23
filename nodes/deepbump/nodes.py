"""
DeepBump AI-powered texture processing nodes.

This module wraps the DeepBump library for AI-based texture processing:
- Color to Normal map generation
- Normal to Height map conversion
- Normal to Curvature map conversion
- Texture upscaling (2x or 4x)
"""

import os
import sys

import numpy as np
import torch
from comfy.utils import ProgressBar

from ...core import (
    BASE_DIR,
    LEPUTEN_UTILS_CATEGORY,
    is_verbose_mode,
    log_error,
    log_info,
    log_warning,
)

# Try to import tqdm for verbose console progress bars
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    log_warning("DeepBump", "tqdm not installed. Verbose mode will use simple log messages instead of progress bars.")


# --- Constants and Path Setup ---
deepbump_dir = os.path.join(BASE_DIR, "vendor", "DeepBump")
DEEPBUMP_TILE_SIZE = 256  # Tile size used by DeepBump models

# Add DeepBump to sys.path to allow direct imports
if deepbump_dir not in sys.path:
    sys.path.append(deepbump_dir)

try:
    import module_color_to_normals
    import module_lowres_to_highres
    import module_normals_to_curvature
    import module_normals_to_height
    import utils_inference

    DEEPBUMP_AVAILABLE = True
except ImportError as e:
    DEEPBUMP_AVAILABLE = False
    log_error(
        "DeepBump",
        f"Could not import one or more DeepBump modules: {e}. Please ensure the DeepBump submodule is cloned correctly.",
    )


# Log GPU status on startup
if DEEPBUMP_AVAILABLE:
    providers = utils_inference.get_execution_providers()
    primary_provider = providers[0] if providers else "None"
    if primary_provider == "DmlExecutionProvider":
        log_info("DeepBump", "GPU acceleration enabled via DirectML (Windows)")
    else:
        log_info(
            "DeepBump", "GPU acceleration not available, using CPU. Install 'onnxruntime-directml' for GPU support."
        )


# --- Stage name mappings for readable progress ---
STAGE_NAMES = {
    # Color to Normals
    "tiling": "Tiling",
    "loading_model": "Loading Model",
    "generating": "Generating",
    "merging": "Merging",
    # Normals to Height
    "preparing": "Preparing",
    "grid_setup": "Grid Setup",
    "fft_forward": "FFT Forward",
    "integrating": "Integrating",
    "complete": "Complete",
    # Normals to Curvature
    "h_convolution": "H-Convolution",
    "v_convolution": "V-Convolution",
    "h_blur": "H-Blur",
    "v_blur": "V-Blur",
}


def get_stage_name(stage: str) -> str:
    """Convert internal stage name to human-readable name."""
    return STAGE_NAMES.get(stage, stage.replace("_", " ").title())


# --- Nodes ---


class DeepBumpColorToNormal:
    DESCRIPTION = """
    Generates a normal map from a color/albedo image using the DeepBump AI model.
    Works best on tiling textures.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The input color image (albedo/diffuse texture) from which to generate a normal map."},
                ),
                "overlap": (
                    ["SMALL", "MEDIUM", "LARGE"],
                    {
                        "default": "LARGE",
                        "tooltip": "Controls tile blending. LARGE produces smoother results but is slower.",
                    },
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if not DEEPBUMP_AVAILABLE:
            return "DeepBump modules not found. Please ensure the DeepBump submodule is cloned into vendor/DeepBump."
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DeepBump"

    def generate(self, image: torch.Tensor, overlap: str) -> tuple[torch.Tensor]:
        if not DEEPBUMP_AVAILABLE:
            raise ImportError("DeepBump modules not found.")
        verbose = is_verbose_mode()
        results = []
        total_tiles = 0
        tile_size = DEEPBUMP_TILE_SIZE
        overlaps = {"SMALL": tile_size // 6, "MEDIUM": tile_size // 4, "LARGE": tile_size // 2}
        stride_size = tile_size - overlaps[overlap]

        # Count total tiles for progress bar
        for img_tensor in image:
            img_numpy = img_tensor.permute(2, 0, 1).cpu().numpy()
            img_gray = np.mean(img_numpy[0:3], axis=0, keepdims=True).astype(np.float32)
            tiles, _ = utils_inference.tiles_split(img_gray, (tile_size, tile_size), (stride_size, stride_size))
            total_tiles += len(tiles)

        pbar = ProgressBar(total_tiles)
        console_pbar = None
        if verbose and TQDM_AVAILABLE:
            console_pbar = tqdm(total=total_tiles, desc="ColorToNormal", unit="tile", leave=True, file=sys.stderr)
        elif verbose:
            log_info(
                "DeepBumpColorToNormal",
                f"Processing {len(image)} image(s) with {total_tiles} total tiles (overlap: {overlap})",
            )

        for img_tensor in image:
            img_numpy = img_tensor.permute(2, 0, 1).cpu().numpy()

            def progress_callback(current, total):
                # DeepBump calls with (0, total) first, then (1, total), (2, total), etc.
                # Only update on actual progress (current > 0)
                if current > 0:
                    pbar.update(1)
                    if console_pbar:
                        console_pbar.update(1)

            def on_stage(stage_name):
                if console_pbar:
                    console_pbar.set_description(f"ColorToNormal: {get_stage_name(stage_name)}")

            output_numpy = module_color_to_normals.apply(img_numpy, overlap, progress_callback, on_stage)
            output_tensor = torch.from_numpy(output_numpy).float().permute(1, 2, 0).unsqueeze(0)
            results.append(output_tensor)

        if console_pbar:
            console_pbar.n = console_pbar.total  # Ensure 100%
            console_pbar.set_description("ColorToNormal: Complete")
            console_pbar.refresh()
            console_pbar.close()

        return (torch.cat(results, dim=0),)


class DeepBumpNormalToHeight:
    DESCRIPTION = """
    Generates a height/displacement map (grey-scale) from a normal map using the DeepBump AI model.
    Useful for parallax mapping and tessellation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input normal map (DirectX or OpenGL format)."}),
                "seamless": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable for tileable textures to ensure seamless height map edges."},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if not DEEPBUMP_AVAILABLE:
            return "DeepBump modules not found. Please ensure the DeepBump submodule is cloned into vendor/DeepBump."
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DeepBump"

    def generate(self, image: torch.Tensor, seamless: bool) -> tuple[torch.Tensor]:
        if not DEEPBUMP_AVAILABLE:
            raise ImportError("DeepBump modules not found.")
        verbose = is_verbose_mode()
        results = []
        total_steps = len(image) * 4  # preparing + 3 FFT stages

        pbar = ProgressBar(total_steps)
        console_pbar = None
        if verbose and TQDM_AVAILABLE:
            console_pbar = tqdm(total=total_steps, desc="NormalToHeight", unit="stage", leave=True, file=sys.stderr)
        elif verbose:
            log_info("DeepBumpNormalToHeight", f"Processing {len(image)} image(s), seamless={seamless}")

        for img_tensor in image:
            img_numpy = img_tensor.permute(2, 0, 1).cpu().numpy()

            def progress_callback(current, total):
                pbar.update(1)
                if console_pbar:
                    console_pbar.update(1)

            def on_stage(stage_name):
                if console_pbar:
                    console_pbar.set_description(f"NormalToHeight: {get_stage_name(stage_name)}")

            output_numpy = module_normals_to_height.apply(img_numpy, seamless, progress_callback, on_stage)
            output_tensor = torch.from_numpy(output_numpy).float().permute(1, 2, 0).unsqueeze(0)
            results.append(output_tensor)

        if console_pbar:
            console_pbar.n = console_pbar.total  # Ensure 100%
            console_pbar.set_description("NormalToHeight: Complete")
            console_pbar.refresh()
            console_pbar.close()

        return (torch.cat(results, dim=0),)


class DeepBumpNormalToCurvature:
    DESCRIPTION = """
    Generates a curvature map from a normal map highlighting edges and convex/concave areas.
    Useful for wear and tear masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input normal map (DirectX or OpenGL format)."}),
                "blur_radius": (
                    ["SMALLEST", "SMALLER", "SMALL", "MEDIUM", "LARGE", "LARGER", "LARGEST"],
                    {"default": "MEDIUM", "tooltip": "Controls the scale of detected features."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if not DEEPBUMP_AVAILABLE:
            return "DeepBump modules not found. Please ensure the DeepBump submodule is cloned into vendor/DeepBump."
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DeepBump"

    def generate(self, image: torch.Tensor, blur_radius: str) -> tuple[torch.Tensor]:
        if not DEEPBUMP_AVAILABLE:
            raise ImportError("DeepBump modules not found.")
        verbose = is_verbose_mode()
        results = []
        total_steps = len(image) * 5  # 4 processing stages + complete

        pbar = ProgressBar(total_steps)
        console_pbar = None
        if verbose and TQDM_AVAILABLE:
            console_pbar = tqdm(total=total_steps, desc="NormalToCurvature", unit="stage", leave=True, file=sys.stderr)
        elif verbose:
            log_info("DeepBumpNormalToCurvature", f"Processing {len(image)} image(s), blur_radius={blur_radius}")

        for img_tensor in image:
            img_numpy = img_tensor.permute(2, 0, 1).cpu().numpy()

            def progress_callback(current, total):
                pbar.update(1)
                if console_pbar:
                    console_pbar.update(1)

            def on_stage(stage_name):
                if console_pbar:
                    console_pbar.set_description(f"NormalToCurvature: {get_stage_name(stage_name)}")

            output_numpy = module_normals_to_curvature.apply(img_numpy, blur_radius, progress_callback, on_stage)
            output_tensor = torch.from_numpy(output_numpy).float().permute(1, 2, 0).unsqueeze(0)
            results.append(output_tensor)

        if console_pbar:
            console_pbar.n = console_pbar.total  # Ensure 100%
            console_pbar.set_description("NormalToCurvature: Complete")
            console_pbar.refresh()
            console_pbar.close()

        return (torch.cat(results, dim=0),)


class DeepBumpUpscale:
    DESCRIPTION = """
    Upscales textures 2x or 4x using the DeepBump AI model.
    Optimized for game textures and PBR maps, preserving details better than standard upscalers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image(s) to upscale."}),
                "scale_factor": (["x2", "x4"], {"default": "x2", "tooltip": "Upscaling factor."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        if not DEEPBUMP_AVAILABLE:
            return "DeepBump modules not found. Please ensure the DeepBump submodule is cloned into vendor/DeepBump."
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DeepBump"

    def generate(self, image: torch.Tensor, scale_factor: str) -> tuple[torch.Tensor]:
        if not DEEPBUMP_AVAILABLE:
            raise ImportError("DeepBump modules not found.")
        verbose = is_verbose_mode()
        results = []
        total_tiles = 0
        tile_size = DEEPBUMP_TILE_SIZE

        # Count total tiles for progress bar
        for img_tensor in image:
            img_numpy = img_tensor.permute(2, 0, 1).cpu().numpy()
            tiles, _ = module_lowres_to_highres.tiles_split(img_numpy, tile_size)
            total_tiles += len(tiles)

        pbar = ProgressBar(total_tiles)
        console_pbar = None
        if verbose and TQDM_AVAILABLE:
            console_pbar = tqdm(
                total=total_tiles, desc=f"Upscale {scale_factor}", unit="tile", leave=True, file=sys.stderr
            )
        elif verbose:
            log_info(
                "DeepBumpUpscale",
                f"Processing {len(image)} image(s) with {total_tiles} total tiles (scale: {scale_factor})",
            )

        for img_tensor in image:
            img_numpy = img_tensor.permute(2, 0, 1).cpu().numpy()

            def progress_callback(current, total):
                # DeepBump calls with (0, total) first, then (1, total), (2, total), etc.
                # Only update on actual progress (current > 0)
                if current > 0:
                    pbar.update(1)
                    if console_pbar:
                        console_pbar.update(1)

            def on_stage(stage_name):
                if console_pbar:
                    console_pbar.set_description(f"Upscale: {get_stage_name(stage_name)}")

            output_numpy = module_lowres_to_highres.apply(img_numpy, scale_factor, progress_callback, on_stage)
            output_tensor = torch.from_numpy(output_numpy).float().permute(1, 2, 0).unsqueeze(0)
            results.append(output_tensor)

        if console_pbar:
            console_pbar.n = console_pbar.total  # Ensure 100%
            console_pbar.set_description("Upscale: Complete")
            console_pbar.refresh()
            console_pbar.close()

        return (torch.cat(results, dim=0),)
