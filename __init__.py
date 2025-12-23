# ComfyUI-Leputen-Utils - Custom nodes for game texture processing  # noqa: N999
# Root package entry point for ComfyUI node discovery

import os
import subprocess

# Register custom API routes (must be imported early)
from . import api
from .core import BASE_DIR, log_error, log_info

# Import all node classes from the nodes package
from .nodes import (
    AddNormals,
    ChannelOperations,
    # Utility
    ColorSpaceConverter,
    CubemapAssembler,
    DDSIterator,
    # DDS
    DDSLoader,
    DDSSaver,
    # DeepBump
    DeepBumpColorToNormal,
    DeepBumpNormalToCurvature,
    DeepBumpNormalToHeight,
    DeepBumpUpscale,
    Equalize,
    EquirectangularToCubemap,
    ExtractFineDetails,
    # PBR
    GenerateAOMap,
    GigapixelCLI,
    HeightAdjustment,
    HeightToNormal,
    HistogramMatcher,
    ImageIterator,
    # Image
    ImageLoadLeputen,
    ImageSaveLeputen,
    LoadCubemapFaces,
    NormalizeNormals,
    NormalMapConverter,
    NormalMapStrength,
    ResizePowerOf2,
    RoughnessGlossinessConverter,
    ZStack,
)

# --- Startup/Dependency Handling ---
deepbump_dir = os.path.join(BASE_DIR, "vendor", "DeepBump")
deepbump_repo = "https://github.com/lilquail/DeepBump-dml.git"


def check_and_clone_deepbump():
    """Checks for the DeepBump directory and clones it if it doesn't exist."""
    if not os.path.isdir(deepbump_dir):
        log_info(
            "DeepBump Setup",
            f"DeepBump not found. Cloning repository from {deepbump_repo}...",
        )
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            subprocess.run(["git", "clone", deepbump_repo, deepbump_dir], check=True)
            log_info("DeepBump Setup", "DeepBump cloned successfully.")
        except FileNotFoundError:
            log_error(
                "DeepBump Setup",
                "Git is not installed or not found in your system's PATH.",
            )
            log_info("DeepBump Setup", f"Manually clone the repository into: {deepbump_dir}")
        except subprocess.CalledProcessError as e:
            log_error(
                "DeepBump Setup",
                f"Error cloning DeepBump repository: {e.stderr.strip() if e.stderr else e}",
            )
            log_info(
                "DeepBump Setup",
                f"Please manually clone the repository into: {deepbump_dir}",
            )
        except Exception as e:
            log_error(
                "DeepBump Setup",
                f"An unexpected error occurred during DeepBump cloning: {e}",
            )
            log_info(
                "DeepBump Setup",
                f"Please manually clone the repository into: {deepbump_dir}",
            )


# Run the check when the module is loaded
check_and_clone_deepbump()

# --- Node Registration ---
# All node classes have been imported at the top of the file

NODE_CLASS_MAPPINGS = {
    "DDSLoader": DDSLoader,
    "DDSSaver": DDSSaver,
    "DDSIterator": DDSIterator,
    "CubemapAssembler": CubemapAssembler,
    "EquirectangularToCubemap": EquirectangularToCubemap,
    "LoadCubemapFaces": LoadCubemapFaces,
    "ImageIterator": ImageIterator,
    "ImageLoadLeputen": ImageLoadLeputen,
    "ImageSaveLeputen": ImageSaveLeputen,
    "DeepBump_ColorToNormal": DeepBumpColorToNormal,
    "DeepBump_NormalToHeight": DeepBumpNormalToHeight,
    "DeepBump_NormalToCurvature": DeepBumpNormalToCurvature,
    "DeepBump_Upscale": DeepBumpUpscale,
    "NormalMapConverter": NormalMapConverter,
    "NormalMapStrength": NormalMapStrength,
    "AddNormals": AddNormals,
    "NormalizeNormals": NormalizeNormals,
    "ExtractFineDetails": ExtractFineDetails,
    "HeightToNormal": HeightToNormal,
    "ColorSpaceConverter": ColorSpaceConverter,
    "HeightAdjustment": HeightAdjustment,
    "Equalize": Equalize,
    "GigapixelCLI": GigapixelCLI,
    "GenerateAOMap": GenerateAOMap,
    "RoughnessGlossinessConverter": RoughnessGlossinessConverter,
    "HistogramMatcher": HistogramMatcher,
    "ChannelOperations": ChannelOperations,
    "ResizePowerOf2": ResizePowerOf2,
    "ZStack": ZStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDSLoader": "Load DDS Image",
    "DDSSaver": "Save DDS Image",
    "DDSIterator": "Load DDS Images (Batch)",
    "CubemapAssembler": "Cubemap Assembler",
    "EquirectangularToCubemap": "Equirectangular to Cubemap",
    "LoadCubemapFaces": "Load Cubemap Faces",
    "ImageIterator": "Load Images (Batch)",
    "ImageLoadLeputen": "Load Image (Leputen)",
    "ImageSaveLeputen": "Save Image (Leputen)",
    "DeepBump_ColorToNormal": "DeepBump Color to Normal",
    "DeepBump_NormalToHeight": "DeepBump Normal to Height",
    "DeepBump_NormalToCurvature": "DeepBump Normal to Curvature",
    "DeepBump_Upscale": "DeepBump Upscale",
    "NormalMapConverter": "Normal Map Format Converter",
    "NormalMapStrength": "Normal Map Strength",
    "AddNormals": "Add Normals",
    "NormalizeNormals": "Normal Map Normalize",
    "ExtractFineDetails": "Extract Fine Details",
    "HeightToNormal": "Height to Normal Map",
    "ColorSpaceConverter": "Color Space Converter",
    "HeightAdjustment": "Height Map Adjust",
    "Equalize": "Equalize",
    "GigapixelCLI": "Gigapixel CLI",
    "GenerateAOMap": "Generate AO Map",
    "RoughnessGlossinessConverter": "Roughness/Glossiness Converter",
    "HistogramMatcher": "Histogram Matcher",
    "ChannelOperations": "Channel Operations",
    "ResizePowerOf2": "Resize Power of 2",
    "ZStack": "Z-Stack (Median/Mean)",
}

# Web directory for JavaScript extensions (settings panel, etc.)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
