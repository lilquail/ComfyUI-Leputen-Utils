"""
Nodes package for ComfyUI-Leputen-Utils.

This package organizes custom nodes into subpackages:
- dds: DDS texture loading, saving, and cubemap processing.
- deepbump: AI-powered normal and height map generation.
- image: Enhanced standard image loading and batching.
- pbr: Physically Based Rendering map utilities (normals, AO, etc.).
- utility: General image adjustment and channel operations.
"""  # noqa: N999

from .dds import (
    CubemapAssembler,
    DDSIterator,
    DDSLoader,
    DDSSaver,
    EquirectangularToCubemap,
    LoadCubemapFaces,
)
from .deepbump import (
    DeepBumpColorToNormal,
    DeepBumpNormalToCurvature,
    DeepBumpNormalToHeight,
    DeepBumpUpscale,
)
from .image import (
    ImageIterator,
    ImageLoadLeputen,
    ImageSaveLeputen,
)
from .pbr import (
    AddNormals,
    ExtractFineDetails,
    GenerateAOMap,
    HeightToNormal,
    NormalizeNormals,
    NormalMapConverter,
    NormalMapStrength,
    RoughnessGlossinessConverter,
)
from .utility import (
    ChannelOperations,
    ColorSpaceConverter,
    Equalize,
    GigapixelCLI,
    HeightAdjustment,
    HistogramMatcher,
    ResizePowerOf2,
    ZStack,
)

# All node classes for export
NODE_CLASSES = [
    # DDS
    DDSLoader,
    DDSSaver,
    DDSIterator,
    CubemapAssembler,
    EquirectangularToCubemap,
    LoadCubemapFaces,
    # Image
    ImageLoadLeputen,
    ImageSaveLeputen,
    ImageIterator,
    # PBR
    GenerateAOMap,
    AddNormals,
    NormalMapConverter,
    ExtractFineDetails,
    NormalizeNormals,
    NormalMapStrength,
    HeightToNormal,
    RoughnessGlossinessConverter,
    # DeepBump
    DeepBumpColorToNormal,
    DeepBumpNormalToHeight,
    DeepBumpNormalToCurvature,
    DeepBumpUpscale,
    # Utility
    ColorSpaceConverter,
    Equalize,
    HeightAdjustment,
    HistogramMatcher,
    GigapixelCLI,
    ChannelOperations,
    ResizePowerOf2,
    ZStack,
]

__all__ = [
    # DDS
    "DDSLoader",
    "DDSSaver",
    "DDSIterator",
    "CubemapAssembler",
    "EquirectangularToCubemap",
    "LoadCubemapFaces",
    # Image
    "ImageLoadLeputen",
    "ImageSaveLeputen",
    "ImageIterator",
    "RoughnessGlossinessConverter",
    # PBR
    "GenerateAOMap",
    "AddNormals",
    "NormalMapConverter",
    "ExtractFineDetails",
    "NormalizeNormals",
    "NormalMapStrength",
    "HeightToNormal",
    # DeepBump
    "DeepBumpColorToNormal",
    "DeepBumpNormalToHeight",
    "DeepBumpNormalToCurvature",
    "DeepBumpUpscale",
    # Utility
    "ColorSpaceConverter",
    "Equalize",
    "HeightAdjustment",
    "HistogramMatcher",
    "GigapixelCLI",
    "ChannelOperations",
    "ResizePowerOf2",
    "ZStack",
    # List
    "NODE_CLASSES",
]
