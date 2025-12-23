"""
Core library package for ComfyUI-Leputen-Utils.

This package provides centralized utilities for image processing, DDS texture handling,
PBR math, and shared node base classes. It is the architectural foundation of the node pack.
"""  # noqa: N999

from .dds_header import (
    DDSHeaderParser,
    DDSInfo,
    DXGIFormat,
    parse_dds_header,
)
from .dds_utils import (
    DDS_FORMATS,
    convert_dds_batch_to_tga_paths,
    convert_single_dds_to_pil,
    get_dds_color_space,
    get_dds_info,
    load_png_and_process,
    normalize_dds_format,
)
from .loader_base import (
    IteratorLoaderBase,
    ListLoaderBase,
)
from .pbr_utils import (
    image_to_normal,
    normal_to_image,
)
from .texconv import (
    TexconvDLL,
    get_texconv,
    run_texconv,
    texassemble_available,
    texassemble_call,
    texconv_available,
    texconv_convert,
)
from .utils import (
    BASE_DIR,
    BIN_DIR,
    LEPUTEN_UTILS_CATEGORY,
    SCIPY_AVAILABLE,
    TEXCONV_PATH,
    TEXDIAG_PATH,
    calc_power_of_2,
    get_changed_hash,
    get_file_info,
    is_verbose_mode,
    load_exr,
    log_error,
    log_info,
    log_verbose,
    log_warning,
    normalize_path,
    pil2tensor,
    pil_to_comfy_tensors,
    process_standard_image,
    save_exr,
    save_hdr,
    tensor2pil,
    unpremultiply_alpha,
)

__all__ = [
    # Constants
    "LEPUTEN_UTILS_CATEGORY",
    "BASE_DIR",
    "BIN_DIR",
    "TEXCONV_PATH",
    "TEXDIAG_PATH",
    "SCIPY_AVAILABLE",
    # Logging
    "log_info",
    "log_warning",
    "log_error",
    "log_verbose",
    "is_verbose_mode",
    # Tensor/PIL
    "tensor2pil",
    "pil2tensor",
    "pil_to_comfy_tensors",
    "unpremultiply_alpha",
    # File utilities
    "get_file_info",
    "normalize_path",
    "get_changed_hash",
    "process_standard_image",
    "load_exr",
    "save_exr",
    "save_hdr",
    "calc_power_of_2",
    # Texconv
    "TexconvDLL",
    "get_texconv",
    "run_texconv",
    "texconv_available",
    "texconv_convert",
    "texassemble_available",
    "texassemble_call",
    # DDS utilities
    "convert_single_dds_to_pil",
    "convert_dds_batch_to_tga_paths",
    "load_png_and_process",
    "get_dds_info",
    "get_dds_color_space",
    "normalize_dds_format",
    "DDS_FORMATS",
    # DDS header
    "DDSHeaderParser",
    "DDSInfo",
    "DXGIFormat",
    "parse_dds_header",
    # Loader base
    "ListLoaderBase",
    "IteratorLoaderBase",
    # PBR utils
    "image_to_normal",
    "normal_to_image",
]
