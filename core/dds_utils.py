"""
DDS-specific utilities for texture loading and format detection.

This module handles DDS file operations including:
- Format and color space detection via header parsing
- DDS to PNG conversion (via texconv DLL or subprocess)
- Batch conversion operations
"""

import os
from typing import Optional

from PIL import Image

from .utils import (
    TEXCONV_PATH,
    TEXDIAG_PATH,
    log_error,
    log_info,
    log_verbose,
    log_warning,
    normalize_path,
    unpremultiply_alpha,
)

# --- DLL Support ---
# Lazy import to avoid circular imports and allow fallback
_texconv_dll_available: Optional[bool] = None


def _check_texconv_dll() -> bool:
    """Check if the texconv DLL is available (lazy initialization)."""
    global _texconv_dll_available
    if _texconv_dll_available is None:
        try:
            from .texconv import texconv_available

            _texconv_dll_available = texconv_available()
        except ImportError:
            _texconv_dll_available = False
    return _texconv_dll_available


def _get_texconv_dll():
    """Get the texconv DLL instance."""
    from .texconv import get_texconv

    return get_texconv()


# --- DDS Conversion Functions ---


def convert_single_dds_to_pil(dds_path: str, alpha_bleed: bool = True, blur_radius: float = 0.0) -> Image.Image:
    """Converts a single DDS file to a PIL Image, with optional alpha bleeding.

    Uses the texconv DLL if available, falling back to subprocess-based conversion.
    The temporary TGA file created during conversion is cleaned up after loading.
    TGA is used instead of PNG for faster encoding/decoding.

    Args:
        dds_path: Path to the DDS file.
        alpha_bleed: Whether to apply alpha bleeding (unpremultiply alpha).
        blur_radius: Gaussian blur radius for alpha bleeding.

    Returns:
        PIL Image in RGBA mode.
    """
    import folder_paths

    temp_dir = folder_paths.get_temp_directory()
    base_filename = os.path.basename(dds_path)
    tga_filename = os.path.splitext(base_filename)[0] + ".tga"
    tga_path = os.path.join(temp_dir, tga_filename)

    # Try DLL-based conversion first
    if _check_texconv_dll():
        _convert_dds_via_dll(dds_path, temp_dir, "DDS Conversion")
    else:
        _convert_dds_via_subprocess(dds_path, temp_dir, "DDS Conversion")

    if not os.path.exists(tga_path):
        log_error(
            "DDS Conversion",
            f"Conversion failed. TGA file not found at {tga_path} after texconv execution for {dds_path}.",
        )
        raise FileNotFoundError(f"Conversion failed. TGA file not found at {tga_path} for {dds_path}.")

    try:
        # Open and load image data into memory
        with Image.open(tga_path) as img:
            # Load the image data before closing the file
            i = img.convert("RGBA")
            # Force load into memory so we can delete the file
            i.load()

        if alpha_bleed:
            i = unpremultiply_alpha(i, blur_radius)

        return i
    finally:
        # Clean up temporary TGA file
        if tga_path and os.path.exists(tga_path):
            try:
                os.remove(tga_path)
            except OSError as e:
                log_warning("DDS Conversion", f"Failed to clean up temp file {tga_path}: {e}")


def _convert_dds_via_dll(dds_path: str, output_dir: str, node_name: str = "DDS"):
    """Convert a DDS file to TGA using the texconv DLL."""
    texconv = _get_texconv_dll()
    args = ["-ft", "tga", "-o", output_dir, "-y", "--", dds_path]
    result, error = texconv.convert(args, verbose=False)
    if result != 0:
        log_error(node_name, f"texconv DLL failed for {dds_path}: {error}")
        raise RuntimeError(f"texconv DLL failed for {dds_path}: {error}")


def _convert_dds_via_subprocess(dds_path: str, output_dir: str, node_name: str = "DDS"):
    """Convert a DDS file to TGA using subprocess (fallback method)."""
    import shutil
    import subprocess

    if not os.path.exists(TEXCONV_PATH) and not shutil.which(TEXCONV_PATH):
        log_error(node_name, f"texconv.exe not found at {TEXCONV_PATH}. Please ensure it's installed and accessible.")
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}")

    command = [TEXCONV_PATH, "-ft", "tga", "-o", output_dir, "-y", dds_path]

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        if process.returncode != 0:
            log_error(node_name, f"texconv.exe failed for {dds_path}. Stderr: {process.stderr.strip()}")
            raise RuntimeError(f"texconv.exe failed for {dds_path}. See console for details.")
    except subprocess.TimeoutExpired as e:
        log_error(node_name, f"texconv.exe timed out for {dds_path}")
        raise RuntimeError(f"texconv.exe timed out for {dds_path}") from e
    except subprocess.CalledProcessError as e:
        log_error(node_name, f"texconv.exe failed for {dds_path}. Stderr: {e.stderr.strip()}")
        raise RuntimeError(f"texconv.exe failed for {dds_path}. See console for details.") from e
    except FileNotFoundError as e:
        log_error(node_name, f"texconv.exe not found at {TEXCONV_PATH}. Please ensure it's installed and accessible.")
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}") from e
    except Exception as e:
        log_error(node_name, f"An unexpected error occurred during DDS conversion of {dds_path}: {e}")
        raise RuntimeError(f"Unexpected error during DDS conversion of {dds_path}.") from e


def convert_dds_batch_to_tga_paths(dds_paths: list[str], output_dir: str, chunk_id: int = 0) -> list[str]:
    """
    Converts a list of DDS files to TGAs in a specified output directory.

    Uses the texconv DLL if available (processing files individually, which is fast
    due to no process spawn overhead), or falls back to subprocess batch mode.
    TGA is used instead of PNG for faster encoding.

    Returns a list of paths to the converted TGA files.
    """
    if not dds_paths:
        return []

    node_name = "DDS Batch Conversion"

    # Try DLL-based conversion first (parallel for speed)
    if _check_texconv_dll():
        import concurrent.futures

        texconv = _get_texconv_dll()

        def convert_single(dds_path: str) -> tuple[str, bool, str]:
            args = ["-ft", "tga", "-o", output_dir, "-y", "--", dds_path]
            result, error = texconv.convert(args, verbose=False)
            return (dds_path, result == 0, error)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(convert_single, p) for p in dds_paths]
            for future in concurrent.futures.as_completed(futures):
                dds_path, success, error = future.result()
                if not success:
                    log_warning(node_name, f"texconv DLL failed for {dds_path}: {error}")
    else:
        # Subprocess fallback with file list for batch processing
        _convert_dds_batch_via_subprocess(dds_paths, output_dir, chunk_id, node_name)

    # Collect converted files
    converted_tga_paths = []
    for dds_path in dds_paths:
        base_filename = os.path.basename(dds_path)
        tga_filename = os.path.splitext(base_filename)[0] + ".tga"
        tga_path = os.path.join(output_dir, tga_filename)
        if os.path.exists(tga_path):
            converted_tga_paths.append(tga_path)
        else:
            log_warning(
                node_name,
                f"Expected TGA not found after conversion for {dds_path} at {tga_path}. This DDS file might have failed silently.",
            )

    return converted_tga_paths


def _convert_dds_batch_via_subprocess(dds_paths: list[str], output_dir: str, chunk_id: int, node_name: str):
    """Convert multiple DDS files to TGA using subprocess with file list (fallback method)."""
    import shutil
    import subprocess

    if not os.path.exists(TEXCONV_PATH) and not shutil.which(TEXCONV_PATH):
        log_error(node_name, f"texconv.exe not found at {TEXCONV_PATH}. Please ensure it's installed and accessible.")
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}")

    # Create a temporary file to list all DDS paths
    dds_list_file = os.path.join(output_dir, f"dds_list_{chunk_id}.txt")
    with open(dds_list_file, "w") as f:
        for p in dds_paths:
            f.write(f"{p}\n")

    command = [
        TEXCONV_PATH,
        "-ft",
        "tga",
        "-o",
        output_dir,
        "-y",  # Overwrite existing files
        "-flist",
        dds_list_file,
    ]

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        if process.returncode != 0:
            log_error(node_name, f"texconv.exe failed for batch {chunk_id}. Stderr: {process.stderr.strip()}")
            raise RuntimeError(f"texconv.exe failed for batch {chunk_id}. See console for details.")
    except subprocess.TimeoutExpired as e:
        log_error(node_name, f"texconv.exe timed out for batch {chunk_id}")
        raise RuntimeError(f"texconv.exe timed out for batch {chunk_id}") from e
    except subprocess.CalledProcessError as e:
        log_error(node_name, f"texconv.exe failed for batch {chunk_id}. Stderr: {e.stderr.strip()}")
        raise RuntimeError(f"texconv.exe failed for batch {chunk_id}. See console for details.") from e
    except FileNotFoundError as e:
        log_error(node_name, f"texconv.exe not found at {TEXCONV_PATH}. Please ensure it's installed and accessible.")
        raise FileNotFoundError(f"texconv.exe not found at {TEXCONV_PATH}") from e
    except Exception as e:
        log_error(node_name, f"An unexpected error occurred during DDS batch conversion {chunk_id}: {e}")
        raise RuntimeError(f"Unexpected error during DDS batch conversion {chunk_id}.") from e
    finally:
        if os.path.exists(dds_list_file):
            os.remove(dds_list_file)


def load_image_and_process(image_path: str, alpha_bleed: bool = True, blur_radius: float = 0.0) -> Image.Image:
    """Loads an image file (TGA, PNG, etc.), applies alpha bleeding if enabled, and returns a PIL Image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    i = Image.open(image_path)
    i = i.convert("RGBA")
    if alpha_bleed:
        i = unpremultiply_alpha(i, blur_radius)
    return i


# Keep old name as alias for backwards compatibility
load_png_and_process = load_image_and_process


# --- DDS Info/Format Detection ---


def get_dds_info(dds_path: str, node_name: str = "DDS") -> tuple[str, str]:
    """
    Gets the format and color space of a DDS file by parsing the header.

    Uses pure Python header parsing for speed. Falls back to texdiag subprocess
    if parsing fails.

    Args:
        dds_path: Path to the DDS file
        node_name: Name of the calling node for logging

    Returns:
        tuple of (format_name, color_space):
            - format_name: e.g., "BC3_UNORM", "BC1_UNORM_SRGB", "Unknown"
            - color_space: "sRGB", "Linear", or "Unknown"
    """
    # Normalize path for cross-platform compatibility
    normalized_path = normalize_path(dds_path)

    # Try pure Python parser first (fast, no subprocess)
    try:
        from .dds_header import DDSHeaderParser
    except ImportError as e:
        log_warning(node_name, f"Failed to import DDS header parser: {e}")
    else:
        try:
            info = DDSHeaderParser.parse(normalized_path)
            log_verbose(node_name, f"Parsed DDS: {info.format_name} -> {info.color_space}")
            return (info.format_name, info.color_space)
        except Exception as e:
            log_warning(node_name, f"Python DDS parser failed for {normalized_path}: {e}")

    # Fallback to texdiag subprocess
    import re
    import subprocess

    try:
        command = [TEXDIAG_PATH, "info", normalized_path]
        process = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=10
        )
        output = process.stdout

        format_match = re.search(r"format\s*=\s*([\w_]+)", output)
        if format_match:
            format_str = format_match.group(1).upper()
            log_info(node_name, f"texdiag format: {format_str}")
            color_space = "sRGB" if "SRGB" in format_str else "Linear"
            return (format_str, color_space)
        return ("Unknown", "Linear")
    except Exception as e:
        log_warning(node_name, f"Could not get DDS info for {normalized_path}, defaulting to 'Unknown': {e}")
        return ("Unknown", "Unknown")


def get_dds_color_space(dds_path: str, node_name: str = "DDS") -> str:
    """
    Determines the color space of a DDS file by parsing the header.

    This is a convenience wrapper around get_dds_info() that returns only the color_space.

    Args:
        dds_path: Path to the DDS file
        node_name: Name of the calling node for logging

    Returns:
        "sRGB", "Linear", or "Unknown"
    """
    _, color_space = get_dds_info(dds_path, node_name)
    return color_space


# Supported DDS formats for combo output (matches DDSSaver format input)
DDS_FORMATS = [
    # Block-Compressed
    "BC1_UNORM",
    "BC2_UNORM",
    "BC3_UNORM",
    "BC4_UNORM",
    "BC5_UNORM",
    "BC6H_UF16",
    "BC6H_SF16",
    "BC7_UNORM",
    # Floating-Point (HDR)
    "R16G16B16A16_FLOAT",
    "R32G32B32A32_FLOAT",
    # Uncompressed RGBA
    "R8G8B8A8_UNORM",
    "B8G8R8A8_UNORM",
    "B8G8R8X8_UNORM",
    "R10G10B10A2_UNORM",
]


def normalize_dds_format(format_name: str) -> str:
    """
    Normalize a DDS format name to match DDSSaver format input options.

    Strips color space suffixes (_SRGB) since color space is handled separately.
    Returns "BC3_UNORM" as fallback for unknown formats.

    Args:
        format_name: Raw format name like "BC3_UNORM_SRGB" or "BC1_UNORM"

    Returns:
        Normalized format name like "BC3_UNORM", or fallback if unknown
    """
    if not format_name or format_name == "Unknown":
        return "BC3_UNORM"  # Safe default

    # Strip _SRGB suffix
    normalized = format_name.upper()
    if normalized.endswith("_SRGB"):
        normalized = normalized[:-5]

    # Return if it's a known format
    if normalized in DDS_FORMATS:
        return normalized

    # Try to find best match for similar formats
    for fmt in DDS_FORMATS:
        if normalized.startswith(fmt.split("_")[0]):
            return fmt

    return "BC3_UNORM"  # Safe fallback
