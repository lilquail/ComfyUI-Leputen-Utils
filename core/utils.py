"""
Core utilities for ComfyUI-Leputen-Utils.

This module contains general-purpose utilities including logging, tensor/PIL conversion,
path handling, and other shared functionality used across all nodes.
"""

import os

# Enable OpenCV EXR support BEFORE cv2 is imported anywhere
# Must be set at module load time to take effect
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
from PIL import Image, ImageFilter

# --- Constants ---
LEPUTEN_UTILS_CATEGORY = "Leputen-Utils"

# --- Executable Paths ---
# BASE_DIR points to the package root (parent of core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(BASE_DIR, "bin")
TEXCONV_PATH = os.path.join(BIN_DIR, "texconv.exe")
TEXDIAG_PATH = os.path.join(BIN_DIR, "texdiag.exe")


# --- Path Utilities ---
def normalize_path(path: str) -> str:
    """
    Normalize a file path for the current OS.

    Converts forward slashes to backslashes on Windows to handle paths
    from sources that use forward slashes (e.g., ComfyUI's folder_paths).

    Args:
        path: The file path to normalize.

    Returns:
        The normalized path string.
    """
    return path.replace("/", os.sep)


def calc_power_of_2(value: int, mode: str) -> int:
    """
    Calculate a power of 2 dimension based on the specified mode.

    Useful for game engine textures that require power of 2 dimensions.

    Args:
        value: The dimension value to convert.
        mode: Resize mode - 'Nearest', 'Up', or 'Down'.
            - 'Nearest': Closest power of 2 (rounds to nearest).
            - 'Up': Next larger or equal power of 2.
            - 'Down': Previous smaller or equal power of 2.

    Returns:
        The power of 2 dimension (minimum 1).

    Examples:
        >>> calc_power_of_2(300, "Nearest")  # -> 256
        >>> calc_power_of_2(300, "Up")       # -> 512
        >>> calc_power_of_2(300, "Down")     # -> 256
        >>> calc_power_of_2(1920, "Nearest") # -> 2048
    """
    if value <= 0:
        return 1
    if mode == "Up":
        # Next power of 2 >= value
        return 1 << (value - 1).bit_length()
    elif mode == "Down":
        # Previous power of 2 <= value
        return 1 << (value.bit_length() - 1)
    else:  # Nearest
        up = 1 << (value - 1).bit_length()
        down = up >> 1 if up > 1 else 1
        return down if (value - down) < (up - value) else up


# --- Logging Configuration ---
# Log levels: Verbose=0, Normal=1, Quiet=2, Silent=3
# Verbose: All messages including progress
# Normal: Standard info and warnings
# Quiet: Warnings and errors only
# Silent: Errors only

_log_level_cache: dict[str, object] = {"level": 1, "last_check": 0}  # Default: Normal


def _get_log_level() -> int:
    """
    Get the current log level from ComfyUI settings.

    Caches the value and refreshes every 10 seconds to avoid disk I/O on every log call.

    Returns:
        0=Verbose, 1=Normal, 2=Quiet, 3=Silent
    """
    import json
    import time

    current_time = time.time()
    # Refresh cache every 10 seconds
    if current_time - _log_level_cache.get("last_check", 0) < 10:
        return _log_level_cache.get("level", 1)

    _log_level_cache["last_check"] = current_time

    try:
        # ComfyUI stores settings in user/default/comfy.settings.json
        import folder_paths

        user_dir = folder_paths.get_user_directory()
        settings_path = os.path.join(user_dir, "default", "comfy.settings.json")

        if os.path.exists(settings_path):
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
                level_str = settings.get("LeputenUtils.LogLevel", "Info")
                level_map = {"Debug": 0, "Info": 1, "Warning": 2, "Error": 3}
                _log_level_cache["level"] = level_map.get(level_str, 1)
    except Exception:
        _log_level_cache["level"] = 1  # Default to Info on error

    return _log_level_cache["level"]


def log_error(node_name: str, message: str):
    """Logs an error message to stderr. Always shown (all log levels)."""
    print(f"ERROR: [{node_name}] {message}")


def log_warning(node_name: str, message: str):
    """Logs a warning message. Shown at Verbose/Normal/Quiet levels."""
    if _get_log_level() <= 2:  # Verbose, Normal, or Quiet
        print(f"WARNING: [{node_name}] {message}")


def log_info(node_name: str, message: str):
    """Logs an informational message. Shown at Verbose/Normal levels."""
    if _get_log_level() <= 1:  # Verbose or Normal
        print(f"INFO: [{node_name}] {message}")


def log_verbose(node_name: str, message: str):
    """Logs a verbose/debug message. Only shown at Verbose level."""
    if _get_log_level() == 0:  # Verbose only
        print(f"DEBUG: [{node_name}] {message}")


def is_verbose_mode() -> bool:
    """Returns True if log verbosity is set to Info or Debug level."""
    return _get_log_level() <= 1  # Verbose at Debug (0) or Info (1)


# --- Dependency Checks ---
try:
    from scipy.ndimage import distance_transform_edt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# --- Tensor/PIL Conversion ---
def tensor2pil(image):
    """Converts a torch tensor to a PIL Image."""
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    """Converts a PIL Image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil_to_comfy_tensors(pil_image: Image.Image, invert_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a PIL RGBA image to ComfyUI-compatible image and mask tensors.

    Args:
        pil_image: PIL Image in RGBA mode
        invert_mask: If True, inverts the alpha mask

    Returns:
        tuple of (image_rgb, alpha_mask):
            - image_rgb: Tensor of shape (1, H, W, 3) with RGB values in [0, 1]
            - alpha_mask: Tensor of shape (H, W) with mask values in [0, 1]
    """
    image_data = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_data)[None,]

    image_rgb = image_tensor[:, :, :, :3]
    # Alpha channel is inverted: 0 (transparent) -> 1 (white in mask), 1 (opaque) -> 0 (black in mask)
    alpha_channel = 1.0 - image_tensor[0, :, :, 3]

    if invert_mask:
        alpha_channel = 1.0 - alpha_channel

    return image_rgb, alpha_channel


# --- Alpha Processing ---
def unpremultiply_alpha(image, blur_radius=0.0):
    """Corrects the RGB channels of a PIL Image with an alpha channel.
    It replaces the color of transparent pixels with the color of the nearest visible pixels.
    This is also known as alpha bleeding or un-premultiplying.

    Args:
        image: PIL RGBA Image.
        blur_radius (float, optional): Gaussian blur radius for RGB channels after bleeding (0 = none). Defaults to 0.0.

    Returns:
        PIL RGBA Image with filled and optionally blurred RGB.
    """
    if image.mode != "RGBA":
        return image

    # Check for transparency
    img_np = np.array(image)
    alpha = img_np[:, :, 3]
    mask = alpha == 0
    if not np.any(mask):
        return image  # No transparency, no change

    if not SCIPY_AVAILABLE:
        log_warning(
            "Alpha Bleed",
            "SciPy not installed, which is required for the 'nearest' fill method. Skipping alpha bleed.",
        )
        if blur_radius > 0:
            log_warning("Alpha Bleed", "Cannot apply blur without bleeding.")
        return image
    # Nearest-pixel bleeding logic
    binary_map = np.where(mask, 1, 0)
    distances, indices = distance_transform_edt(binary_map, return_indices=True)
    bleed_img = img_np[indices[0], indices[1], :]
    bleed_img[:, :, 3] = alpha  # Preserve original alpha
    temp_image = Image.fromarray(bleed_img.astype(np.uint8))

    # Apply blur if radius > 0, masked to transparent areas
    if blur_radius > 0:
        try:
            # Get current RGB from temp_image
            current_rgb_np = np.array(temp_image)[:, :, :3]
            current_rgb_pil = Image.fromarray(current_rgb_np)

            # Blur it
            blurred_pil = current_rgb_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_np = np.array(blurred_pil)

            # Blend: Opaque uses current (filled) RGB, transparent uses blurred
            mask_float = alpha / 255.0
            final_rgb = current_rgb_np * mask_float[..., np.newaxis] + blurred_np * (1 - mask_float[..., np.newaxis])

            # Stack with alpha
            final_img_np = np.dstack((final_rgb, alpha))
            final_image = Image.fromarray(final_img_np.astype(np.uint8))
        except Exception as e:
            log_warning("Alpha Bleed", f"Blur failed ({e}). Using unblurred fill.")
            final_image = temp_image
    else:
        final_image = temp_image

    return final_image


# --- EXR Loading ---
def load_exr(file_path: str, tone_map: bool = False) -> np.ndarray:
    """
    Loads an EXR file and returns it as a float32 RGBA numpy array.

    Uses OpenCV with EXR support enabled.

    Args:
        file_path: Path to the EXR file
        tone_map: If True, applies simple Reinhard tone mapping.
                  If False, preserves raw HDR values (may be > 1.0).

    Returns:
        numpy array of shape (H, W, 4) with float32 dtype, RGBA format
    """
    import cv2

    # Read EXR with OpenCV (EXR support enabled at module load)
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if data is None:
        raise RuntimeError(f"OpenCV failed to read EXR file: {file_path}")

    log_verbose("EXR Loader", f"Loaded EXR: shape={data.shape}, dtype={data.dtype}")

    # Ensure float32
    data = data.astype(np.float32)

    # Log HDR range info
    log_verbose("EXR Loader", f"Value range: min={data.min():.4f}, max={data.max():.4f}")

    # Handle different channel configurations
    if len(data.shape) == 2:
        # Grayscale - expand to RGB
        data = np.stack([data, data, data], axis=-1)

    # OpenCV loads as BGR(A), convert to RGB(A)
    if data.shape[-1] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        alpha = np.ones((*data.shape[:-1], 1), dtype=np.float32)
    elif data.shape[-1] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
        alpha = data[..., 3:4]
        data = data[..., :3]
    else:
        # Single channel - expand to RGB
        data = np.stack([data[..., 0]] * 3, axis=-1)
        alpha = np.ones((*data.shape[:-1], 1), dtype=np.float32)

    rgb = data

    if tone_map:
        # Simple Reinhard tone mapping for HDR -> LDR
        log_verbose("EXR Loader", "Applying Reinhard tone mapping")
        rgb = rgb / (1.0 + rgb)
    # else: preserve raw HDR values (no clipping!)

    # Combine RGBA (keep as float32)
    rgba = np.concatenate([rgb, alpha], axis=-1)

    return rgba


def save_exr(image_np: np.ndarray, file_path: str) -> None:
    """
    Saves a float32 numpy array as an EXR file.

    Image data typically expected in RGB or RGBA format.
    Uses OpenCV for saving.

    Args:
        image_np: Numpy array of shape (H, W, C) or (H, W)
        file_path: Output file path (must end in .exr)
    """
    import cv2

    if not file_path.lower().endswith(".exr"):
        raise ValueError("File path must end with .exr")

    # Ensure float32
    data = image_np.astype(np.float32)

    # OpenCV expects BGR(A), convert from RGB(A)
    if len(data.shape) == 3:
        if data.shape[-1] == 3:
            # RGB -> BGR
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        elif data.shape[-1] == 4:
            # RGBA -> BGRA
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)

    # Save
    success = cv2.imwrite(file_path, data)
    if not success:
        raise RuntimeError(f"Failed to save EXR file: {file_path}")

    log_verbose("EXR Saver", f"Saved EXR: {file_path}")


def save_hdr(image_np: np.ndarray, file_path: str) -> None:
    """
    Saves a float32 numpy array as a Radiance HDR file.

    Image data typically expected in RGB or RGBA format.
    Uses OpenCV for saving.

    Args:
        image_np: Numpy array of shape (H, W, C) or (H, W)
        file_path: Output file path (must end in .hdr)
    """
    import cv2

    if not file_path.lower().endswith(".hdr"):
        raise ValueError("File path must end with .hdr")

    # Ensure float32
    data = image_np.astype(np.float32)

    # OpenCV expects BGR, convert from RGB
    # HDR files normally don't support alpha, so we strip it
    if len(data.shape) == 3:
        if data.shape[-1] >= 3:
            # RGB(A) -> BGR(A)
            if data.shape[-1] == 4:
                # Strip alpha for HDR format
                data = data[:, :, :3]

            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

    # Save
    success = cv2.imwrite(file_path, data)
    if not success:
        raise RuntimeError(f"Failed to save HDR file: {file_path}")

    log_verbose("HDR Saver", f"Saved HDR: {file_path}")


def get_file_info(file_path: str) -> tuple[str, str]:
    """
    Extracts filename (without extension) and directory path from a file path.

    Args:
        file_path: Full path to a file

    Returns:
        tuple of (filename_no_ext, directory_path)
    """
    base_filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(base_filename)[0]
    directory_path = os.path.dirname(file_path)
    return filename_no_ext, directory_path


def process_standard_image(
    file_path: str,
    alpha_bleed: bool = True,
    blur_radius: float = 0.0,
    invert_mask: bool = False,
    node_name: str = "ImageLoader",
    exr_tone_map: bool = False,
) -> tuple:
    """
    Loads and processes an image with standardized handle-heavy operations.

    This function centralizes:
    1. Loading via PIL or specialized EXR loader.
    2. Converting to RGBA (preserving alpha where present).
    3. Optional alpha bleeding (unpremultiplying) to prevent halo artifacts.
    4. Optional Reinhardt tone mapping for HDR (EXR) content.
    5. Mask generation and optional inversion.
    6. PIL to ComfyUI-compatible torch tensor conversion.

    Args:
        file_path: Absolute path to the image file.
        alpha_bleed: If True, applies color bleeding to transparent pixels.
        blur_radius: Blur radius for alpha bleeding.
        invert_mask: If True, inverts the output alpha mask.
        node_name: Name of the node for logging context.
        exr_tone_map: If True, applies tone mapping to EXR content.

    Returns:
        tuple: (image_rgb_tensor, alpha_mask_tensor, filename_no_ext, directory_path)
    """
    from PIL import Image  # Local import to avoid circular dependency

    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".exr":
            # EXR loading - returns float32 numpy array
            rgba_np = load_exr(file_path, tone_map=exr_tone_map)

            # Convert numpy to tensor: (H, W, 4) -> (1, H, W, 4)
            image_tensor = torch.from_numpy(rgba_np).unsqueeze(0)

            # Note: alpha_bleed not applied to EXR files
            if alpha_bleed:
                log_warning(node_name, "Alpha bleed is not applied to EXR files")

            image_rgb = image_tensor[:, :, :, :3]
            alpha_channel = 1.0 - image_tensor[0, :, :, 3]

            log_info(
                node_name,
                f"EXR loaded as float32: range=[{image_rgb.min():.4f}, {image_rgb.max():.4f}]",
            )
        else:
            # Standard image loading via PIL
            img = Image.open(file_path).convert("RGBA")
            if alpha_bleed:
                img = unpremultiply_alpha(img, blur_radius)

            image_tensor = pil2tensor(img)
            image_rgb = image_tensor[:, :, :, :3]
            alpha_channel = 1.0 - image_tensor[0, :, :, 3]

        if invert_mask:
            alpha_channel = 1.0 - alpha_channel

        filename_no_ext, directory_path = get_file_info(file_path)

        return (image_rgb, alpha_channel, filename_no_ext, directory_path)
    except Exception as e:
        log_error(node_name, f"Could not load image file {file_path}: {e}")
        raise e


def get_changed_hash(kwargs, exclude_keys=None):
    """Computes a hash for the given kwargs, excluding specified keys."""
    if exclude_keys is None:
        exclude_keys = set()

    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}

    # Use a more stable hashing method by converting values to strings
    hashed_items = [f"{k}:{v}" for k, v in sorted(filtered_kwargs.items())]

    return hash(tuple(hashed_items))
