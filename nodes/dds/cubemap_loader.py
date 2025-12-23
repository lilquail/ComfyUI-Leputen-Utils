"""
Cubemap Face Loader node - loads 6 cubemap face images by naming pattern.
"""

import os
import re

from PIL import Image

from ...core import (
    LEPUTEN_UTILS_CATEGORY,
    log_info,
    log_verbose,
    log_warning,
    normalize_path,
    pil2tensor,
)

# Common cubemap face naming patterns (case-insensitive)
FACE_PATTERNS = {
    # DirectX/standard suffixes
    "pos_x": [r"[_-]?(pos[_-]?x|px|\+x|right|rt|posx)", r"_0$", r"^0$"],
    "neg_x": [r"[_-]?(neg[_-]?x|nx|-x|left|lf|negx)", r"_1$", r"^1$"],
    "pos_y": [r"[_-]?(pos[_-]?y|py|\+y|top|up|tp|posy)", r"_2$", r"^2$"],
    "neg_y": [r"[_-]?(neg[_-]?y|ny|-y|bottom|down|dn|bt|negy)", r"_3$", r"^3$"],
    "pos_z": [r"[_-]?(pos[_-]?z|pz|\+z|front|ft|posz)", r"_4$", r"^4$"],
    "neg_z": [r"[_-]?(neg[_-]?z|nz|-z|back|bk|negz)", r"_5$", r"^5$"],
}

FACE_ORDER = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"]
FACE_DISPLAY = ["+X (Right)", "-X (Left)", "+Y (Top)", "-Y (Bottom)", "+Z (Front)", "-Z (Back)"]


def match_face(filename: str, base_name: str) -> str | None:
    """
    Attempts to match a filename to a cubemap face.

    Args:
        filename: The filename without extension (case-insensitive)
        base_name: The base name pattern to strip before matching

    Returns:
        Face key (e.g., "pos_x") or None if no match
    """
    # Remove extension and convert to lowercase
    name_lower = filename.lower()
    base_lower = base_name.lower() if base_name else ""

    # Strip base name if present
    suffix = name_lower[len(base_lower) :] if base_lower and name_lower.startswith(base_lower) else name_lower

    # Try to match against each face pattern
    for face_key, patterns in FACE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, suffix, re.IGNORECASE):
                return face_key

    return None


def find_cubemap_faces(directory: str, base_pattern: str) -> dict[str, str]:
    """
    Scans a directory for cubemap face images matching a base pattern.

    Args:
        directory: Directory to scan
        base_pattern: Base filename pattern (e.g., "sky" matches sky_px.png, sky_right.jpg, etc.)

    Returns:
        Dict mapping face keys to full file paths
    """
    if not os.path.isdir(directory):
        return {}

    valid_extensions = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".tif", ".dds", ".webp"}
    found_faces = {}

    base_lower = base_pattern.lower() if base_pattern else ""

    for entry in os.scandir(directory):
        if not entry.is_file():
            continue

        name, ext = os.path.splitext(entry.name)
        if ext.lower() not in valid_extensions:
            continue

        # Check if filename starts with base pattern (case-insensitive)
        if base_lower and not name.lower().startswith(base_lower):
            continue

        # Try to match to a face
        face_key = match_face(name, base_pattern)
        if face_key and face_key not in found_faces:
            found_faces[face_key] = entry.path

    return found_faces


class LoadCubemapFaces:
    """
    Loads 6 cubemap face images from a directory using naming pattern matching.

    The node searches for files matching a base pattern with standard cubemap
    face suffixes. Supports multiple naming conventions:

    - Axis notation: _px, _nx, _py, _ny, _pz, _nz
    - Direction names: _right, _left, _top, _bottom, _front, _back
    - Numeric indices: _0 through _5
    - Blender style: _posx, _negx, _posy, _negy, _posz, _negz

    All matching is case-insensitive.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "tooltip": "Directory containing the cubemap face images."}),
                "base_pattern": (
                    "STRING",
                    {
                        "default": "cubemap",
                        "tooltip": "Base filename pattern. E.g., 'sky' matches sky_px.png, sky_right.jpg, etc. Leave empty to match any file with face suffixes.",
                    },
                ),
            },
            "optional": {
                "alpha_bleed": ("BOOLEAN", {"default": False, "tooltip": "Apply alpha bleeding to loaded images."}),
                "blur_radius": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "Blur radius for alpha bleeding.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z")
    OUTPUT_TOOLTIPS = (
        "+X face (Right)",
        "-X face (Left)",
        "+Y face (Top)",
        "-Y face (Bottom)",
        "+Z face (Front)",
        "-Z face (Back)",
    )

    FUNCTION = "load_faces"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DDS"
    OUTPUT_NODE = True  # Enable ui dict return for inline text display
    DESCRIPTION = """Loads 6 cubemap face images by pattern matching.

Supports naming conventions:
• Axis: _px, _nx, _py, _ny, _pz, _nz
• Direction: _right, _left, _top, _bottom, _front, _back
• Index: _0 through _5

All matching is case-insensitive."""

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        """Validates that directory exists."""
        if not directory:
            return "Please specify a directory"

        normalized = normalize_path(directory)
        if not os.path.isdir(normalized):
            return f"Directory not found: {normalized}"

        return True

    def load_faces(
        self,
        directory: str,
        base_pattern: str,
        alpha_bleed: bool = False,
        blur_radius: float = 0.0,
    ):
        """
        Loads cubemap faces from directory by pattern matching.

        Returns:
            Dict with 'ui' for inline display and 'result' with 6 image tensors
        """
        from ...core import unpremultiply_alpha

        directory = normalize_path(directory)
        log_info("LoadCubemapFaces", f"Scanning {directory} for pattern '{base_pattern}'")

        # Find matching faces
        faces = find_cubemap_faces(directory, base_pattern)

        if not faces:
            raise ValueError(f"No cubemap faces found in {directory} matching pattern '{base_pattern}'")

        # Check for missing faces
        missing = [FACE_DISPLAY[i] for i, key in enumerate(FACE_ORDER) if key not in faces]
        if missing:
            raise ValueError(f"Missing cubemap faces: {', '.join(missing)}")

        # Report found faces
        found_info = []
        for i, key in enumerate(FACE_ORDER):
            filename = os.path.basename(faces[key])
            found_info.append(f"{FACE_DISPLAY[i]}: {filename}")
            log_verbose("LoadCubemapFaces", f"Found {FACE_DISPLAY[i]}: {filename}")

        # Load all faces
        loaded_tensors = []
        reference_size = None

        for key in FACE_ORDER:
            filepath = faces[key]

            try:
                pil_image = Image.open(filepath)
                pil_image = pil_image.convert("RGBA")

                # Check size consistency
                if reference_size is None:
                    reference_size = pil_image.size
                elif pil_image.size != reference_size:
                    log_warning(
                        "LoadCubemapFaces",
                        f"{os.path.basename(filepath)} has size {pil_image.size}, expected {reference_size}. Resizing.",
                    )
                    pil_image = pil_image.resize(reference_size, Image.Resampling.LANCZOS)

                # Apply alpha bleeding if requested
                if alpha_bleed:
                    pil_image = unpremultiply_alpha(pil_image, blur_radius)

                # Convert to RGB for output
                pil_rgb = pil_image.convert("RGB")
                tensor = pil2tensor(pil_rgb)
                loaded_tensors.append(tensor)

            except Exception as e:
                raise RuntimeError(f"Failed to load {filepath}: {e}") from e

        # Build progress text for inline display
        progress_text = f"Found 6 faces ({reference_size[0]}x{reference_size[1]})\n" + "\n".join(found_info)
        log_info("LoadCubemapFaces", f"Loaded 6 faces ({reference_size[0]}x{reference_size[1]})")

        return {"ui": {"progress_text": [progress_text]}, "result": tuple(loaded_tensors)}
