"""
Cubemap Assembler node - creates DDS cubemaps from 6 face images using texassemble.
"""

import os
import shutil
import tempfile

import folder_paths
import numpy as np
import torch

from ...core import (
    DDS_FORMATS,
    LEPUTEN_UTILS_CATEGORY,
    log_info,
    log_verbose,
    log_warning,
    run_texconv,
    tensor2pil,
    texassemble_available,
    texassemble_call,
)


class CubemapAssembler:
    """
    Assembles a DDS cubemap from 6 individual face images.

    Face ordering follows the DirectX convention:
    - Positive X (+X): Right face
    - Negative X (-X): Left face
    - Positive Y (+Y): Top/Up face
    - Negative Y (-Y): Bottom/Down face
    - Positive Z (+Z): Front face
    - Negative Z (-Z): Back face

    The output is an uncompressed cubemap DDS that can be further processed
    with compression and mipmap generation.
    """

    def __init__(self):
        """Initializes the CubemapAssembler node."""
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        return {
            "required": {
                "pos_x": ("IMAGE", {"tooltip": "Positive X (+X) face - Right side of the cubemap."}),
                "neg_x": ("IMAGE", {"tooltip": "Negative X (-X) face - Left side of the cubemap."}),
                "pos_y": ("IMAGE", {"tooltip": "Positive Y (+Y) face - Top/Up side of the cubemap."}),
                "neg_y": ("IMAGE", {"tooltip": "Negative Y (-Y) face - Bottom/Down side of the cubemap."}),
                "pos_z": ("IMAGE", {"tooltip": "Positive Z (+Z) face - Front side of the cubemap."}),
                "neg_z": ("IMAGE", {"tooltip": "Negative Z (-Z) face - Back side of the cubemap."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "cubemap",
                        "tooltip": "Prefix for the output filename. The output will be named 'prefix.dds'.",
                    },
                ),
                "format": (
                    DDS_FORMATS,
                    {
                        "default": "BC6H_UF16",
                        "tooltip": "DDS compression format. BC6H for HDR cubemaps, BC1 for diffuse, BC7 for highest quality.",
                    },
                ),
                "output_color_space": (
                    ["sRGB", "Linear"],
                    {
                        "default": "Linear",
                        "tooltip": "Color space of the output. HDR/environment maps are typically Linear.",
                    },
                ),
            },
            "optional": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: Absolute path for output. If empty, saves to ComfyUI output directory.",
                    },
                ),
                "mipmap_levels": (
                    ["Full Chain", "None", "2", "3", "4", "5", "6", "7", "8"],
                    {
                        "default": "Full Chain",
                        "tooltip": "Mipmap levels: 'Full Chain' = all, 'None' = disabled, or specify count.",
                    },
                ),
                "mipmap_filter": (
                    ["Box", "Linear", "Cubic", "Triangle", "Point"],
                    {"default": "Box", "tooltip": "Mipmap filter. Box: fast default. Linear/Cubic: smoother."},
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "assemble_cubemap"
    OUTPUT_NODE = True
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DDS"
    DESCRIPTION = """Assembles a DDS cubemap from 6 individual face images.

Face inputs follow DirectX convention:
   • +X (Right), -X (Left)
   • +Y (Top), -Y (Bottom)
   • +Z (Front), -Z (Back)

The node creates a properly formatted DDS cubemap with optional compression and mipmaps."""

    @classmethod
    def VALIDATE_INPUTS(cls, output_path=None, **kwargs):
        """Validates inputs before execution."""
        # Check that texassemble is available
        if not texassemble_available():
            return "texassemble not available. DLL may need to be rebuilt with TEXCONV_USE_TEXASSEMBLE option."

        # Validate output path if specified
        if output_path:
            if os.path.isabs(output_path):
                parent_dir = os.path.dirname(output_path.rstrip(os.sep))
                if parent_dir and not os.path.exists(parent_dir):
                    return f"Parent directory does not exist: {parent_dir}"

        return True

    def assemble_cubemap(
        self,
        pos_x: torch.Tensor,
        neg_x: torch.Tensor,
        pos_y: torch.Tensor,
        neg_y: torch.Tensor,
        pos_z: torch.Tensor,
        neg_z: torch.Tensor,
        filename_prefix: str,
        format: str,
        output_color_space: str,
        output_path: str = "",
        mipmap_levels: str = "Full Chain",
        mipmap_filter: str = "Box",
    ):
        """
        Assembles a cubemap from 6 face images.

        Args:
            pos_x: +X face tensor (right)
            neg_x: -X face tensor (left)
            pos_y: +Y face tensor (top)
            neg_y: -Y face tensor (bottom)
            pos_z: +Z face tensor (front)
            neg_z: -Z face tensor (back)
            filename_prefix: Output filename prefix
            format: DDS compression format
            output_color_space: sRGB or Linear
            output_path: Optional custom output path
            mipmap_levels: Mipmap generation settings
            mipmap_filter: Mipmap filter algorithm

        Returns:
            Empty dict (OUTPUT_NODE)
        """
        # Get first image from each batch (cubemaps use single images per face)
        faces = [pos_x, neg_x, pos_y, neg_y, pos_z, neg_z]
        face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        # Validate all faces have same dimensions
        first_shape = faces[0].shape[1:3]  # H, W
        for _, (face, name) in enumerate(zip(faces, face_names)):
            if face.shape[1:3] != first_shape:
                raise ValueError(
                    f"Face {name} has different dimensions {face.shape[1:3]} than +X {first_shape}. All faces must be the same size."
                )

        h, w = first_shape
        if h != w:
            log_warning(
                "CubemapAssembler", f"Face images are not square ({w}x{h}). Cubemaps typically use square faces."
            )

        log_info("CubemapAssembler", f"Assembling cubemap from 6 faces ({w}x{h})")

        # Setup output directory
        final_output_dir = output_path if output_path else self.output_dir

        if not os.path.isabs(final_output_dir):
            final_output_dir = os.path.join(self.output_dir, final_output_dir)

        os.makedirs(final_output_dir, exist_ok=True)

        # Create temp directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix="cubemap_", dir=folder_paths.get_temp_directory())

        try:
            # Phase 1: Save all 6 faces to temp dir
            face_paths = []
            is_hdr = False

            # Check for HDR content in any face
            # We assume if one face needs HDR, all should be treated as HDR
            for face in faces:
                if face.max() > 1.0:
                    is_hdr = True
                    break

            ext = ".hdr" if is_hdr else ".tga"
            face_filenames = [
                f"face_px{ext}",
                f"face_nx{ext}",
                f"face_py{ext}",
                f"face_ny{ext}",
                f"face_pz{ext}",
                f"face_nz{ext}",
            ]

            if is_hdr:
                log_info("CubemapAssembler", "HDR content detected (>1.0), using HDR intermediates")

            # Import save_hdr inside method or at top if available
            from ...core import save_hdr

            for i, (face, fname) in enumerate(zip(faces, face_filenames)):
                # Get first image from batch
                img_tensor = face[0] if len(face.shape) == 4 else face

                temp_path = os.path.join(temp_dir, fname)

                if is_hdr:
                    # Save as HDR (float32)
                    # tensor is (H, W, C), save_hdr expects (H, W, C) numpy
                    img_np = img_tensor.cpu().numpy()
                    save_hdr(img_np, temp_path)
                else:
                    # Save as TGA (LDR)
                    pil_image = tensor2pil(img_tensor)
                    pil_image.save(temp_path)

                face_paths.append(temp_path)
                log_verbose("CubemapAssembler", f"Saved face {face_names[i]} to {fname}")

            # Phase 2: Assemble cubemap with texassemble
            raw_cubemap = os.path.join(temp_dir, "cubemap_raw.dds")

            assemble_args = [
                "cube",
                "-w",
                str(w),
                "-h",
                str(h),
                "-o",
                raw_cubemap,  # Specify full output filename with .dds extension
                "-y",
                "--",
            ] + face_paths

            log_verbose("CubemapAssembler", "Calling texassemble to create cubemap...")
            result, error = texassemble_call(assemble_args, verbose=False)

            if result != 0:
                raise RuntimeError(f"texassemble failed: {error}")

            if not os.path.exists(raw_cubemap):
                raise RuntimeError("texassemble did not create output DDS file")

            log_verbose("CubemapAssembler", f"Raw cubemap created: {raw_cubemap}")

            # Phase 3: Apply compression and mipmaps with texconv
            final_dds_path = os.path.join(final_output_dir, f"{filename_prefix}.dds")

            # Handle naming conflicts
            counter = 1
            while os.path.exists(final_dds_path):
                final_dds_path = os.path.join(final_output_dir, f"{filename_prefix}_{counter:05}.dds")
                counter += 1

            # Build texconv args
            srgb_format_map = {
                "BC1_UNORM": "BC1_UNORM_SRGB",
                "BC2_UNORM": "BC2_UNORM_SRGB",
                "BC3_UNORM": "BC3_UNORM_SRGB",
                "BC7_UNORM": "BC7_UNORM_SRGB",
                "R8G8B8A8_UNORM": "R8G8B8A8_UNORM_SRGB",
            }

            texconv_flags = []
            final_format = format
            if output_color_space == "sRGB":
                texconv_flags.append("-srgbi")
                final_format = srgb_format_map.get(format, format)

            # Mipmap args
            mipmap_args = []
            if mipmap_levels == "Full Chain":
                mipmap_args.extend(["-m", "0"])
            elif mipmap_levels == "None":
                mipmap_args.extend(["-m", "1"])
            else:
                mipmap_args.extend(["-m", mipmap_levels])

            # Mipmap filter
            filter_map = {"Box": "BOX", "Linear": "LINEAR", "Cubic": "CUBIC", "Triangle": "TRIANGLE", "Point": "POINT"}
            mipmap_args.extend(["-if", filter_map.get(mipmap_filter, "BOX")])

            conv_args = (
                [
                    "-ft",
                    "dds",
                    "-f",
                    final_format,
                ]
                + texconv_flags
                + mipmap_args
                + ["-o", final_output_dir, "-y", "--", raw_cubemap]
            )

            log_verbose("CubemapAssembler", f"Applying compression ({final_format}) and mipmaps...")

            result, error = run_texconv(conv_args)
            if result != 0:
                raise RuntimeError(f"texconv failed: {error}")

            # Rename from texconv output name to final name
            texconv_output = os.path.join(final_output_dir, os.path.splitext(os.path.basename(raw_cubemap))[0] + ".dds")
            if os.path.exists(texconv_output) and texconv_output != final_dds_path:
                if os.path.exists(final_dds_path):
                    os.remove(final_dds_path)
                os.rename(texconv_output, final_dds_path)

            log_info("CubemapAssembler", f"Cubemap saved: {final_dds_path}")

        finally:
            # Cleanup temp directory
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)

        return {}


class EquirectangularToCubemap:
    """
    Converts an equirectangular (360 panorama) image to 6 cubemap faces.

    Equirectangular images are the standard format for HDRIs and 360 photos.
    This node projects them onto a cube, outputting 6 individual face images
    that can be fed into the CubemapAssembler or used directly.

    The conversion is done via geometric projection using numpy, preserving
    HDR values when working with float32 tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Equirectangular panorama image (2:1 aspect ratio for full 360x180)."},
                ),
                "face_size": (
                    ["Auto", "128", "256", "512", "1024", "2048", "4096", "8192"],
                    {
                        "default": "1024",
                        "tooltip": "Output size for each cubemap face. Auto = width / 4.",
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

    FUNCTION = "convert"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DDS"
    DESCRIPTION = """Converts an equirectangular (360 panorama/HDRI) image to 6 cubemap faces.

Outputs 6 separate images following DirectX convention:
• +X (Right), -X (Left)
• +Y (Top), -Y (Bottom)
• +Z (Front), -Z (Back)

Feed outputs directly into Cubemap Assembler for DDS creation."""

    def convert(self, image: torch.Tensor, face_size: str):
        """
        Converts equirectangular image to 6 cubemap faces.

        Args:
            image: Equirectangular panorama tensor [B, H, W, C]
            face_size: Output size for each face ("Auto" or numeric string)

        Returns:
            Tuple of 6 face tensors
        """

        # Get first image from batch
        img_tensor = image[0] if len(image.shape) == 4 else image

        # Convert to numpy (keep float32 for HDR support)
        img_np = img_tensor.cpu().numpy()
        h, w, c = img_np.shape

        # Determine target size
        if face_size == "Auto":
            target_size = w // 4
            # Snap to nearest power of 2 for compatibility
            target_size = 2 ** int(np.round(np.log2(target_size)))
            log_info("EquirectangularToCubemap", f"Auto-calculated face size: {target_size} (from width {w})")
        else:
            target_size = int(face_size)

        log_info("EquirectangularToCubemap", f"Converting {w}x{h} panorama to {target_size}x{target_size} faces")

        # Validate aspect ratio (should be 2:1 for full 360x180)
        aspect = w / h
        if abs(aspect - 2.0) > 0.1:
            log_warning(
                "EquirectangularToCubemap",
                f"Image aspect ratio {aspect:.2f} differs from expected 2:1. Results may be distorted.",
            )

        # Generate cubemap faces
        faces = []
        face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        for face_idx, face_name in enumerate(face_names):
            face = self._render_face(img_np, face_idx, target_size)
            # Convert back to tensor [1, H, W, C]
            face_tensor = torch.from_numpy(face).unsqueeze(0)
            faces.append(face_tensor)
            log_verbose("EquirectangularToCubemap", f"Generated face {face_name}")

        log_info("EquirectangularToCubemap", "Conversion complete")
        return tuple(faces)

    def _render_face(self, equirect: np.ndarray, face_idx: int, size: int) -> np.ndarray:
        """
        Renders a single cubemap face from equirectangular image.

        Args:
            equirect: Source equirectangular image [H, W, C]
            face_idx: Face index (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
            size: Output face size

        Returns:
            Face image as numpy array [size, size, C]
        """

        h, w, c = equirect.shape

        # Create pixel coordinate grid for the output face
        # Range from -1 to 1
        u = np.linspace(-1, 1, size)
        v = np.linspace(-1, 1, size)
        u, v = np.meshgrid(u, v)

        # Convert face UV to 3D direction vectors based on face index
        # DirectX cubemap convention
        if face_idx == 0:  # +X (Right)
            x = np.ones_like(u)
            y = -v
            z = -u
        elif face_idx == 1:  # -X (Left)
            x = -np.ones_like(u)
            y = -v
            z = u
        elif face_idx == 2:  # +Y (Top)
            x = u
            y = np.ones_like(u)
            z = v
        elif face_idx == 3:  # -Y (Bottom)
            x = u
            y = -np.ones_like(u)
            z = -v
        elif face_idx == 4:  # +Z (Front)
            x = u
            y = -v
            z = np.ones_like(u)
        else:  # -Z (Back)
            x = -u
            y = -v
            z = -np.ones_like(u)

        # Convert 3D direction to spherical coordinates (theta, phi)
        # theta: longitude (-pi to pi)
        # phi: latitude (-pi/2 to pi/2)
        theta = np.arctan2(x, z)
        phi = np.arctan2(y, np.sqrt(x * x + z * z))

        # Convert spherical to equirectangular UV coordinates
        # u: 0 to 1 (left to right = -pi to pi)
        # v: 0 to 1 (top to bottom = pi/2 to -pi/2)
        eq_u = (theta / np.pi + 1) / 2  # [0, 1]
        eq_v = 0.5 - phi / np.pi  # [0, 1]

        # Convert to pixel coordinates
        px = (eq_u * (w - 1)).astype(np.float32)
        py = (eq_v * (h - 1)).astype(np.float32)

        # Bilinear interpolation
        x0 = np.floor(px).astype(np.int32)
        y0 = np.floor(py).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)

        # Wrap x coordinates for seamless horizontal tiling
        x0 = x0 % w
        x1 = x1 % w

        # Interpolation weights
        wx = px - np.floor(px)
        wy = py - np.floor(py)

        # Sample 4 corners
        c00 = equirect[y0, x0]
        c10 = equirect[y0, x1]
        c01 = equirect[y1, x0]
        c11 = equirect[y1, x1]

        # Bilinear blend
        wx = wx[:, :, np.newaxis]
        wy = wy[:, :, np.newaxis]

        result = c00 * (1 - wx) * (1 - wy) + c10 * wx * (1 - wy) + c01 * (1 - wx) * wy + c11 * wx * wy

        return result.astype(np.float32)
