"""
DDS Saver node - saves images as DDS files using texconv.
"""

import os
import shutil
import tempfile

import folder_paths
from comfy.utils import ProgressBar

from ...core import (
    DDS_FORMATS,
    LEPUTEN_UTILS_CATEGORY,
    TEXCONV_PATH,
    log_info,
    log_verbose,
    run_texconv,
    tensor2pil,
    texconv_available,
)


class DDSSaver:
    """
    Saves a batch of image tensors as DirectDraw Surface (DDS) files using the
    `texconv` command-line tool. This node is essential for workflows that
    require DDS output, commonly used in game development and 3D applications.
    It offers fine-grained control over compression format and mipmap generation.
    """

    def __init__(self):
        """Initializes the DDSSaver node, setting up output directory and type."""
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The batch of image tensors to be saved as DDS files."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "A prefix for the output filenames. Each saved DDS file will be named 'prefix_00001.dds', 'prefix_00002.dds', etc.",
                    },
                ),
                "format": (
                    DDS_FORMATS,
                    {
                        "default": "BC3_UNORM",
                        "tooltip": "The compression format for the output DDS file. BC1 is best for opaque textures (no alpha). BC3 is ideal for textures with alpha transparency. BC7 offers the highest quality but larger file sizes. BC4/BC5 are for grayscale and normal maps.",
                    },
                ),
                "output_color_space": (
                    ["sRGB", "Linear"],
                    {
                        "default": "sRGB",
                        "tooltip": "The color space of the output file. sRGB for albedo/diffuse textures, Linear for normal maps, height maps, etc. Can connect from DDS loader.",
                    },
                ),
            },
            "optional": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: Specify an absolute path for saving. If empty, saves to ComfyUI output directory.",
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
                    {
                        "default": "Box",
                        "tooltip": "Mipmap filter. Box: fast default. Linear/Cubic: smoother. Triangle: good for alpha. Point: nearest.",
                    },
                ),
                "preserve_alpha_coverage": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Preserve alpha coverage during mipmapping. 0 = off. Use ~0.5 for foliage/hair.",
                    },
                ),
                "gpu_adapter": (
                    ["Auto (GPU 0)", "GPU 1", "GPU 2", "CPU Only"],
                    {
                        "default": "Auto (GPU 0)",
                        "tooltip": "GPU for BC6/BC7 compression. Use 'GPU 1' if discrete GPU isn't adapter 0.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_dds"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True  # Accept all images at once from list loaders
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DDS"
    DESCRIPTION = """
    Saves a batch of image tensors to PNG, JPEG, WebP, TIFF, TGA, or EXR files with custom naming and output paths.
    
    Supports parallel saving for speed and preserves 32-bit float data for EXR/HDR workflows.
    """

    @classmethod
    def VALIDATE_INPUTS(cls, output_path=None, output_color_space=None, **kwargs):
        """Validates inputs before execution."""
        # Handle list/tuple inputs from INPUT_IS_LIST
        if isinstance(output_path, (list, tuple)):
            output_path = output_path[0] if output_path else ""

        # Default values
        output_path = output_path or ""

        # Check that texconv is available (DLL or EXE)
        if not texconv_available() and not os.path.exists(TEXCONV_PATH):
            return "Texconv not found. Please install texconv.dll or texconv.exe in the bin folder."

        # Validate output path if specified
        if output_path:
            # Check if parent directory exists for absolute paths
            if os.path.isabs(output_path):
                parent_dir = os.path.dirname(output_path.rstrip(os.sep))
                if parent_dir and not os.path.exists(parent_dir):
                    return f"Parent directory does not exist: {parent_dir}"

        # Validate color space (handle single values, lists, tuples, and None)
        if output_color_space:
            # Unwrap from list/tuple
            if isinstance(output_color_space, (list, tuple)):
                cs_to_check = output_color_space[0] if output_color_space else "sRGB"
            else:
                cs_to_check = output_color_space

            # Skip validation if None or empty
            if cs_to_check is None or cs_to_check == "":
                pass  # Allow None/empty, will use default in save_dds
            elif cs_to_check not in ("sRGB", "Linear"):
                return f"Invalid output_color_space: '{cs_to_check}'. Must be 'sRGB' or 'Linear'"

        return True

    def save_dds(
        self,
        images,
        filename_prefix,
        format,
        output_color_space,
        output_path=None,
        mipmap_levels=None,
        mipmap_filter=None,
        gpu_adapter=None,
        preserve_alpha_coverage=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        """Saves images as DDS files with parallel processing for speed."""
        import concurrent.futures

        # Handle list inputs from INPUT_IS_LIST
        # Keep filename_prefix as list if provided as list (for per-file naming)
        prefix_list = filename_prefix if isinstance(filename_prefix, list) else [filename_prefix]
        format = format[0] if isinstance(format, list) else format
        # Keep output_color_space as list if provided as list (for per-file color space)
        color_space_list = (
            output_color_space if isinstance(output_color_space, list) else [output_color_space or "sRGB"]
        )
        output_path = (output_path[0] if output_path else "") if isinstance(output_path, list) else (output_path or "")
        mipmap_levels = (
            (mipmap_levels[0] if mipmap_levels else "Full Chain")
            if isinstance(mipmap_levels, list)
            else (mipmap_levels or "Full Chain")
        )
        mipmap_filter = (
            (mipmap_filter[0] if mipmap_filter else "Box")
            if isinstance(mipmap_filter, list)
            else (mipmap_filter or "Box")
        )
        gpu_adapter = (
            (gpu_adapter[0] if gpu_adapter else "Auto (GPU 0)")
            if isinstance(gpu_adapter, list)
            else (gpu_adapter or "Auto (GPU 0)")
        )
        preserve_alpha_coverage = (
            (preserve_alpha_coverage[0] if preserve_alpha_coverage else 0.0)
            if isinstance(preserve_alpha_coverage, list)
            else (preserve_alpha_coverage if preserve_alpha_coverage is not None else 0.0)
        )

        # Flatten images list
        all_images = []
        for img in images:
            if len(img.shape) == 4:
                for i in range(img.shape[0]):
                    all_images.append(img[i])
            else:
                all_images.append(img)

        # Default color space for files beyond the list
        default_color_space = color_space_list[0] if color_space_list else "sRGB"

        log_verbose("DDSSaver", f"Saving {len(all_images)} image(s) with parallel processing.")
        log_verbose("DDSSaver", f"Format: {format}, Color space: {default_color_space}, Mipmaps: {mipmap_levels}")

        # Check texconv availability (using run_texconv's logic implicitly via existence check)
        if not texconv_available() and not os.path.exists(TEXCONV_PATH) and not shutil.which(TEXCONV_PATH):
            raise FileNotFoundError("texconv not found in bin folder")

        # Setup output directory
        final_output_dir = output_path if output_path else self.output_dir

        if not os.path.isabs(final_output_dir):
            final_output_dir = os.path.join(self.output_dir, final_output_dir)

        os.makedirs(final_output_dir, exist_ok=True)

        # Format map for sRGB
        srgb_format_map = {
            "BC1_UNORM": "BC1_UNORM_SRGB",
            "BC2_UNORM": "BC2_UNORM_SRGB",
            "BC3_UNORM": "BC3_UNORM_SRGB",
            "BC7_UNORM": "BC7_UNORM_SRGB",
            "R8G8B8A8_UNORM": "R8G8B8A8_UNORM_SRGB",
            "B8G8R8A8_UNORM": "B8G8R8A8_UNORM_SRGB",
        }

        # Create shared temp directory for all TGA files
        temp_dir = tempfile.mkdtemp(prefix="dds_batch_save_", dir=folder_paths.get_temp_directory())

        try:
            # Phase 1: Prepare job data (sequential, fast)
            jobs = []
            used_paths = set()
            for i, image in enumerate(all_images):
                # Get per-file prefix or use single prefix for all
                file_prefix = prefix_list[i] if i < len(prefix_list) else prefix_list[0]

                subfolder = os.path.dirname(os.path.normpath(file_prefix))
                base_filename = os.path.basename(os.path.normpath(file_prefix))
                full_output_folder = os.path.join(final_output_dir, subfolder)
                os.makedirs(full_output_folder, exist_ok=True)

                # Find unique filename
                final_dds_path = os.path.join(full_output_folder, f"{base_filename}.dds")
                counter = 1
                while os.path.exists(final_dds_path) or final_dds_path in used_paths:
                    final_dds_path = os.path.join(full_output_folder, f"{base_filename}_{counter:05}.dds")
                    counter += 1
                used_paths.add(final_dds_path)

                # Determine color space
                file_color_space = default_color_space
                if color_space_list and i < len(color_space_list):
                    cs = color_space_list[i]
                    if cs in ("sRGB", "Linear"):
                        file_color_space = cs

                # Prepare format and flags
                texconv_flags = []
                final_format = format
                if file_color_space == "sRGB":
                    texconv_flags.append("-srgbi")
                    final_format = srgb_format_map.get(format, format)

                # TGA temp path
                temp_tga_path = os.path.join(temp_dir, f"img_{i:05}.tga")

                jobs.append({
                    "index": i,
                    "image": image,
                    "tga_path": temp_tga_path,
                    "dds_path": final_dds_path,
                    "output_folder": full_output_folder,
                    "format": final_format,
                    "flags": texconv_flags,
                })

            # Phase 2: Write all TGA files in parallel (fast, no progress bar)
            def write_tga(job):
                pil_image = tensor2pil(job["image"])
                pil_image.save(job["tga_path"])
                return job["index"]

            log_verbose("DDSSaver", "Phase 1: Writing temp TGA files in parallel...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(write_tga, job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # Raise any exceptions

            # Progress bar only for the slow DDS conversion phase
            pbar = ProgressBar(len(jobs))

            # Phase 3: Convert TGA to DDS
            # Build GPU args once
            gpu_args = []
            if gpu_adapter == "GPU 1":
                gpu_args = ["-gpu", "1"]
            elif gpu_adapter == "GPU 2":
                gpu_args = ["-gpu", "2"]
            elif gpu_adapter == "CPU Only":
                gpu_args = ["-nogpu"]

            # Build mipmap args
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

            # Alpha coverage preservation
            if preserve_alpha_coverage > 0:
                mipmap_args.extend(["-keepcoverage", str(preserve_alpha_coverage)])

            def convert_to_dds(job):
                args = ["-ft", "dds", "-f", job["format"]] + job["flags"] + ["-o", job["output_folder"], "-y"]
                args.extend(gpu_args)
                args.extend(mipmap_args)
                args.extend(["--", job["tga_path"]])

                result, error = run_texconv(args)
                if result != 0:
                    raise RuntimeError(f"texconv failed for {job['tga_path']}: {error}")

                # Rename to final path
                converted_dds = os.path.join(
                    full_output_folder, os.path.splitext(os.path.basename(job["tga_path"]))[0] + ".dds"
                )
                os.rename(converted_dds, job["dds_path"])
                return job["index"]

            log_verbose("DDSSaver", "Phase 2: Converting to DDS in parallel...")

            # Both DLL and subprocess now support parallel processing
            # DLL uses per-call COM init, so should be thread-safe
            saved_count = [0]  # Mutable for closure
            total_count = len(jobs)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(convert_to_dds, job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # Raise any exceptions
                    saved_count[0] += 1
                    pbar.update(1)
                    # In-place progress update
                    print(f"\rINFO: [DDSSaver] Saved {saved_count[0]}/{total_count} DDS file(s)", end="", flush=True)

            # Clear the in-place line and print final summary
            print()  # Newline after in-place updates
            log_info("DDSSaver", f"Successfully saved {len(jobs)} DDS file(s).")

        finally:
            # Cleanup temp directory
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)

        return {}
