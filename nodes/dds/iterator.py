"""
DDS Iterator node - iterates through DDS files in batches.

Optimized to batch-convert the current batch of DDS files to TGA in a single
temp directory before parallel processing, reducing texconv subprocess overhead.
"""

import concurrent.futures
import math
import os
import shutil
import tempfile

import folder_paths
from comfy.utils import ProgressBar

from ...core import (
    DDS_FORMATS,
    LEPUTEN_UTILS_CATEGORY,
    IteratorLoaderBase,
    convert_dds_batch_to_tga_paths,
    get_changed_hash,
    get_dds_info,
    get_file_info,
    load_png_and_process,
    log_error,
    log_info,
    log_verbose,
    normalize_dds_format,
    pil_to_comfy_tensors,
)


class DDSIterator(IteratorLoaderBase):
    """
    Loads a batch of DDS files from a directory based on a batch index.
    This node is designed to iterate through a directory in chunks (batches),
    making it ideal for processing large numbers of files.

    Performance: Uses batch DDS→TGA conversion for the current batch,
    followed by parallel TGA processing.
    """

    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/DDS"
    DESCRIPTION = "Iterates through and loads DDS files from a directory in batches."
    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", DDS_FORMATS, ["sRGB", "Linear"])
    RETURN_NAMES = ("image", "mask", "filename", "path", "format", "color_space")
    OUTPUT_IS_LIST = (True, True, True, True, True, True)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "alpha_bleed": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": "Enable alpha bleeding to prevent halo artifacts around transparent areas.",
                },
            ),
        })
        inputs["optional"] = {
            "blur_radius": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Apply blur during alpha bleeding for smoother transitions.",
                },
            ),
            "invert_mask": (
                "BOOLEAN",
                {"default": False, "tooltip": "Invert the output mask (white becomes black and vice-versa)."},
            ),
        }
        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Custom change detection to exclude processing-only inputs from cache key."""
        return get_changed_hash(kwargs, {"alpha_bleed", "blur_radius", "invert_mask"})

    def get_supported_formats(self) -> list[str]:
        return ["*.dds"]

    def run(
        self,
        directory: str,
        pattern: str,
        include_subfolders: bool,
        batch_index: int,
        batch_size: int,
        alpha_bleed: bool = True,
        blur_radius: float = 0.0,
        invert_mask: bool = False,
        **kwargs,
    ):
        """
        Optimized run method that batch-converts the current batch of DDS files to TGA,
        then processes TGAs in parallel.
        """
        node_name = self.__class__.__name__
        log_info(node_name, f"Scanning directory: {directory} with pattern: {pattern}")

        if not os.path.isdir(directory):
            log_error(node_name, f"Directory '{directory}' not found.")
            raise FileNotFoundError(f"Directory '{directory}' not found.")

        files = self._get_files(directory, pattern, include_subfolders)
        total_files = len(files)

        if total_files == 0:
            error_msg = f"No files found in '{directory}' matching pattern '{pattern}'"
            log_error(node_name, error_msg)
            raise ValueError(error_msg)

        # Handle batch_size <= 0 as "load all files"
        if batch_size <= 0:
            batch_files = files
            progress_text = f"All {total_files} files"
            log_verbose(node_name, f"Loading all {total_files} files")
        else:
            # Calculate batch boundaries
            total_batches = math.ceil(total_files / batch_size)
            current_batch_index = batch_index % total_batches
            start_index = current_batch_index * batch_size
            end_index = min(start_index + batch_size, total_files)
            batch_files = files[start_index:end_index]

            progress_text = f"Batch {current_batch_index + 1}/{total_batches} (Files {start_index + 1}-{end_index} of {total_files})"
            log_verbose(node_name, progress_text)

        # Create shared temp directory for batch TGA conversion
        temp_dir = tempfile.mkdtemp(prefix="dds_batch_iter_", dir=folder_paths.get_temp_directory())

        try:
            # Phase 1: Batch convert current batch's DDS files to TGA
            log_verbose(node_name, f"Phase 1: Batch converting {len(batch_files)} DDS files to TGA...")
            tga_paths = convert_dds_batch_to_tga_paths(batch_files, temp_dir)

            # Build mapping: tga_path -> original dds_path
            tga_to_dds = {}
            for dds_path in batch_files:
                base_filename = os.path.basename(dds_path)
                tga_filename = os.path.splitext(base_filename)[0] + ".tga"
                tga_path = os.path.join(temp_dir, tga_filename)
                if tga_path in tga_paths:
                    tga_to_dds[tga_path] = dds_path

            # Phase 2: Parallel process TGA files
            log_verbose(node_name, f"Phase 2: Processing {len(tga_paths)} TGA files in parallel...")

            pbar = ProgressBar(len(tga_paths))
            results = {}

            def process_tga(tga_path: str):
                dds_path = tga_to_dds.get(tga_path)
                if not dds_path:
                    return None

                raw_format, color_space = get_dds_info(dds_path, node_name)
                format_name = normalize_dds_format(raw_format)  # Strip _SRGB suffix
                pil_image = load_png_and_process(tga_path, alpha_bleed, blur_radius)
                image_rgb, alpha_channel = pil_to_comfy_tensors(pil_image, invert_mask)
                filename_no_ext, path = get_file_info(dds_path)

                return (image_rgb, alpha_channel, filename_no_ext, path, format_name, color_space)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_tga = {executor.submit(process_tga, tga): tga for tga in tga_paths}

                for future in concurrent.futures.as_completed(future_to_tga):
                    tga_path = future_to_tga[future]
                    try:
                        result = future.result()
                        if result:
                            results[tga_path] = result
                    except Exception as e:
                        log_error(node_name, f"Failed to process {tga_path}: {e}")
                        raise e
                    pbar.update(1)

            # Preserve original file order
            ordered_results = [results[tga] for tga in tga_paths if tga in results]

            if not ordered_results:
                error_msg = "All files failed to process"
                log_error(node_name, error_msg)
                raise ValueError(error_msg)

            # Unzip results
            unzipped = list(zip(*ordered_results))
            final_results = [list(item) for item in unzipped]

            log_info(node_name, f"Successfully processed {len(final_results[0])} DDS files.")

            # Return both ui data (for in-node display) and result tuple (for connections)
            return {"ui": {"progress_text": [progress_text]}, "result": tuple(final_results)}

        finally:
            # Cleanup temp directory
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)

    def process_file(self, file_path: str, **kwargs) -> tuple | None:
        """Not used - run() is overridden for batch optimization."""
        pass
