"""
Image Save node - saves images as PNG with metadata.
"""

import json
import os

import folder_paths
import numpy as np
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ...core import LEPUTEN_UTILS_CATEGORY, log_error, log_info, log_verbose, save_exr


class ImageSaveLeputen:
    """
    Saves a batch of image tensors to PNG files. This node provides options for
    custom output paths and filename prefixes, and embeds generation metadata
    (prompt and extra_pnginfo) into the PNG files for reproducibility.
    It also displays a preview of the saved images directly on the node.
    """

    def __init__(self):
        """Initializes the ImageSaveLeputen node, setting up output directory and type."""
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The batch of image tensors to be saved."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "A prefix for the output filenames. Files will be named 'prefix_00001.ext', 'prefix_00002.ext', etc.",
                    },
                ),
                "format": (
                    ["PNG", "JPEG", "WebP", "TIFF", "TGA", "EXR"],
                    {
                        "default": "PNG",
                        "tooltip": "Output format. PNG: lossless with metadata. JPEG: lossy, smaller. WebP: modern, good compression. TIFF: lossless. TGA: fast, uncompressed. EXR: HDR float32.",
                    },
                ),
            },
            "optional": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: Specify an absolute path for saving. If left empty, files will be saved in the default ComfyUI output directory.",
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Quality for JPEG/WebP (1-100). Higher = better quality, larger files. Ignored for PNG/TIFF.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = "Saves a batch of image tensors to PNG, JPEG, WebP, TIFF, TGA, or EXR files with custom naming and output paths."

    def save_images(
        self,
        images,
        filename_prefix="ComfyUI",
        format="PNG",
        output_path="",
        quality=95,
        prompt=None,
        extra_pnginfo=None,
    ) -> dict:
        """Saves a batch of image tensors to PNG files with parallel processing."""
        import concurrent.futures

        from comfy.utils import ProgressBar

        # Handle list inputs from INPUT_IS_LIST
        # Keep filename_prefix as list if provided as list (for per-file naming)
        prefix_list = filename_prefix if isinstance(filename_prefix, list) else [filename_prefix]
        format = format[0] if isinstance(format, list) else format
        output_path = (output_path[0] if output_path else "") if isinstance(output_path, list) else (output_path or "")
        quality = (
            (quality[0] if quality else 95) if isinstance(quality, list) else (quality if quality is not None else 95)
        )

        # Format settings
        format_ext = {"PNG": "png", "JPEG": "jpg", "WebP": "webp", "TIFF": "tiff", "TGA": "tga", "EXR": "exr"}
        ext = format_ext.get(format, "png")

        # Flatten images list
        all_images = []
        for img in images:
            if len(img.shape) == 4:
                for i in range(img.shape[0]):
                    all_images.append(img[i])
            else:
                all_images.append(img)

        # Determine output directory
        if not output_path:
            final_output_dir = self.output_dir
            output_type = "output"
        else:
            final_output_dir = output_path
            output_type = "temp"

        if not os.path.isabs(final_output_dir):
            final_output_dir = os.path.join(self.output_dir, final_output_dir)
            output_type = "output"

        os.makedirs(final_output_dir, exist_ok=True)

        log_verbose("ImageSaveLeputen", f"Saving {len(all_images)} image(s) as {format} with parallel processing.")

        # Handle hidden inputs that come as lists from INPUT_IS_LIST
        actual_prompt = prompt[0] if isinstance(prompt, list) and prompt else prompt
        actual_extra_pnginfo = extra_pnginfo[0] if isinstance(extra_pnginfo, list) and extra_pnginfo else extra_pnginfo

        # Prepare PNG metadata (only used for PNG format)
        png_metadata = None
        if format == "PNG" and not args.disable_metadata:
            png_metadata = PngInfo()
            if actual_prompt is not None:
                png_metadata.add_text("prompt", json.dumps(actual_prompt))
            if actual_extra_pnginfo is not None:
                for x in actual_extra_pnginfo:
                    png_metadata.add_text(x, json.dumps(actual_extra_pnginfo[x]))

        # Prepare jobs with pre-assigned filenames
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
            final_filepath = os.path.join(full_output_folder, f"{base_filename}.{ext}")
            counter = 1
            while os.path.exists(final_filepath) or final_filepath in used_paths:
                final_filepath = os.path.join(full_output_folder, f"{base_filename}_{counter:05}.{ext}")
                counter += 1
            used_paths.add(final_filepath)

            final_filename = os.path.basename(final_filepath)
            jobs.append({
                "index": i,
                "image": image,
                "filepath": final_filepath,
                "filename": final_filename,
                "subfolder": subfolder,
            })

        results = []
        pbar = ProgressBar(len(jobs))
        temp_dir = folder_paths.get_temp_directory()

        def save_single_image(job):
            try:
                img_np = (255.0 * job["image"].cpu().numpy()).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                # Format-specific save options
                if format == "PNG":
                    pil_img.save(job["filepath"], pnginfo=png_metadata, compress_level=4)
                elif format == "JPEG":
                    # JPEG doesn't support alpha, convert to RGB
                    if pil_img.mode == "RGBA":
                        pil_img = pil_img.convert("RGB")
                    pil_img.save(job["filepath"], quality=quality, optimize=True)
                elif format == "WebP":
                    pil_img.save(job["filepath"], quality=quality, method=4)
                elif format == "TIFF":
                    pil_img.save(job["filepath"], compression="tiff_lzw")
                elif format == "TGA":
                    pil_img.save(job["filepath"])
                elif format == "EXR":
                    # Save as HDR EXR utilizing shared utility
                    # Keep float32 values (don't multiply by 255)
                    img_float = job["image"].cpu().numpy().astype(np.float32)

                    save_exr(img_float, job["filepath"])

                    log_verbose("ImageSaveLeputen", f"Saved EXR: range=[{img_float.min():.4f}, {img_float.max():.4f}]")

                    # Create tone-mapped preview for browser
                    # Reinhard tone mapping: rgb / (1 + rgb)
                    preview_rgb = img_float / (1.0 + img_float)
                    preview_uint8 = (np.clip(preview_rgb, 0, 1) * 255).astype(np.uint8)
                    preview_pil = Image.fromarray(preview_uint8)
                    preview_filename = f"preview_{os.path.splitext(job['filename'])[0]}.png"
                    preview_pil.save(os.path.join(temp_dir, preview_filename))

                    return {"filename": preview_filename, "subfolder": "", "type": "temp"}

                if output_type == "output":
                    return {"filename": job["filename"], "subfolder": job["subfolder"], "type": output_type}
                else:
                    # For preview, save PNG in temp (browser compatibility)
                    preview_filename = f"preview_{os.path.splitext(job['filename'])[0]}.png"
                    pil_img.save(os.path.join(temp_dir, preview_filename))
                    return {"filename": preview_filename, "subfolder": "", "type": "temp"}
            except Exception as e:
                log_error("ImageSaveLeputen", f"Error saving image: {e}")
                return None

        # Parallel save - preserve order by tracking index
        results_dict = {}
        saved_count = [0]  # Mutable for closure
        total_count = len(jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {executor.submit(save_single_image, job): job["index"] for job in jobs}
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                if result:
                    results_dict[idx] = result
                saved_count[0] += 1
                pbar.update(1)
                # In-place progress update
                print(
                    f"\rINFO: [ImageSaveLeputen] Saved {saved_count[0]}/{total_count} {format} image(s)",
                    end="",
                    flush=True,
                )

        # Clear the in-place line and print final summary
        print()  # Newline after in-place updates

        # Sort results by original index
        results = [results_dict[i] for i in sorted(results_dict.keys())]

        if not results:
            log_error("ImageSaveLeputen", "No images were successfully saved.")
            return {"ui": {"images": []}}

        log_info("ImageSaveLeputen", f"Successfully saved {len(results)} {format} image(s).")
        return {"ui": {"images": results}}
