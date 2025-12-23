"""
GigapixelCLI node - wrapper for Topaz Gigapixel AI CLI.
"""

import os
import platform
import shutil
import subprocess
import tempfile

import folder_paths
import torch
from comfy.utils import ProgressBar
from PIL import Image

from ...core import (
    LEPUTEN_UTILS_CATEGORY,
    log_error,
    log_info,
    log_warning,
    pil2tensor,
    tensor2pil,
)


class GigapixelCLI:
    """
    A wrapper node for the Topaz Gigapixel AI command-line interface (CLI).

    This node allows users to upscale images using Topaz Gigapixel AI directly within
    ComfyUI workflows. It requires a licensed version of Gigapixel AI, specifically
    the Pro version, to grant access to the command-line interface.
    """

    MODEL_ALIASES = {
        "Art & CG": "art",
        "Lines": "lines",
        "Very Compressed": "vc",
        "High Fidelity": "hf",
        "Low Resolution": "lowres",
        "Standard": "std",
        "Text & Shapes": "text",
        "Redefine": "redefine",
    }

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types for the node."""
        if platform.system() == "Windows" or platform.system() == "Darwin":
            pass
        else:
            pass

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A batch of images to be upscaled."}),
                "model": (
                    list(cls.MODEL_ALIASES.keys()),
                    {"tooltip": "The AI model to use for the upscaling process."},
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 6.0,
                        "step": 0.1,
                        "tooltip": "The upscaling factor (e.g., 2.0 for 2x).",
                    },
                ),
                "autopilot": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Let Gigapixel AI automatically determine the best model and settings.",
                    },
                ),
            },
            "optional": {
                "denoise": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "label": "Denoise (0-100)",
                        "tooltip": "Amount of noise reduction to apply (0-100).",
                    },
                ),
                "sharpen": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "label": "Sharpen (0-100)",
                        "tooltip": "Amount of sharpening to apply (0-100).",
                    },
                ),
                "compression": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "label": "Fix Compression (0-100)",
                        "tooltip": "Amount of compression artifact fixing to apply (0-100).",
                    },
                ),
                "creativity": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 6,
                        "step": 1,
                        "label": "Redefine Creativity (1-6)",
                        "tooltip": "Creativity level for the 'Redefine' model.",
                    },
                ),
                "texture": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 6,
                        "step": 1,
                        "label": "Redefine Texture (1-6)",
                        "tooltip": "Texture level for the 'Redefine' model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "label": "Redefine Prompt",
                        "tooltip": "The prompt for the Redefine model.",
                    },
                ),
            },
            "hidden": {
                "gigapixel_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "execute_gigapixel"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Upscaling"
    DESCRIPTION = """
    Upscales a batch of images using the Topaz Gigapixel AI command-line interface.

    Requires a licensed installation of Topaz Gigapixel AI (Pro) to access the CLI.
    """

    @classmethod
    def VALIDATE_INPUTS(cls, scale: float = 2.0, **kwargs):
        """Validates inputs before execution."""
        if scale < 1.0 or scale > 6.0:
            return f"Scale must be between 1.0 and 6.0, got {scale}"
        return True

    def execute_gigapixel(
        self,
        image,
        model,
        scale,
        autopilot,
        denoise=0,
        sharpen=0,
        compression=0,
        creativity=3,
        texture=3,
        prompt="",
        gigapixel_path="",
    ):
        """Executes the Topaz Gigapixel AI command-line tool to upscale a batch of images."""
        log_info("GigapixelCLI", f"Starting Gigapixel AI upscaling for {len(image)} images.")

        denoise = denoise if denoise is not None else 0
        sharpen = sharpen if sharpen is not None else 0
        compression = compression if compression is not None else 0
        creativity = creativity if creativity is not None else 3
        texture = texture if texture is not None else 3

        # Determine executable path: setting > default locations > PATH
        executable = gigapixel_path
        if not executable or not os.path.isfile(executable):
            # Try default paths
            if platform.system() == "Windows":
                default_path = "C:\\Program Files\\Topaz Labs LLC\\Topaz Gigapixel AI\\gigapixel.exe"
            elif platform.system() == "Darwin":
                default_path = "/Applications/Topaz Gigapixel AI.app/Contents/Resources/bin/gigapixel"
            else:
                default_path = None

            if default_path and os.path.isfile(default_path):
                executable = default_path
            else:
                # Try PATH
                exe_name = "gigapixel.exe" if platform.system() == "Windows" else "gigapixel"
                if shutil.which(exe_name):
                    executable = exe_name
                else:
                    log_error(
                        "GigapixelCLI",
                        "Gigapixel AI executable not found. Set path in Settings > Leputen Utils > Upscaling.",
                    )
                    raise FileNotFoundError(
                        "Topaz Gigapixel AI executable not found. "
                        "Please set the path in Settings > Leputen Utils > Upscaling > Gigapixel."
                    )

        log_info("GigapixelCLI", f"Using executable: {executable}")
        log_info("GigapixelCLI", f"Model: {model}, Scale: {scale}, Autopilot: {autopilot}")
        if not autopilot:
            log_info("GigapixelCLI", f"Denoise: {denoise}, Sharpen: {sharpen}, Compression: {compression}")
            if model == "Redefine":
                log_info("GigapixelCLI", f"Creativity: {creativity}, Texture: {texture}, Prompt: '{prompt}'")

        run_temp_dir = tempfile.mkdtemp(prefix="gigapixel_run_", dir=folder_paths.get_temp_directory())

        try:
            results = []
            pbar = ProgressBar(len(image))
            for i, img_tensor in enumerate(image):
                input_filename = f"input_image_{i}.png"
                input_path = os.path.join(run_temp_dir, input_filename)

                pil_image = tensor2pil(img_tensor.unsqueeze(0))
                pil_image.save(input_path)
                log_info("GigapixelCLI", f"Saved input image {i + 1}/{len(image)} to {input_path}")

                output_suffix = "_gigapixel"
                cmd = [
                    executable,
                    "-i",
                    input_path,
                    "-o",
                    run_temp_dir,
                    "--suffix",
                    output_suffix,
                    "--scale",
                    str(scale),
                    "-m",
                    self.MODEL_ALIASES.get(model, model.lower()),
                ]

                if model in ["Standard", "Low Resolution", "High Fidelity"]:
                    cmd.extend(["--mv", "2"])

                if not autopilot:
                    if denoise > 0:
                        cmd.extend(["--denoise", str(denoise)])
                    if sharpen > 0:
                        cmd.extend(["--sharpen", str(sharpen)])
                    if compression > 0:
                        cmd.extend(["--compression", str(compression)])

                    if model == "Redefine":
                        cmd.extend(["--cr", str(creativity)])
                        cmd.extend(["--tx", str(texture)])
                        if prompt and prompt.strip():
                            cmd.extend(["--prompt", prompt.strip()])

                kwargs = {"stdin": subprocess.DEVNULL, "capture_output": True, "text": True}
                if platform.system() == "Windows":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

                log_info("GigapixelCLI", f"Processing image {i + 1}/{len(image)} with command: {' '.join(cmd)}")
                process = subprocess.run(cmd, **kwargs)
                log_info("GigapixelCLI", f"Gigapixel AI Stdout for image {i + 1}: {process.stdout}")
                if process.stderr:
                    log_warning("GigapixelCLI", f"Gigapixel AI Stderr for image {i + 1}: {process.stderr}")

                output_filename_base = os.path.splitext(input_filename)[0]
                output_path = os.path.join(run_temp_dir, f"{output_filename_base}{output_suffix}.png")

                # Check if output file exists (prioritize this over return code)
                # Gigapixel can crash during cleanup but still produce valid output
                if not os.path.exists(output_path):
                    # List actual files for debugging
                    actual_files = os.listdir(run_temp_dir)
                    log_error("GigapixelCLI", f"Output not found for image {i + 1}. Expected: {output_path}")
                    log_error("GigapixelCLI", f"Actual files in temp dir: {actual_files}")
                    if process.returncode != 0:
                        log_warning(
                            "GigapixelCLI",
                            f"Return code was non-zero: {process.returncode} (0x{process.returncode & 0xFFFFFFFF:08X})",
                        )
                    continue

                # Warn about non-zero return code but continue since file exists
                if process.returncode != 0:
                    log_warning(
                        "GigapixelCLI",
                        f"Image {i + 1} saved but CLI had non-zero return code: {process.returncode} (likely crashed during cleanup)",
                    )

                result_image = Image.open(output_path)
                result_image = result_image.convert("RGB")
                result_tensor = pil2tensor(result_image)
                pbar.update(1)
                results.append(result_tensor)

            if not results:
                log_error("GigapixelCLI", "Gigapixel CLI failed to produce any output files.")
                raise RuntimeError("Gigapixel CLI failed to produce any output files.")

            log_info("GigapixelCLI", f"Successfully processed {len(results)} images.")
            return (torch.cat(results, dim=0),)

        except FileNotFoundError as e:
            log_error("GigapixelCLI", f"Executable or file not found: {e}")
            raise RuntimeError(f"Gigapixel AI CLI error: {e}.") from e
        except subprocess.CalledProcessError as e:
            log_error("GigapixelCLI", f"Gigapixel AI CLI command failed with exit code {e.returncode}.")
            raise RuntimeError(f"Gigapixel AI CLI command failed. Stderr: {e.stderr}") from e
        except Exception as e:
            log_error("GigapixelCLI", f"An unexpected error occurred: {e}")
            raise RuntimeError(f"An unexpected error occurred during Gigapixel AI upscaling: {e}") from e
        finally:
            if os.path.exists(run_temp_dir):
                shutil.rmtree(run_temp_dir)
                log_info("GigapixelCLI", f"Cleaned up temporary directory: {run_temp_dir}")
