"""
Generate AO Map node - creates ambient occlusion from height maps.
"""

import torch
import torchvision.transforms.functional as tf

from ...core import LEPUTEN_UTILS_CATEGORY, log_verbose


class GenerateAOMap:
    """
    Generates an Ambient Occlusion (AO) map from a height map using a
    Screen-Space Ambient Occlusion (SSAO)-like algorithm adapted for 2D textures.

    This implementation is GPU-accelerated using grid_sample for efficient
    texture sampling and fully vectorized PyTorch operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_map": ("IMAGE", {"tooltip": "The input grayscale height map (white is high, black is low)."}),
                "auto_levels": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically stretch the height map's contrast to use the full 0-1 range before processing.",
                    },
                ),
                "radius_percent": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Sample radius as percentage of image size. Scales automatically with resolution.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Controls the intensity of the AO effect. Higher values result in darker shadows.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Adjusts the falloff curve. Higher = sharper shadows, Lower = softer gradients.",
                    },
                ),
                "bias": (
                    "FLOAT",
                    {
                        "default": 0.001,
                        "min": 0.0,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": "Helps prevent 'acne' artifacts on flat surfaces. Lower = more detail but more noise.",
                    },
                ),
                "num_samples": (
                    "INT",
                    {
                        "default": 24,
                        "min": 4,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Number of samples per pixel. Higher = smoother but slower. 16-32 is usually sufficient.",
                    },
                ),
                "blur_radius": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "tooltip": "Final Gaussian blur to smooth the result. Set to 0 to disable.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_ao"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/PBR"
    DESCRIPTION = "Generates an Ambient Occlusion map from a height map. GPU-accelerated for fast processing."

    def generate_ao(
        self,
        height_map: torch.Tensor,
        auto_levels: bool,
        radius_percent: float,
        strength: float,
        contrast: float,
        bias: float,
        num_samples: int,
        blur_radius: float,
    ):
        original_device = height_map.device
        compute_device = torch.device("cuda") if torch.cuda.is_available() else original_device

        with torch.no_grad():
            if compute_device != original_device:
                log_verbose("GenerateAOMap", f"Moving tensors to {compute_device} for GPU-accelerated processing.")
                height_map = height_map.to(compute_device)

            height_map_bchw = height_map.permute(0, 3, 1, 2)[:, 0:1, :, :]
            B, C, H, W = height_map_bchw.shape

            # Calculate actual radius in pixels from percentage
            image_size = min(H, W)
            sample_radius = (radius_percent / 100.0) * image_size

            log_verbose(
                "GenerateAOMap",
                f"Starting AO generation on {compute_device} with {num_samples} samples, radius {sample_radius:.1f}px ({radius_percent}% of {image_size}px).",
            )

            if auto_levels:
                log_verbose("GenerateAOMap", "Applying auto-levels to input height map.")
                min_vals = height_map_bchw.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
                max_vals = height_map_bchw.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
                range_vals = max_vals - min_vals
                range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
                height_map_bchw = (height_map_bchw - min_vals) / range_vals

            kernel, falloffs = self._generate_hemisphere_kernel_gpu(
                num_samples, sample_radius, contrast, compute_device, height_map.dtype
            )

            y_coords = torch.linspace(-1, 1, H, device=compute_device, dtype=height_map.dtype)
            x_coords = torch.linspace(-1, 1, W, device=compute_device, dtype=height_map.dtype)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            base_grid = torch.stack([grid_x, grid_y], dim=-1)
            base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

            pixel_scale_x = 2.0 / W
            pixel_scale_y = 2.0 / H

            total_occlusion = torch.zeros_like(height_map_bchw)

            for i in range(num_samples):
                offset_x = kernel[i, 0] * pixel_scale_x
                offset_y = kernel[i, 1] * pixel_scale_y

                offset_grid = base_grid.clone()
                offset_grid[..., 0] = offset_grid[..., 0] + offset_x
                offset_grid[..., 1] = offset_grid[..., 1] + offset_y

                shifted_map = torch.nn.functional.grid_sample(
                    height_map_bchw, offset_grid, mode="bilinear", padding_mode="border", align_corners=True
                )

                occlusion_sample = torch.clamp(shifted_map - height_map_bchw - bias, min=0)
                total_occlusion = total_occlusion + occlusion_sample * falloffs[i]

            ao_map = 1.0 - torch.clamp((total_occlusion / num_samples) * strength, 0.0, 1.0)

            if blur_radius > 0:
                kernel_size = int(blur_radius * 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                if kernel_size >= 3:
                    log_verbose(
                        "GenerateAOMap",
                        f"Applying final Gaussian blur with radius {blur_radius} (kernel size {kernel_size}).",
                    )
                    ao_map = tf.gaussian_blur(ao_map, kernel_size=[kernel_size, kernel_size], sigma=blur_radius)

            ao_map_bhwc = ao_map.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)

            if compute_device != original_device:
                ao_map_bhwc = ao_map_bhwc.to(original_device)

            log_verbose("GenerateAOMap", "AO generation complete.")
            return (ao_map_bhwc,)

    def _generate_hemisphere_kernel_gpu(
        self, num_samples: int, sample_radius: float, contrast: float, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)

        r = torch.rand(num_samples, generator=gen, dtype=dtype)
        theta = torch.rand(num_samples, generator=gen, dtype=dtype) * 2.0 * torch.pi

        r = r.to(device)
        theta = theta.to(device)

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        # Progressive scaling - samples closer to center are weighted more
        indices = torch.arange(num_samples, device=device, dtype=dtype)
        scale = indices / num_samples
        scale = 0.1 + 0.9 * (scale * scale)

        x = x * scale * sample_radius
        y = y * scale * sample_radius

        kernel = torch.stack([x, y], dim=-1)

        # Apply contrast to falloff curve
        distances = torch.sqrt(x**2 + y**2)
        falloffs = 1.0 - (distances / sample_radius)
        falloffs = torch.pow(torch.clamp(falloffs, min=0.0), contrast)

        return kernel, falloffs
