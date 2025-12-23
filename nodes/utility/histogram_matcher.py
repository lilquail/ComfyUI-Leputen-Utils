"""
Histogram Matcher node - matches histograms between images.
"""
import numpy as np
import torch

from ...core import LEPUTEN_UTILS_CATEGORY


class HistogramMatcher:
    """
    Matches the histogram of a source image to that of a reference image.
    Supports grayscale, luminance-preserving, and per-channel matching.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE", {"tooltip": "The image whose histogram will be matched to the reference."}),
                "reference": ("IMAGE", {"tooltip": "The target histogram to match against. Only the first image in the batch is used."}),
                "mode": (["Grayscale", "Luminance (Preserve Color)", "Per Channel (RGB)"], {
                    "default": "Grayscale",
                    "tooltip": "Grayscale: Output is grayscale. Luminance: Match brightness, preserve colors. Per Channel: Match R/G/B independently."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_histogram"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = """Matches histogram of source to reference image.

• Grayscale: Outputs grayscale. Best for height maps.
• Luminance (Preserve Color): Matches brightness, keeps colors. Best for color grading.
• Per Channel (RGB): Matches R/G/B independently. Best for normal maps."""

    def _match_channel(self, src_np: np.ndarray, ref_np: np.ndarray) -> np.ndarray:
        """Match histogram of a single channel."""
        # Compute reference CDF
        ref_hist, _ = np.histogram(ref_np.flatten(), bins=256, range=(0, 1))
        ref_cdf = ref_hist.cumsum()
        ref_cdf_normalized = ref_cdf / ref_cdf[-1]

        # Compute source CDF
        src_hist, _ = np.histogram(src_np.flatten(), bins=256, range=(0, 1))
        src_cdf = src_hist.cumsum()
        src_cdf_normalized = src_cdf / src_cdf[-1]

        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.float32)
        for src_level in range(256):
            diff = np.abs(ref_cdf_normalized - src_cdf_normalized[src_level])
            lookup_table[src_level] = np.argmin(diff) / 255.0

        # Apply lookup table
        src_quantized = (src_np * 255).astype(np.uint8)
        return lookup_table[src_quantized]

    def match_histogram(self, source: torch.Tensor, reference: torch.Tensor, mode: str) -> tuple[torch.Tensor]:
        ref_tensor = reference[0]

        results = []
        for src_tensor in source:
            if mode == "Grayscale":
                # Convert to grayscale (luminance) and output grayscale
                src_gray = (
                    src_tensor[..., 0].cpu().numpy() * 0.299 +
                    src_tensor[..., 1].cpu().numpy() * 0.587 +
                    src_tensor[..., 2].cpu().numpy() * 0.114
                )
                ref_gray = (
                    ref_tensor[..., 0].cpu().numpy() * 0.299 +
                    ref_tensor[..., 1].cpu().numpy() * 0.587 +
                    ref_tensor[..., 2].cpu().numpy() * 0.114
                )

                matched = self._match_channel(src_gray, ref_gray)
                matched_tensor = torch.from_numpy(matched).float().to(source.device).unsqueeze(-1).repeat(1, 1, 3)

            elif mode == "Luminance (Preserve Color)":
                # Match luminance but preserve color ratios
                src_np = src_tensor.cpu().numpy()
                ref_np = ref_tensor.cpu().numpy()

                # Compute luminance
                src_lum = src_np[..., 0] * 0.299 + src_np[..., 1] * 0.587 + src_np[..., 2] * 0.114
                ref_lum = ref_np[..., 0] * 0.299 + ref_np[..., 1] * 0.587 + ref_np[..., 2] * 0.114

                # Match luminance histogram
                matched_lum = self._match_channel(src_lum, ref_lum)

                # Scale RGB by the luminance ratio
                src_lum_safe = np.maximum(src_lum, 1e-6)  # Prevent division by zero
                scale = matched_lum / src_lum_safe

                matched_rgb = np.clip(src_np * scale[..., np.newaxis], 0.0, 1.0)
                matched_tensor = torch.from_numpy(matched_rgb).float().to(source.device)

            else:  # Per Channel (RGB)
                channels = []
                for c in range(3):
                    src_ch = src_tensor[..., c].cpu().numpy()
                    ref_ch = ref_tensor[..., c].cpu().numpy()
                    matched_ch = self._match_channel(src_ch, ref_ch)
                    channels.append(torch.from_numpy(matched_ch).float())

                matched_tensor = torch.stack(channels, dim=-1).to(source.device)

            results.append(matched_tensor)

        return (torch.stack(results),)
