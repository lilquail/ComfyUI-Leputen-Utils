"""
Image Iterator node - iterates through image files in batches.
"""

from ...core import (
    LEPUTEN_UTILS_CATEGORY,
    IteratorLoaderBase,
    get_changed_hash,
    process_standard_image,
)


class ImageIterator(IteratorLoaderBase):
    """
    Loads a batch of image files from a directory based on a batch index.
    This node is designed to iterate through a directory in chunks (batches),
    making it ideal for processing large numbers of files.
    Supports EXR (HDR) files with float32 values.
    """
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/Image"
    DESCRIPTION = "Iterates through and loads image files (including EXR/HDR) from a directory in batches."
    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "path")
    OUTPUT_IS_LIST = (True, True, True, True)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "alpha_bleed": ("BOOLEAN", {"default": True, "tooltip": "Enable alpha bleeding to prevent halo artifacts around transparent areas."}),
        })
        inputs["optional"] = {
            "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.5, "tooltip": "Apply blur during alpha bleeding for smoother transitions."}),
            "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Invert the output mask (white becomes black and vice-versa)."}),
            "exr_tone_map": ("BOOLEAN", {"default": False, "tooltip": "For EXR files: Apply Reinhard tone mapping. If disabled, raw HDR values are preserved."}),
        }
        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Custom change detection to exclude processing-only inputs from cache key."""
        return get_changed_hash(kwargs, {'alpha_bleed', 'blur_radius', 'invert_mask', 'exr_tone_map'})


    def get_supported_formats(self) -> list[str]:
        """Returns the list of supported image file extensions."""
        return ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff", "*.exr"]

    def process_file(self, file_path: str, **kwargs) -> tuple | None:
        """Loads and processes a single image file using shared utility."""
        return process_standard_image(
            file_path,
            alpha_bleed=kwargs.get('alpha_bleed', True),
            blur_radius=kwargs.get('blur_radius', 0.0),
            invert_mask=kwargs.get('invert_mask', False),
            exr_tone_map=kwargs.get('exr_tone_map', False),
            node_name=self.__class__.__name__
        )

