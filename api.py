"""
Custom API routes for Leputen Utils.

This module registers custom HTTP endpoints on the ComfyUI server.
"""
import io
import os

import folder_paths
from aiohttp import web
from server import PromptServer

from .core import convert_single_dds_to_pil, log_error, log_verbose

routes = PromptServer.instance.routes


@routes.get('/leputen/dds_preview')
async def dds_preview(request):
    """
    Serves a DDS file as a PNG image for browser preview.

    Query Parameters:
        filename: The relative path to the DDS file in the input directory.

    Returns:
        PNG image bytes with content-type image/png.
    """
    filename = request.rel_url.query.get('filename', '')

    if not filename:
        return web.Response(status=400, text="Missing 'filename' parameter")

    if not filename.lower().endswith('.dds'):
        return web.Response(status=400, text="Not a DDS file")

    try:
        image_path = folder_paths.get_annotated_filepath(filename)

        # Security: Ensure resolved path is within input directory
        input_dir = folder_paths.get_input_directory()
        real_path = os.path.realpath(image_path)
        real_input_dir = os.path.realpath(input_dir)
        if not real_path.startswith(real_input_dir):
            return web.Response(status=403, text="Access denied: path traversal detected")

        if not os.path.exists(image_path):
            return web.Response(status=404, text=f"File not found: {filename}")

        log_verbose("DDS Preview API", f"Converting {filename} to PNG for preview")

        # Convert DDS to PIL without alpha bleeding for quick preview
        pil_image = convert_single_dds_to_pil(image_path, alpha_bleed=False)

        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return web.Response(
            body=img_byte_arr.read(),
            content_type='image/png',
            headers={'Cache-Control': 'max-age=3600'}  # Cache for 1 hour
        )

    except Exception as e:
        log_error("DDS Preview API", f"Error converting {filename}: {e}")
        return web.Response(status=500, text=f"Error converting DDS: {str(e)}")
