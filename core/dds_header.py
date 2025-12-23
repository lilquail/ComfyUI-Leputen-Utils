"""
Pure Python DDS header parser for ComfyUI-Leputen-Utils.

Parses DDS file headers to extract metadata without requiring external tools.
This eliminates the need for texdiag.exe subprocess calls.

Reference: https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
"""
import os
import struct
from dataclasses import dataclass
from pathlib import Path

# --- Constants ---

# DDS Magic Number
DDS_MAGIC = 0x20534444  # "DDS " in little-endian

# DDS Header Flags
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSD_DEPTH = 0x800000

# DDS Pixel Format Flags
DDPF_ALPHAPIXELS = 0x1
DDPF_ALPHA = 0x2
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40
DDPF_YUV = 0x200
DDPF_LUMINANCE = 0x20000

# DDS Caps
DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP = 0x400000

# DDS Caps2
DDSCAPS2_CUBEMAP = 0x200
DDSCAPS2_CUBEMAP_POSITIVEX = 0x400
DDSCAPS2_CUBEMAP_NEGATIVEX = 0x800
DDSCAPS2_CUBEMAP_POSITIVEY = 0x1000
DDSCAPS2_CUBEMAP_NEGATIVEY = 0x2000
DDSCAPS2_CUBEMAP_POSITIVEZ = 0x4000
DDSCAPS2_CUBEMAP_NEGATIVEZ = 0x8000
DDSCAPS2_VOLUME = 0x200000

# FourCC codes
FOURCC_DXT1 = 0x31545844  # "DXT1"
FOURCC_DXT2 = 0x32545844  # "DXT2"
FOURCC_DXT3 = 0x33545844  # "DXT3"
FOURCC_DXT4 = 0x34545844  # "DXT4"
FOURCC_DXT5 = 0x35545844  # "DXT5"
FOURCC_DX10 = 0x30315844  # "DX10"
FOURCC_BC4U = 0x55344342  # "BC4U"
FOURCC_BC4S = 0x53344342  # "BC4S"
FOURCC_BC5U = 0x55354342  # "BC5U"
FOURCC_BC5S = 0x53354342  # "BC5S"
FOURCC_ATI1 = 0x31495441  # "ATI1" (BC4)
FOURCC_ATI2 = 0x32495441  # "ATI2" (BC5)


# DXGI Format enum (subset - most common formats)
# Full list: https://docs.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
class DXGIFormat:
    UNKNOWN = 0
    R32G32B32A32_FLOAT = 2
    R32G32B32_FLOAT = 6
    R16G16B16A16_FLOAT = 10
    R16G16B16A16_UNORM = 11
    R32G32_FLOAT = 16
    R10G10B10A2_UNORM = 24
    R8G8B8A8_UNORM = 28
    R8G8B8A8_UNORM_SRGB = 29
    R16G16_FLOAT = 34
    R16G16_UNORM = 35
    R32_FLOAT = 41
    R8G8_UNORM = 49
    R16_FLOAT = 54
    R16_UNORM = 56
    R8_UNORM = 61
    BC1_UNORM = 71
    BC1_UNORM_SRGB = 72
    BC2_UNORM = 74
    BC2_UNORM_SRGB = 75
    BC3_UNORM = 77
    BC3_UNORM_SRGB = 78
    BC4_UNORM = 80
    BC4_SNORM = 81
    BC5_UNORM = 83
    BC5_SNORM = 84
    B5G6R5_UNORM = 85
    B5G5R5A1_UNORM = 86
    B8G8R8A8_UNORM = 87
    B8G8R8X8_UNORM = 88
    B8G8R8A8_UNORM_SRGB = 91
    B8G8R8X8_UNORM_SRGB = 93
    BC6H_UF16 = 95
    BC6H_SF16 = 96
    BC7_UNORM = 98
    BC7_UNORM_SRGB = 99


# Map DXGI format codes to human-readable names
DXGI_FORMAT_NAMES = {
    DXGIFormat.UNKNOWN: "UNKNOWN",
    DXGIFormat.R32G32B32A32_FLOAT: "R32G32B32A32_FLOAT",
    DXGIFormat.R32G32B32_FLOAT: "R32G32B32_FLOAT",
    DXGIFormat.R16G16B16A16_FLOAT: "R16G16B16A16_FLOAT",
    DXGIFormat.R16G16B16A16_UNORM: "R16G16B16A16_UNORM",
    DXGIFormat.R32G32_FLOAT: "R32G32_FLOAT",
    DXGIFormat.R10G10B10A2_UNORM: "R10G10B10A2_UNORM",
    DXGIFormat.R8G8B8A8_UNORM: "R8G8B8A8_UNORM",
    DXGIFormat.R8G8B8A8_UNORM_SRGB: "R8G8B8A8_UNORM_SRGB",
    DXGIFormat.R16G16_FLOAT: "R16G16_FLOAT",
    DXGIFormat.R16G16_UNORM: "R16G16_UNORM",
    DXGIFormat.R32_FLOAT: "R32_FLOAT",
    DXGIFormat.R8G8_UNORM: "R8G8_UNORM",
    DXGIFormat.R16_FLOAT: "R16_FLOAT",
    DXGIFormat.R16_UNORM: "R16_UNORM",
    DXGIFormat.R8_UNORM: "R8_UNORM",
    DXGIFormat.BC1_UNORM: "BC1_UNORM",
    DXGIFormat.BC1_UNORM_SRGB: "BC1_UNORM_SRGB",
    DXGIFormat.BC2_UNORM: "BC2_UNORM",
    DXGIFormat.BC2_UNORM_SRGB: "BC2_UNORM_SRGB",
    DXGIFormat.BC3_UNORM: "BC3_UNORM",
    DXGIFormat.BC3_UNORM_SRGB: "BC3_UNORM_SRGB",
    DXGIFormat.BC4_UNORM: "BC4_UNORM",
    DXGIFormat.BC4_SNORM: "BC4_SNORM",
    DXGIFormat.BC5_UNORM: "BC5_UNORM",
    DXGIFormat.BC5_SNORM: "BC5_SNORM",
    DXGIFormat.B5G6R5_UNORM: "B5G6R5_UNORM",
    DXGIFormat.B5G5R5A1_UNORM: "B5G5R5A1_UNORM",
    DXGIFormat.B8G8R8A8_UNORM: "B8G8R8A8_UNORM",
    DXGIFormat.B8G8R8X8_UNORM: "B8G8R8X8_UNORM",
    DXGIFormat.B8G8R8A8_UNORM_SRGB: "B8G8R8A8_UNORM_SRGB",
    DXGIFormat.B8G8R8X8_UNORM_SRGB: "B8G8R8X8_UNORM_SRGB",
    DXGIFormat.BC6H_UF16: "BC6H_UF16",
    DXGIFormat.BC6H_SF16: "BC6H_SF16",
    DXGIFormat.BC7_UNORM: "BC7_UNORM",
    DXGIFormat.BC7_UNORM_SRGB: "BC7_UNORM_SRGB",
}

# sRGB format codes
SRGB_FORMATS = {
    DXGIFormat.R8G8B8A8_UNORM_SRGB,
    DXGIFormat.BC1_UNORM_SRGB,
    DXGIFormat.BC2_UNORM_SRGB,
    DXGIFormat.BC3_UNORM_SRGB,
    DXGIFormat.B8G8R8A8_UNORM_SRGB,
    DXGIFormat.B8G8R8X8_UNORM_SRGB,
    DXGIFormat.BC7_UNORM_SRGB,
}

# FourCC to DXGI format mapping (for legacy DDS files)
FOURCC_TO_DXGI = {
    FOURCC_DXT1: DXGIFormat.BC1_UNORM,
    FOURCC_DXT2: DXGIFormat.BC2_UNORM,
    FOURCC_DXT3: DXGIFormat.BC2_UNORM,
    FOURCC_DXT4: DXGIFormat.BC3_UNORM,
    FOURCC_DXT5: DXGIFormat.BC3_UNORM,
    FOURCC_BC4U: DXGIFormat.BC4_UNORM,
    FOURCC_BC4S: DXGIFormat.BC4_SNORM,
    FOURCC_BC5U: DXGIFormat.BC5_UNORM,
    FOURCC_BC5S: DXGIFormat.BC5_SNORM,
    FOURCC_ATI1: DXGIFormat.BC4_UNORM,
    FOURCC_ATI2: DXGIFormat.BC5_UNORM,
}


# --- Data Classes ---

@dataclass
class DDSInfo:
    """Parsed DDS file header information."""
    width: int
    height: int
    depth: int
    mip_count: int
    array_size: int
    dxgi_format: int
    format_name: str
    color_space: str  # "sRGB", "Linear", or "Unknown"
    is_cubemap: bool
    is_volume: bool
    is_dx10: bool

    def __str__(self) -> str:
        return (
            f"DDSInfo({self.width}x{self.height}, {self.format_name}, "
            f"{self.color_space}, mips={self.mip_count}, array={self.array_size})"
        )


# --- Parser Class ---

class DDSHeaderParser:
    """
    Pure Python parser for DDS file headers.

    Reads and parses DDS headers to extract texture metadata without
    requiring external tools like texdiag.exe.

    Usage:
        info = DDSHeaderParser.parse("texture.dds")
        print(f"Format: {info.format_name}, Color Space: {info.color_space}")
    """

    @staticmethod
    def parse(file_path: str | Path) -> DDSInfo:
        """
        Parse a DDS file header and return metadata.

        Args:
            file_path: Path to the DDS file.

        Returns:
            DDSInfo dataclass with parsed metadata.

        Raises:
            ValueError: If the file is not a valid DDS file.
            FileNotFoundError: If the file doesn't exist.
        """
        # Normalize path to handle mixed slashes (e.g., A:\foo\bar/baz.dds)
        path_str = str(file_path).replace('/', os.sep)
        file_path = Path(path_str)

        if not file_path.exists():
            raise FileNotFoundError(f"DDS file not found: {file_path}")

        with open(file_path, "rb") as f:
            # Read magic number (4 bytes)
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != DDS_MAGIC:
                raise ValueError(f"Not a valid DDS file: {file_path} (invalid magic number)")

            # Read DDS_HEADER (124 bytes)
            header_data = f.read(124)
            if len(header_data) < 124:
                raise ValueError(f"Incomplete DDS header: {file_path}")

            # Parse header fields
            (
                size,           # DWORD dwSize (should be 124)
                flags,          # DWORD dwFlags
                height,         # DWORD dwHeight
                width,          # DWORD dwWidth
                pitch_or_size,  # DWORD dwPitchOrLinearSize
                depth,          # DWORD dwDepth
                mip_count,      # DWORD dwMipMapCount
                reserved1_0,    # First element of dwReserved1[11]
            ) = struct.unpack("<8I", header_data[:32])

            # Skip reserved1 (11 DWORDs = 44 bytes) and get pixel format at offset 76

            # Parse DDS_PIXELFORMAT (32 bytes) at offset 76 in header
            pf_data = header_data[72:104]
            (
                pf_size,        # DWORD dwSize (should be 32)
                pf_flags,       # DWORD dwFlags
                pf_fourcc,      # DWORD dwFourCC
                pf_rgb_bits,    # DWORD dwRGBBitCount
                pf_r_mask,      # DWORD dwRBitMask
                pf_g_mask,      # DWORD dwGBitMask
                pf_b_mask,      # DWORD dwBBitMask
                pf_a_mask,      # DWORD dwABitMask
            ) = struct.unpack("<8I", pf_data)

            # Parse caps (after pixel format)
            caps_data = header_data[104:120]
            caps1, caps2, caps3, caps4 = struct.unpack("<4I", caps_data)

            # Check for DX10 extended header
            is_dx10 = (pf_flags & DDPF_FOURCC) and (pf_fourcc == FOURCC_DX10)
            dxgi_format = DXGIFormat.UNKNOWN
            array_size = 1

            if is_dx10:
                # Read DDS_HEADER_DXT10 (20 bytes)
                dx10_data = f.read(20)
                if len(dx10_data) < 20:
                    raise ValueError(f"Incomplete DX10 header: {file_path}")

                (
                    dxgi_format,     # DXGI_FORMAT dxgiFormat
                    resource_dim,    # D3D10_RESOURCE_DIMENSION resourceDimension
                    misc_flag,       # UINT miscFlag
                    array_size,      # UINT arraySize
                    misc_flags2,     # UINT miscFlags2
                ) = struct.unpack("<5I", dx10_data)
            else:
                # Map legacy FourCC to DXGI format
                if pf_flags & DDPF_FOURCC:
                    dxgi_format = FOURCC_TO_DXGI.get(pf_fourcc, DXGIFormat.UNKNOWN)
                elif pf_flags & DDPF_RGB:
                    # Uncompressed RGB format - determine based on bit masks
                    if pf_rgb_bits == 32:
                        if pf_r_mask == 0x000000FF:
                            dxgi_format = DXGIFormat.R8G8B8A8_UNORM
                        elif pf_r_mask == 0x00FF0000:
                            dxgi_format = DXGIFormat.B8G8R8A8_UNORM
                    elif pf_rgb_bits == 24:
                        dxgi_format = DXGIFormat.B8G8R8X8_UNORM

            # Determine color space
            # Simple rule: formats with _SRGB suffix are sRGB, otherwise Linear
            # This matches the DXGI format naming convention
            if dxgi_format in SRGB_FORMATS:
                color_space = "sRGB"
            elif dxgi_format == DXGIFormat.UNKNOWN:
                color_space = "Unknown"
            else:
                color_space = "Linear"

            # Get format name
            format_name = DXGI_FORMAT_NAMES.get(dxgi_format, f"UNKNOWN({dxgi_format})")

            # Check texture type
            is_cubemap = bool(caps2 & DDSCAPS2_CUBEMAP)
            is_volume = bool(caps2 & DDSCAPS2_VOLUME)

            # Handle missing values
            if not (flags & DDSD_MIPMAPCOUNT):
                mip_count = 1
            if not (flags & DDSD_DEPTH):
                depth = 1

            return DDSInfo(
                width=width,
                height=height,
                depth=depth,
                mip_count=mip_count,
                array_size=array_size,
                dxgi_format=dxgi_format,
                format_name=format_name,
                color_space=color_space,
                is_cubemap=is_cubemap,
                is_volume=is_volume,
                is_dx10=is_dx10,
            )

    @staticmethod
    def get_color_space(file_path: str | Path) -> str:
        """
        Quickly get just the color space of a DDS file.

        Args:
            file_path: Path to the DDS file.

        Returns:
            "sRGB", "Linear", or "Unknown"
        """
        try:
            info = DDSHeaderParser.parse(file_path)
            return info.color_space
        except Exception:
            return "Unknown"


# --- Convenience Functions ---

def parse_dds_header(file_path: str | Path) -> DDSInfo:
    """Parse a DDS file header and return metadata."""
    return DDSHeaderParser.parse(file_path)


def get_dds_color_space(file_path: str | Path) -> str:
    """Get the color space of a DDS file ('sRGB', 'Linear', or 'Unknown')."""
    return DDSHeaderParser.get_color_space(file_path)


def get_dds_format(file_path: str | Path) -> str:
    """Get the format name of a DDS file (e.g., 'BC3_UNORM_SRGB')."""
    try:
        info = DDSHeaderParser.parse(file_path)
        return info.format_name
    except Exception:
        return "Unknown"
