"""
Texconv DLL wrapper for cross-platform DDS texture conversion.

Uses matyalatte/Texconv-Custom-DLL for in-process texture conversion,
with automatic fallback to subprocess-based texconv.exe if DLL is unavailable.

Repository: https://github.com/matyalatte/Texconv-Custom-DLL
"""

import builtins
import contextlib
import ctypes
import sys
import threading
from pathlib import Path
from typing import Optional

from .utils import BIN_DIR, TEXCONV_PATH, log_info, log_warning

# --- Constants ---
_DLL_NAMES = {
    "win32": "texconv.dll",
    "darwin": "libtexconv.dylib",
    "linux": "libtexconv.so",
}


class TexconvDLL:
    """
    Python wrapper for Texconv-Custom-DLL.

    This class provides a Python interface to the texconv DLL for texture
    conversion operations. It handles:
    - Platform-specific DLL loading (Windows/macOS/Linux)
    - Per-call COM initialization on Windows (avoids STA/MTA conflicts)
    - Thread-safe conversion calls
    - Error handling via error buffer
    - Singleton pattern to ensure DLL is only loaded once

    Usage:
        texconv = get_texconv()
        result, error = texconv.convert(["-ft", "png", "-o", "outdir", "-y", "--", "input.dds"])
        if result != 0:
            print(f"Error: {error}")
    """

    _instance: Optional["TexconvDLL"] = None
    _dll: Optional[ctypes.CDLL] = None
    _available: bool = False
    _lock: threading.Lock = threading.Lock()
    _failure_count: int = 0
    _max_failures: int = 3  # Disable DLL after this many consecutive failures

    def __new__(cls):
        """Singleton pattern - only load DLL once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _get_dll_path(self) -> Path:
        """Get platform-appropriate DLL path from the bin directory."""
        bin_dir = Path(BIN_DIR)
        dll_name = _DLL_NAMES.get(sys.platform)

        if dll_name is None:
            # Fallback for other Linux-like systems
            if "linux" in sys.platform.lower():
                dll_name = "libtexconv.so"
            else:
                raise OSError(f"Unsupported platform: {sys.platform}")

        return bin_dir / dll_name

    def _initialize(self):
        """Load the texconv DLL (COM is initialized per-call, not here)."""
        dll_path = self._get_dll_path()

        if not dll_path.exists():
            log_warning("TexconvDLL", f"DLL not found at {dll_path}. Will use subprocess fallback.")
            self._available = False
            return

        try:
            self._dll = ctypes.cdll.LoadLibrary(str(dll_path))
            self._available = True
            self._lock = threading.Lock()
            log_info("TexconvDLL", f"Loaded DLL from {dll_path}")

            # Log COM status for informational purposes (but don't fail based on it)
            if sys.platform == "win32":
                result = self._dll.init_com()
                if result == 0:
                    log_info("TexconvDLL", "COM pre-check: MTA mode available")
                    # We successfully initialized COM, uninit it since we'll use per-call init
                    with contextlib.suppress(builtins.BaseException):
                        self._dll.uninit_com()
                elif result == 1:
                    log_info("TexconvDLL", "COM pre-check: Already initialized (compatible)")
                elif result == -2147417850:
                    log_info("TexconvDLL", "COM pre-check: STA mode detected, will use per-call initialization")
                else:
                    log_info("TexconvDLL", f"COM pre-check returned: {result}")

        except OSError as e:
            log_warning("TexconvDLL", f"Failed to load DLL: {e}. Will use subprocess fallback.")
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if the DLL is loaded and available."""
        return self._available and self._failure_count < self._max_failures

    def _disable_due_to_failures(self):
        """Disable DLL after too many consecutive failures."""
        log_warning(
            "TexconvDLL", f"DLL disabled after {self._failure_count} consecutive failures. Using subprocess fallback."
        )
        self._available = False

    def convert(self, args: list[str], verbose: bool = False, use_lock: bool = False) -> tuple[int, str]:
        """
        Call texconv with the given arguments.

        This method wraps the DLL's texconv() function, providing a Python-friendly
        interface for texture conversion operations. COM is initialized per-call
        to avoid STA/MTA conflicts with Qt and other UI frameworks.

        Since each call uses per-call COM initialization (init_com=True), calls
        should be thread-safe and can be made concurrently for parallel processing.

        Args:
            args: Command-line arguments for texconv (without 'texconv' prefix).
                  Example: ["-ft", "png", "-o", "outdir", "-y", "--", "input.dds"]
            verbose: If True, texconv will print info to console.
            use_lock: If True, use thread lock (for debugging thread safety issues).

        Returns:
            tuple of (return_code, error_message):
                - return_code: 0 = success, 1 = failed
                - error_message: Description of error if return_code != 0

        Raises:
            RuntimeError: If DLL is not available.
        """
        if not self.is_available:
            raise RuntimeError("Texconv DLL is not available. Use texconv_available() to check before calling.")

        def _do_convert():
            # Build argument array for ctypes
            # texconv expects wchar_t* argv[] on all platforms
            argv = [ctypes.c_wchar_p(arg) for arg in args]
            argv_array = (ctypes.c_wchar_p * len(argv))(*argv)

            # Error buffer for capturing error messages (larger buffer for detailed errors)
            err_buf = ctypes.create_unicode_buffer(4096)

            # Call DLL function:
            # int texconv(int argc, wchar_t* argv[], bool verbose,
            #             bool init_com, bool allow_slow_codec,
            #             wchar_t* err_buf, int err_buf_size)
            result = self._dll.texconv(
                len(argv),  # argc
                argv_array,  # argv
                verbose,  # verbose output
                True,  # init_com: Let DLL handle COM per-call (solves STA/MTA conflict!)
                True,  # allow_slow_codec (for BC6/BC7 CPU fallback)
                err_buf,  # error buffer
                4096,  # error buffer size
            )

            error_message = err_buf.value if result != 0 else ""
            return result, error_message

        try:
            if use_lock:
                with self._lock:
                    result, error_message = _do_convert()
            else:
                result, error_message = _do_convert()

            # Track failures for auto-disable
            if result != 0:
                self._failure_count += 1
                if self._failure_count >= self._max_failures:
                    self._disable_due_to_failures()
            else:
                self._failure_count = 0  # Reset on success

            return result, error_message

        except Exception as e:
            self._failure_count += 1
            if self._failure_count >= self._max_failures:
                self._disable_due_to_failures()
            raise RuntimeError(f"DLL call failed: {e}") from e

    def assemble(self, args: list[str], verbose: bool = False) -> tuple[int, str]:
        """
        Call texassemble with the given arguments.

        This method wraps the DLL's texassemble() function for creating cubemaps,
        volume maps, and texture arrays from individual images.

        Note: Requires DLL built with TEXCONV_USE_TEXASSEMBLE option.

        Args:
            args: Command-line arguments for texassemble (without 'texassemble' prefix).
                  Example: ["cube", "-w", "256", "-h", "256", "-o", "output.dds", "--", "face1.tga", ...]
            verbose: If True, texassemble will print info to console.

        Returns:
            tuple of (return_code, error_message):
                - return_code: 0 = success, 1 = failed
                - error_message: Description of error if return_code != 0

        Raises:
            RuntimeError: If DLL is not available or texassemble function not found.
        """
        if not self.is_available:
            raise RuntimeError("Texconv DLL is not available.")

        if not hasattr(self._dll, "texassemble"):
            raise RuntimeError(
                "texassemble function not found in DLL. DLL may need to be rebuilt with TEXCONV_USE_TEXASSEMBLE option."
            )

        # Build argument array for ctypes
        argv = [ctypes.c_wchar_p(arg) for arg in args]
        argv_array = (ctypes.c_wchar_p * len(argv))(*argv)

        # Error buffer for capturing error messages
        err_buf = ctypes.create_unicode_buffer(4096)

        try:
            # Call DLL function:
            # int texassemble(int argc, wchar_t* argv[], bool verbose,
            #                 bool init_com, wchar_t* err_buf, int err_buf_size)
            result = self._dll.texassemble(
                len(argv),  # argc
                argv_array,  # argv
                verbose,  # verbose output
                True,  # init_com: Let DLL handle COM per-call
                err_buf,  # error buffer
                4096,  # error buffer size
            )

            error_message = err_buf.value if result != 0 else ""
            return result, error_message

        except Exception as e:
            raise RuntimeError(f"texassemble DLL call failed: {e}") from e


# --- Module-level functions ---

_texconv_instance: Optional[TexconvDLL] = None


def get_texconv() -> TexconvDLL:
    """
    Get the singleton TexconvDLL instance.

    Returns:
        TexconvDLL: The singleton instance.

    Note:
        Always check texconv_available() before calling methods on the instance.
    """
    global _texconv_instance
    if _texconv_instance is None:
        _texconv_instance = TexconvDLL()
    return _texconv_instance


def texconv_available() -> bool:
    """
    Check if the texconv DLL is available.

    This function loads the DLL if not already loaded and checks if it's available.
    Use this before calling get_texconv().convert() to determine if DLL or
    subprocess fallback should be used.

    Returns:
        bool: True if DLL is loaded and available, False otherwise.
    """
    return get_texconv().is_available


def texassemble_available() -> bool:
    """
    Check if the texassemble function is available in the DLL.

    Returns:
        bool: True if DLL is loaded and texassemble function exists, False otherwise.
    """
    texconv = get_texconv()
    if not texconv.is_available:
        return False
    return hasattr(texconv._dll, "texassemble")


def run_texconv(args: list[str], verbose: bool = False) -> tuple[int, str]:
    """
    Executes texconv with the given arguments, handling fallback mechanism.

    Tries to use the TexconvDLL first. If the DLL is unavailable or disabled,
    falls back to executing texconv.exe via subprocess.

    Args:
        args: Command-line arguments for texconv (without 'texconv' prefix).
        verbose: If True, prints info to console.

    Returns:
        tuple of (return_code, error_message)
    """
    # 1. Try DLL
    if texconv_available():
        try:
            return get_texconv().convert(args, verbose)
        except Exception as e:
            log_warning("run_texconv", f"DLL call failed ({e}). Falling back to subprocess.")
            # Fallthrough to subprocess

    # 2. Fallback to subprocess
    if not Path(TEXCONV_PATH).exists():
        return 1, f"texconv executable not found at {TEXCONV_PATH}"

    import subprocess

    try:
        command = [TEXCONV_PATH] + args
        # Run subprocess (blocking)
        # Note: texconv prints to stdout/stderr. We capture it for error message.
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        if result.returncode != 0:
            return result.returncode, result.stderr.strip() or result.stdout.strip()

        return 0, ""

    except Exception as e:
        return 1, f"Subprocess failed: {e}"


def texconv_convert(args: list[str], verbose: bool = False) -> tuple[int, str]:
    """
    Convenience function to convert textures using the DLL.

    This is a shorthand for get_texconv().convert(args, verbose).

    Args:
        args: Command-line arguments for texconv.
        verbose: If True, texconv will print info to console.

    Returns:
        tuple of (return_code, error_message)

    Raises:
        RuntimeError: If DLL is not available.
    """
    return get_texconv().convert(args, verbose)


def texassemble_call(args: list[str], verbose: bool = False) -> tuple[int, str]:
    """
    Convenience function to assemble textures using the DLL.

    This is a shorthand for get_texconv().assemble(args, verbose).

    Args:
        args: Command-line arguments for texassemble.
        verbose: If True, texassemble will print info to console.

    Returns:
        tuple of (return_code, error_message)

    Raises:
        RuntimeError: If DLL is not available or texassemble not found.
    """
    return get_texconv().assemble(args, verbose)
