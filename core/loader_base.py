"""
Base classes for file loading nodes.

Provides abstract base classes for nodes that load lists of files from directories,
with support for pattern matching, parallel processing, and batch iteration.
"""

import concurrent.futures
import glob
import math
import os
from abc import ABC, abstractmethod

from comfy.utils import ProgressBar

from .utils import get_changed_hash, log_error, log_info, log_warning

# Global cache for scanned file lists to speed up iterative batch loading.
# Key: (directory, pattern, include_subfolders)
# Value: (files_list, directory_mtime)
_file_list_cache: dict[tuple, tuple[list[str], float]] = {}


def _get_dir_mtime(directory: str) -> float:
    """Get directory modification time for cache invalidation."""
    try:
        return os.path.getmtime(directory)
    except OSError:
        return 0.0


class ListLoaderBase(ABC):
    """
    Abstract base class for nodes that load a list of files from a directory.

    This class handles:
    - High-speed file scanning via os.scandir.
    - Directory-level caching with mtime-based invalidation.
    - Semicolon-separated glob pattern matching.
    - Parallel file processing using ThreadPoolExecutor.
    """

    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (
                    "STRING",
                    {
                        "default": "path/to/folder",
                        "tooltip": "The absolute path to the directory containing files to load.",
                    },
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": "*",
                        "tooltip": "Filename filter pattern using glob syntax.\n\nExamples:\n• * = all files\n• *_a.dds = files ending in _a.dds\n• texture_* = files starting with texture_\n\nMultiple patterns: Use semicolon (;) to match multiple patterns:\n• *_a.dds;*_n.dds = albedo and normal maps",
                    },
                ),
                "include_subfolders": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "If enabled, recursively scans subdirectories for matching files."},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        """Validates that the directory exists before execution."""
        if not directory or directory == "path/to/folder":
            return "Please specify a valid directory path"
        if not os.path.exists(directory):
            return f"Directory does not exist: {directory}"
        if not os.path.isdir(directory):
            return f"Path is not a directory: {directory}"
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return get_changed_hash(kwargs)

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        pass

    @abstractmethod
    def process_file(self, file_path: str, **kwargs) -> tuple | None:
        pass

    def run(self, directory: str, pattern: str, include_subfolders: bool, **kwargs):
        node_name = self.__class__.__name__
        log_info(node_name, f"Scanning directory: {directory} with pattern: {pattern}")

        if not os.path.isdir(directory):
            log_error(node_name, f"Directory '{directory}' not found.")
            raise FileNotFoundError(f"Directory '{directory}' not found.")

        files = self._get_files(directory, pattern, include_subfolders)

        if not files:
            error_msg = f"No files found in '{directory}' matching pattern '{pattern}'"
            log_error(node_name, error_msg)
            raise ValueError(error_msg)

        log_info(node_name, f"Found {len(files)} files to process.")

        results = self._process_files_parallel(files, **kwargs)

        if not results:
            error_msg = f"All files in '{directory}' failed to process"
            log_error(node_name, error_msg)
            raise ValueError(error_msg)

        # Unzip the list of tuples into separate lists
        unzipped_results = list(zip(*results))
        final_results = [list(item) for item in unzipped_results]

        log_info(node_name, f"Successfully processed {len(final_results[0])} files.")
        return tuple(final_results)

    def _get_files(self, directory: str, pattern: str, include_subfolders: bool) -> list[str]:
        """Fast file discovery using os.scandir instead of glob.

        os.scandir is 2-10x faster than glob for large directories because it
        avoids unnecessary stat calls and provides directory entry information
        directly from the OS.

        Results are cached to speed up iterative batch loading. Cache is
        invalidated when the directory's modification time changes.
        """
        global _file_list_cache
        node_name = self.__class__.__name__

        # Get extensions from supported formats (e.g., ["*.dds"] -> [".dds"])
        supported_formats = self.get_supported_formats()
        extensions = tuple(fmt.replace("*", "").lower() for fmt in supported_formats)

        # Create cache key
        cache_key = (directory, pattern, include_subfolders, extensions)
        current_mtime = _get_dir_mtime(directory)

        # Check cache
        if cache_key in _file_list_cache:
            cached_files, cached_mtime = _file_list_cache[cache_key]
            if cached_mtime == current_mtime:
                log_info(node_name, f"Using cached file list: {len(cached_files)} files")
                return cached_files
            else:
                log_info(node_name, "Directory changed, rescanning...")

        # Parse patterns (semicolon-separated, e.g., "*_a.dds;*_n.dds")
        patterns = None
        pattern_matches = {}  # Track which patterns matched files
        if pattern and pattern != "*":
            patterns = [p.strip() for p in pattern.split(";") if p.strip()]
            pattern_matches = dict.fromkeys(patterns, False)

        files = []
        dirs_scanned = [0]  # Use list for mutable closure
        last_log_count = [0]  # Track when we last logged

        def scan_directory(dir_path: str):
            """Recursively scan directory using os.scandir with progress feedback."""
            dirs_scanned[0] += 1

            try:
                with os.scandir(dir_path) as entries:
                    for entry in entries:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                # Check extension
                                if entry.name.lower().endswith(extensions):
                                    # Check pattern if specified
                                    if patterns is None:
                                        files.append(entry.path)
                                    else:
                                        for p in patterns:
                                            if glob.fnmatch.fnmatch(entry.name, p):
                                                files.append(entry.path)
                                                pattern_matches[p] = True
                                                break  # Don't add same file twice

                                    # Log progress every 1000 files (update in-place)
                                    if len(files) - last_log_count[0] >= 1000:
                                        last_log_count[0] = len(files)
                                        if include_subfolders:
                                            print(
                                                f"\rINFO: [{node_name}] Scanning... {len(files)} files found ({dirs_scanned[0]} directories)",
                                                end="",
                                                flush=True,
                                            )
                                        else:
                                            print(
                                                f"\rINFO: [{node_name}] Scanning... {len(files)} files found",
                                                end="",
                                                flush=True,
                                            )

                            elif include_subfolders and entry.is_dir(follow_symlinks=False):
                                scan_directory(entry.path)
                        except (PermissionError, OSError):
                            # Skip files/directories we can't access
                            pass
            except (PermissionError, OSError):
                # Skip directories we can't access
                pass

        scan_directory(directory)

        # Log final count if we had any progress logs (means it was a big scan)
        if last_log_count[0] > 0:
            # Clear the in-place line and print final summary with newline
            log_info(node_name, f"Scan complete: {len(files)} files found in {dirs_scanned[0]} directories")

        # Warn about patterns that didn't match any files
        if pattern_matches:
            unmatched = [p for p, matched in pattern_matches.items() if not matched]
            if unmatched:
                log_warning(node_name, f"No files matched pattern(s): {', '.join(unmatched)}")

        # Sort and cache
        sorted_files = sorted(files)
        _file_list_cache[cache_key] = (sorted_files, current_mtime)

        return sorted_files

    def _process_files_parallel(self, files: list[str], **kwargs) -> list[tuple]:
        node_name = self.__class__.__name__
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pbar = ProgressBar(len(files))

            def process_wrapper(file_path):
                return (file_path, self.process_file(file_path, **kwargs))

            futures = {executor.submit(process_wrapper, f): f for f in files}

            for future in concurrent.futures.as_completed(futures):
                original_path = futures[future]
                try:
                    _, result = future.result()
                    if result:
                        results[original_path] = result
                except Exception as e:
                    log_error(node_name, f"Failed to process file '{original_path}'. Halting batch. Error: {e}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e
                pbar.update(1)

        return [results[f] for f in files if f in results]


class IteratorLoaderBase(ListLoaderBase):
    """
    Abstract base class for nodes that iterate through a list of files in batches.
    Inherits from ListLoaderBase and adds batching logic.

    Special batch_size values:
    - batch_size <= 0: Load ALL files at once (same as the old LoadList behavior)
    - batch_size > 0: Load files in batches of the specified size
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "batch_index": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Current batch index. Connect an external iterator for automatic incrementing.",
                },
            ),
            "batch_size": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "step": 1,
                    "tooltip": "Number of files to load per batch. Set to -1 to load ALL files at once.",
                },
            ),
        })
        return inputs

    def run(self, directory: str, pattern: str, include_subfolders: bool, batch_index: int, batch_size: int, **kwargs):
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
            log_info(node_name, f"Loading all {total_files} files")
        else:
            total_batches = math.ceil(total_files / batch_size)
            current_batch_index = batch_index % total_batches

            start_index = current_batch_index * batch_size
            end_index = min(start_index + batch_size, total_files)
            batch_files = files[start_index:end_index]

            progress_text = f"Batch {current_batch_index + 1}/{total_batches} (Files {start_index + 1}-{end_index} of {total_files})"
            log_info(node_name, progress_text)

        results = self._process_files_parallel(batch_files, **kwargs)

        if not results:
            error_msg = "All files failed to process"
            log_error(node_name, error_msg)
            raise ValueError(error_msg)

        unzipped_results = list(zip(*results))
        final_results = [list(item) for item in unzipped_results]

        log_info(node_name, f"Successfully processed {len(final_results[0])} files.")

        # Return both ui data (for in-node display) and result tuple (for connections)
        return {"ui": {"progress_text": [progress_text]}, "result": tuple(final_results)}
