"""Process-wide runtime environment tweaks (macOS OpenMP / PyTorch + FAISS)."""

from __future__ import annotations

import os
from pathlib import Path


def _prepend_library_path(env_var: str, directory: Path) -> None:
    if not directory.is_dir():
        return
    current = os.environ.get(env_var, "")
    path_str = str(directory)
    if path_str not in current.split(os.pathsep):
        os.environ[env_var] = f"{path_str}{os.pathsep}{current}" if current else path_str


def _configure_libomp_search_path() -> None:
    """Help LightGBM find Homebrew's keg-only libomp on macOS."""
    for lib_dir in (
        Path("/opt/homebrew/opt/libomp/lib"),
        Path("/usr/local/opt/libomp/lib"),
    ):
        if (lib_dir / "libomp.dylib").exists():
            _prepend_library_path("DYLD_FALLBACK_LIBRARY_PATH", lib_dir)
            break


def configure_runtime_env() -> None:
    """Avoid OpenMP duplicate-library crashes when PyTorch and FAISS load together."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _configure_libomp_search_path()
