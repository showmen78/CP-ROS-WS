"""Prepare the compiled AD-map runtime for direct Python use."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path

_BOOTSTRAPPED_INSTALL_ROOT: Path | None = None
_AD_MAP_MODULE = None

_REQUIRED_SHARED_LIBRARIES = (
    Path("PROJ4/lib/libproj.so"),
    Path("ad_physics/lib/libad_physics.so"),
    Path("ad_map_opendrive_reader/lib/libad_map_opendrive_reader.so"),
    Path("ad_map_access/lib/libad_map_access.so"),
)

_INSTALL_ROOT_ENV_KEYS = (
    "GLOBAL_PLANNER_AD_MAP_INSTALL",
    "AD_MAP_INSTALL_ROOT",
)


def resolve_ad_map_install_root(ad_map_install_root: str | Path | None = None) -> Path:
    """Resolve the AD-map install folder used by the planner runtime.

    input: optional install-root override (`str | Path | None`)
    output: resolved install root (`Path`)
    """
    candidate_paths: list[Path] = []

    if ad_map_install_root is not None:
        candidate_paths.append(Path(ad_map_install_root).expanduser().resolve())

    for environment_key in _INSTALL_ROOT_ENV_KEYS:
        value = os.environ.get(environment_key)
        if value:
            candidate_paths.append(Path(value).expanduser().resolve())

    candidate_paths.append(Path(__file__).resolve().parents[1] / "map_repo" / "install")

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Could not find the AD-map install folder. "
        "Expected one of: explicit `ad_map_install_root`, "
        "`GLOBAL_PLANNER_AD_MAP_INSTALL`, `AD_MAP_INSTALL_ROOT`, "
        "or `<project>/map_repo/install`."
    )


def _prepend_unique_path(variable_name: str, path: Path) -> None:
    """Prepend one path to an environment variable without duplicates.

    input: variable name (`str`), path (`Path`)
    output: none (`None`)
    """
    value = str(path)
    current_entries = [entry for entry in os.environ.get(variable_name, "").split(os.pathsep) if entry]
    if value in current_entries:
        current_entries.remove(value)
    current_entries.insert(0, value)
    os.environ[variable_name] = os.pathsep.join(current_entries)


def _insert_unique_sys_path(path: Path) -> None:
    """Prepend one directory to `sys.path` without duplicates.

    input: `path` (`Path`)
    output: none (`None`)
    """
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _find_python_package_paths(install_root: Path) -> list[Path]:
    """Locate Python package folders produced by the AD-map build.

    input: `install_root` (`Path`)
    output: package directories (`list[Path]`)
    """
    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    package_paths: list[Path] = []

    for package_name in ("ad_physics", "ad_map_access"):
        # AD-map 2.x installs packages directly in ``lib/pythonX.Y`` while
        # AD-map 3.x uses a conventional site-packages/dist-packages child.
        legacy_path = install_root / package_name / "lib" / version_tag
        if legacy_path.exists():
            package_paths.append(legacy_path)

        for package_folder_name in ("site-packages", "dist-packages"):
            preferred_path = install_root / package_name / "lib" / version_tag / package_folder_name
            if preferred_path.exists():
                package_paths.append(preferred_path)

        for glob_pattern in ("site-packages", "dist-packages"):
            for fallback_path in sorted((install_root / package_name / "lib").glob(f"python*/{glob_pattern}")):
                if fallback_path not in package_paths and fallback_path.exists():
                    package_paths.append(fallback_path)

    return package_paths


def _preload_shared_libraries(install_root: Path) -> None:
    """Load the compiled shared libraries before importing Python bindings.

    input: `install_root` (`Path`)
    output: none (`None`)
    """
    load_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for library_relative_path in _REQUIRED_SHARED_LIBRARIES:
        library_path = install_root / library_relative_path
        if not library_path.exists():
            raise FileNotFoundError(f"Missing AD-map shared library: {library_path}")
        _prepend_unique_path("LD_LIBRARY_PATH", library_path.parent)
        ctypes.CDLL(str(library_path), mode=load_mode)

    # AD-map 2.x installs the Boost.Python support libraries beside the C++
    # libraries rather than embedding a relative runtime search path in the
    # extension modules.  Preload them after their native dependencies so an
    # already-running Python process can import the bindings without requiring
    # callers to export LD_LIBRARY_PATH before startup.
    for package_name in ("ad_physics", "ad_map_access"):
        binding_libraries = sorted(
            (install_root / package_name / "lib").glob(
                f"lib{package_name}_python{sys.version_info.major}{sys.version_info.minor}.so*"
            )
        )
        if not binding_libraries:
            continue
        binding_library = binding_libraries[0]
        _prepend_unique_path("LD_LIBRARY_PATH", binding_library.parent)
        ctypes.CDLL(str(binding_library), mode=load_mode)


def prepare_ad_map_runtime(ad_map_install_root: str | Path | None = None) -> Path:
    """Prepare Python and shared-library paths so `ad_map_access` can be imported.

    input: optional install-root override (`str | Path | None`)
    output: resolved install root (`Path`)
    """
    global _BOOTSTRAPPED_INSTALL_ROOT

    install_root = resolve_ad_map_install_root(ad_map_install_root)
    if _BOOTSTRAPPED_INSTALL_ROOT is not None:
        if _BOOTSTRAPPED_INSTALL_ROOT != install_root:
            raise RuntimeError(
                "AD-map runtime is already prepared with a different install root: "
                f"{_BOOTSTRAPPED_INSTALL_ROOT}"
            )
        return install_root

    package_paths = _find_python_package_paths(install_root)
    if not package_paths:
        raise FileNotFoundError(
            "Could not find the AD-map Python bindings under the install root. "
            f"Checked: {install_root}"
        )

    for package_path in package_paths:
        _insert_unique_sys_path(package_path)
        _prepend_unique_path("PYTHONPATH", package_path)

    _preload_shared_libraries(install_root)
    _BOOTSTRAPPED_INSTALL_ROOT = install_root
    return install_root


def import_ad_map_access(ad_map_install_root: str | Path | None = None):
    """Prepare the runtime and import `ad_map_access`.

    input: optional install-root override (`str | Path | None`)
    output: imported AD-map module (`module`)
    """
    global _AD_MAP_MODULE

    if _AD_MAP_MODULE is not None:
        return _AD_MAP_MODULE

    prepare_ad_map_runtime(ad_map_install_root)
    _AD_MAP_MODULE = importlib.import_module("ad_map_access")
    return _AD_MAP_MODULE
