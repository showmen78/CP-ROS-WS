#!/usr/bin/env bash

# Build AD-map and its Python bindings for the Python used by ROS 2 Jazzy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAP_REPO_DIR="${SCRIPT_DIR}/map_repo"
SOURCE_DIR="${MAP_REPO_DIR}/source"
BUILD_DIR="${MAP_REPO_DIR}/build"
LOG_DIR="${MAP_REPO_DIR}/log"
INSTALL_DIR="${MAP_REPO_DIR}/install"
BUILD_VENV="${MAP_REPO_DIR}/build_venv"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
UPSTREAM_URL="${AD_MAP_UPSTREAM_URL:-https://github.com/carla-simulator/map.git}"
UPSTREAM_TAG="v3.0.0"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

case "${1:-}" in
  "")
    ;;
  --clean)
    # Keep the downloaded source but rebuild all generated files.
    rm -rf "${BUILD_DIR}" "${LOG_DIR}" "${INSTALL_DIR}" "${BUILD_VENV}"
    ;;
  --clean-all)
    # Remove both downloaded source and generated files.
    rm -rf "${MAP_REPO_DIR}"
    ;;
  -h|--help)
    echo "Usage: $0 [--clean|--clean-all]"
    exit 0
    ;;
  *)
    echo "Unknown option: ${1}" >&2
    exit 2
    ;;
esac

for command_name in git cmake c++ castxml; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python was not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_EXECUTABLE="$(readlink -f "$(command -v "${PYTHON_BIN}")")"

PYTHON_VERSION="$(
  "${PYTHON_EXECUTABLE}" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"

"${PYTHON_EXECUTABLE}" - <<'PY'
import sys

version = sys.version_info[:2]

if version < (3, 10) or version > (3, 13):
    raise SystemExit(
        "AD-map v3.0.0 requires Python 3.10 through Python 3.13. "
        f"The selected Python is {sys.version.split()[0]}."
    )
PY

PYTHON_INCLUDE_DIR="$(
  "${PYTHON_EXECUTABLE}" -c \
  'import sysconfig; print(sysconfig.get_path("include"))'
)"

if [[ ! -f "${PYTHON_INCLUDE_DIR}/Python.h" ]]; then
  echo "Python development header was not found:" >&2
  echo "${PYTHON_INCLUDE_DIR}/Python.h" >&2
  echo "Install python3-dev." >&2
  exit 1
fi

mkdir -p "${MAP_REPO_DIR}"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  echo "Downloading AD-map ${UPSTREAM_TAG}..."

  git clone \
    --branch "${UPSTREAM_TAG}" \
    --depth 1 \
    --recurse-submodules \
    --shallow-submodules \
    "${UPSTREAM_URL}" \
    "${SOURCE_DIR}"
else
  CURRENT_TAG="$(
    git -C "${SOURCE_DIR}" describe --tags --exact-match 2>/dev/null || true
  )"

  if [[ "${CURRENT_TAG}" != "${UPSTREAM_TAG}" ]]; then
    echo "The existing source is not AD-map ${UPSTREAM_TAG}." >&2
    echo "Run this script again with --clean-all." >&2
    exit 1
  fi

  echo "Reusing the existing AD-map ${UPSTREAM_TAG} source."

  git -C "${SOURCE_DIR}" submodule sync --recursive
  git -C "${SOURCE_DIR}" submodule update \
    --init \
    --recursive \
    --depth 1
fi

"${PYTHON_EXECUTABLE}" -m venv "${BUILD_VENV}"

"${BUILD_VENV}/bin/python" -m pip install --upgrade \
  pip \
  setuptools \
  wheel

"${BUILD_VENV}/bin/python" -m pip install \
  colcon-common-extensions \
  pygccxml \
  pyplusplus \
  unittest-xml-reporting

mkdir -p "${BUILD_DIR}" "${LOG_DIR}" "${INSTALL_DIR}"

export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"

"${BUILD_VENV}/bin/colcon" \
  --log-base "${LOG_DIR}" \
  build \
  --base-paths "${SOURCE_DIR}" \
  --build-base "${BUILD_DIR}" \
  --install-base "${INSTALL_DIR}" \
  --packages-up-to ad_map_access \
  --metas "${SOURCE_DIR}/colcon_python.meta" \
  --cmake-args \
    -DBUILD_TESTING=OFF \
    -DBUILD_PYTHON_BINDING=ON \
    "-DPYTHON_BINDING_VERSION=${PYTHON_VERSION}" \
    "-DPYTHON_EXECUTABLE:FILEPATH=${BUILD_VENV}/bin/python" \
    "-DPython3_EXECUTABLE:FILEPATH=${BUILD_VENV}/bin/python" \
    -DCMAKE_BUILD_TYPE=Release

# The generated setup file adds the native libraries and Python bindings
# to the current shell for the import test below.
set +u
source "${INSTALL_DIR}/setup.bash"
set -u

"${PYTHON_EXECUTABLE}" -c \
  "import ad_map_access; print('AD-map Python binding works with Python ${PYTHON_VERSION}')"

echo
echo "AD-map build completed."
echo "Install location: ${INSTALL_DIR}"