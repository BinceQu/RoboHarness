#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$MODULE_DIR/../.." && pwd)
LOCK_FILE="$MODULE_DIR/upstream.lock"
PATCH_DIR="$SCRIPT_DIR/patches"

lock_value() {
  sed -n "s/^$1=//p" "$LOCK_FILE"
}

COMMIT=$(lock_value commit)
ARCHIVE_URL=$(lock_value archive)
ARCHIVE_SHA256=$(lock_value sha256)
WORK_ROOT=${BEHAVIOR_RTABMAP_BUILD_ROOT:-"$REPO_ROOT/work/rtabmap_native"}
SOURCE_DIR=${RTABMAP_SOURCE_DIR:-"$WORK_ROOT/source-$COMMIT"}
RTABMAP_BUILD_DIR=${RTABMAP_BUILD_DIR:-"$WORK_ROOT/build-$COMMIT"}
RTABMAP_PREFIX=${RTABMAP_PREFIX:-"$WORK_ROOT/prefix-$COMMIT"}
WORKER_BUILD_DIR=${RTABMAP_WORKER_BUILD_DIR:-"$WORK_ROOT/worker-build-$COMMIT"}
OUTPUT_DIR=${BEHAVIOR_RTABMAP_OUTPUT_DIR:-"$SCRIPT_DIR/build"}
ARCHIVE_PATH="$WORK_ROOT/rtabmap-$COMMIT.tar.gz"

mkdir -p "$WORK_ROOT" "$OUTPUT_DIR"

if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
  if [[ ! -f "$ARCHIVE_PATH" ]]; then
    curl --fail --location --retry 3 "$ARCHIVE_URL" --output "$ARCHIVE_PATH"
  fi
  printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE_PATH" | sha256sum --check --status
  mkdir -p "$SOURCE_DIR"
  tar --extract --gzip --file "$ARCHIVE_PATH" --strip-components=1 --directory "$SOURCE_DIR"
fi

apply_upstream_patches() {
  local patch_file
  for patch_file in "$PATCH_DIR"/*.patch; do
    [[ -f "$patch_file" ]] || continue
    if [[ "$(basename "$patch_file")" == \
          "0007-external-metric-transform-direction.patch" ]]; then
      # Upstream uses CRLF while the earlier API patches add LF hunks.  Make
      # the two cumulative patch targets uniform before applying the semantic
      # direction fix so a fresh build is deterministic on GNU patch.
      sed -i 's/\r$//' \
        "$SOURCE_DIR/corelib/include/rtabmap/core/Rtabmap.h" \
        "$SOURCE_DIR/corelib/src/Rtabmap.cpp"
    fi
    if [[ "$(basename "$patch_file")" == \
          "0009-precomputed-occupancy-grid-precedence.patch" ]]; then
      # Memory.cpp is CRLF in the pinned archive while this small semantic
      # patch is LF. Normalize only its target immediately before applying it,
      # otherwise GNU patch rejects both hunks as a line-ending mismatch.
      sed -i 's/\r$//' "$SOURCE_DIR/corelib/src/Memory.cpp"
    fi
    if [[ "$(basename "$patch_file")" == \
          "0010-transactional-query-signature.patch" ]]; then
      # The pinned archive keeps these headers and the global-grid source in
      # CRLF. Earlier patches already normalize Rtabmap.{h,cpp} and Memory.cpp.
      # Normalize the remaining 0010 targets immediately before applying it so
      # a clean extraction has the same hunk context as the generated patch.
      sed -i 's/\r$//' \
        "$SOURCE_DIR/corelib/include/rtabmap/core/Memory.h" \
        "$SOURCE_DIR/corelib/include/rtabmap/core/global_map/OccupancyGrid.h" \
        "$SOURCE_DIR/corelib/src/global_map/OccupancyGrid.cpp"
    fi
    if patch --binary --directory="$SOURCE_DIR" --strip=1 --forward --batch \
        --dry-run --silent < "$patch_file"; then
      patch --binary --directory="$SOURCE_DIR" --strip=1 --forward --batch \
        < "$patch_file"
    elif patch --binary --directory="$SOURCE_DIR" --strip=1 --reverse --batch \
        --dry-run --silent < "$patch_file"; then
      printf 'upstream patch already applied: %s\n' "$(basename "$patch_file")"
    else
      echo "RTAB-Map source is incompatible with patch: $patch_file" >&2
      exit 1
    fi
  done
}

apply_upstream_patches

if [[ -n "${CMAKE_BIN:-}" ]]; then
  CMAKE="$CMAKE_BIN"
elif command -v cmake >/dev/null 2>&1; then
  CMAKE=$(command -v cmake)
else
  echo "cmake not found; set CMAKE_BIN" >&2
  exit 1
fi

# Conda-forge C++ libraries may require a newer libstdc++ ABI than the host
# compiler. A single dependency prefix is therefore also treated as the
# preferred compiler toolchain when it provides one.
if [[ -n "${CMAKE_PREFIX_PATH:-}" && "$CMAKE_PREFIX_PATH" != *\;* ]]; then
  if [[ -z "${CC:-}" && -x "$CMAKE_PREFIX_PATH/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="$CMAKE_PREFIX_PATH/bin/x86_64-conda-linux-gnu-gcc"
  fi
  if [[ -z "${CXX:-}" && -x "$CMAKE_PREFIX_PATH/bin/x86_64-conda-linux-gnu-g++" ]]; then
    export CXX="$CMAKE_PREFIX_PATH/bin/x86_64-conda-linux-gnu-g++"
  fi
fi

GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
fi

COMMON_PREFIX=()
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  COMMON_PREFIX=(-DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH")
fi

PYTHON_CMAKE=(-DWITH_PYTHON=OFF)
PYTHON_LIB_DIR=
if [[ "${BEHAVIOR_RTABMAP_WITH_PYTHON_GPU:-0}" == "1" ]]; then
  PYTHON_BIN=${BEHAVIOR_RTABMAP_PYTHON_BIN:-python3}
  PYTHON_HOME=${BEHAVIOR_RTABMAP_PYTHON_HOME:-$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')}
  PYTHON_BIN=${BEHAVIOR_RTABMAP_PYTHON_BIN:-"$PYTHON_HOME/bin/python"}
  PYTHON_VERSION=${BEHAVIOR_RTABMAP_PYTHON_VERSION:-$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')}
  PYTHON_INCLUDE="$PYTHON_HOME/include/python$PYTHON_VERSION"
  PYTHON_LIBRARY="$PYTHON_HOME/lib/libpython$PYTHON_VERSION.so"
  PYTHON_SITE_PACKAGES="$PYTHON_HOME/lib/python$PYTHON_VERSION/site-packages"
  PYBIND11_DIR="$PYTHON_SITE_PACKAGES/pybind11/share/cmake/pybind11"
  NUMPY_INCLUDE=$("$PYTHON_BIN" -c 'import numpy; print(numpy.get_include())')
  for required in \
    "$PYTHON_BIN" \
    "$PYTHON_INCLUDE/Python.h" \
    "$PYTHON_LIBRARY" \
    "$PYBIND11_DIR/pybind11Config.cmake" \
    "$NUMPY_INCLUDE/numpy/arrayobject.h"; do
    if [[ ! -e "$required" ]]; then
      echo "Python-backed RTAB-Map dependency missing: $required" >&2
      exit 1
    fi
  done
  "$PYTHON_BIN" -c 'import kornia, torch; assert torch.version.cuda is not None'
  PYTHON_CMAKE=(
    -DWITH_PYTHON=ON
    -DWITH_PYTHON_THREADING=OFF
    -DPython3_EXECUTABLE="$PYTHON_BIN"
    -DPython3_ROOT_DIR="$PYTHON_HOME"
    -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE"
    -DPython3_LIBRARY="$PYTHON_LIBRARY"
    -DPython3_NumPy_INCLUDE_DIR="$NUMPY_INCLUDE"
    -Dpybind11_DIR="$PYBIND11_DIR"
  )
  PYTHON_LIB_DIR="$PYTHON_HOME/lib"
fi

DEPENDENCY_LIB_DIR=${RTABMAP_DEPENDENCY_LIB_DIR:-}
if [[ -z "$DEPENDENCY_LIB_DIR" && -n "${CMAKE_PREFIX_PATH:-}" && "$CMAKE_PREFIX_PATH" != *\;* ]]; then
  DEPENDENCY_LIB_DIR="$CMAKE_PREFIX_PATH/lib"
fi
INSTALL_RPATH="$RTABMAP_PREFIX/lib"
if [[ -n "$DEPENDENCY_LIB_DIR" ]]; then
  INSTALL_RPATH="$INSTALL_RPATH;$DEPENDENCY_LIB_DIR"
fi
if [[ -n "$PYTHON_LIB_DIR" ]]; then
  INSTALL_RPATH="$INSTALL_RPATH;$PYTHON_LIB_DIR"
fi

# Keep the bundled worker's solver deterministic across environments.  The
# minimal target does not need Ceres' optional optimizer and some conda
# installations expose its imported target without the C-language OpenMP
# dependencies required to link it.
"$CMAKE" -S "$SOURCE_DIR" -B "$RTABMAP_BUILD_DIR" "${GENERATOR[@]}" \
  "${COMMON_PREFIX[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX="$RTABMAP_PREFIX" \
  -DCMAKE_BUILD_RPATH="$INSTALL_RPATH" \
  -DCMAKE_INSTALL_RPATH="$INSTALL_RPATH" \
  -DBUILD_APP=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCUMENTATION=OFF -DBUILD_TESTING=OFF -DBUILD_PERF_TESTS=OFF \
  -DWITH_QT=OFF "${PYTHON_CMAKE[@]}" -DWITH_TORCH=OFF \
  -DWITH_OPENMP=OFF -DPCL_OMP=OFF \
  -DWITH_G2O=OFF -DWITH_GTSAM=OFF -DWITH_CERES=ON -DWITH_TORO=ON \
  -DWITH_MRPT=OFF -DWITH_VERTIGO=OFF -DWITH_POINTMATCHER=OFF \
  -DWITH_PDAL=OFF -DWITH_LIBLAS=OFF -DWITH_OPEN3D=OFF -DWITH_OCTOMAP=OFF \
  -DWITH_GRIDMAP=OFF -DWITH_CPUTSDF=OFF -DWITH_OPENCHISEL=OFF \
  -DWITH_FREENECT=OFF -DWITH_FREENECT2=OFF -DWITH_K4W2=OFF -DWITH_K4A=OFF \
  -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF -DWITH_DC1394=OFF \
  -DWITH_FLYCAPTURE2=OFF -DWITH_ZED=OFF -DWITH_ZEDOC=OFF \
  -DWITH_REALSENSE=OFF -DWITH_REALSENSE_SLAM=OFF -DWITH_REALSENSE2=OFF \
  -DWITH_MYNTEYE=OFF -DWITH_DEPTHAI=OFF -DWITH_XVSDK=OFF -DWITH_ORBBEC_SDK=OFF \
  -DWITH_FOVIS=OFF -DWITH_VISO2=OFF -DWITH_DVO=OFF -DWITH_ORB_SLAM=OFF \
  -DWITH_OKVIS=OFF -DWITH_MSCKF_VIO=OFF -DWITH_VINS_FUSION=OFF \
  -DWITH_OPENVINS=OFF -DWITH_CUVSLAM=OFF -DWITH_LOAM=OFF -DWITH_FLOAM=OFF \
  -DWITH_LIOSAM=OFF -DWITH_FASTCV=OFF -DWITH_OPENGV=OFF -DWITH_APRILTAG=OFF

"$CMAKE" --build "$RTABMAP_BUILD_DIR" --parallel "${BUILD_JOBS:-2}"
"$CMAKE" --install "$RTABMAP_BUILD_DIR"

RTABMAP_CMAKE_DIR="$RTABMAP_PREFIX/lib/rtabmap-0.23"
"$CMAKE" -S "$SCRIPT_DIR" -B "$WORKER_BUILD_DIR" "${GENERATOR[@]}" \
  "${COMMON_PREFIX[@]}" \
  "${PYTHON_CMAKE[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DRTABMap_DIR="$RTABMAP_CMAKE_DIR" \
  -DCMAKE_BUILD_RPATH="$INSTALL_RPATH" \
  -DCMAKE_INSTALL_RPATH="$INSTALL_RPATH"
# The repository is commonly on NFS while the build directory is local. Clock
# skew can otherwise make Ninja consider a freshly edited worker.cpp older than
# its object file. The worker is one small translation unit, so rebuild it.
"$CMAKE" --build "$WORKER_BUILD_DIR" --clean-first --parallel "${BUILD_JOBS:-2}"

install -m 0755 "$WORKER_BUILD_DIR/behavior_rtabmap_worker" \
  "$OUTPUT_DIR/behavior_rtabmap_worker"
printf 'worker=%s\nrtabmap_commit=%s\nprefix=%s\n' \
  "$OUTPUT_DIR/behavior_rtabmap_worker" "$COMMIT" "$RTABMAP_PREFIX"
