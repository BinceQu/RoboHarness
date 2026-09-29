#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODULE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$MODULE_DIR/../.." && pwd)
WORK_ROOT=${BEHAVIOR_RTABMAP_RELEASE_ROOT:-"$REPO_ROOT/work/rtabmap_native"}
DEPENDENCY_PREFIX=${BEHAVIOR_RTABMAP_DEPENDENCY_PREFIX:-}
CONDA_BIN=${CONDA_BIN:-conda}
PYTHON_BIN=${BEHAVIOR_RTABMAP_PYTHON_BIN:-python3}
PYTHON_HOME=${BEHAVIOR_RTABMAP_PYTHON_HOME:-$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')}
BUILD_JOBS=${BUILD_JOBS:-2}
DEPENDENCY_SPEC_FILE=
CURRENT_NEXT=
BUILD_ROOT=
BUILD_ROOT_OWNED=0
RELEASE_COMPLETE=0

cleanup() {
  if [[ -n "$DEPENDENCY_SPEC_FILE" ]]; then
    rm -f -- "$DEPENDENCY_SPEC_FILE"
  fi
  if [[ -n "$CURRENT_NEXT" ]]; then
    rm -f -- "$CURRENT_NEXT"
  fi
  if [[ "$BUILD_ROOT_OWNED" == "1" && -n "$BUILD_ROOT" ]]; then
    if [[ "$RELEASE_COMPLETE" == "1" ]]; then
      rm -rf -- "$BUILD_ROOT"
    else
      printf 'failed build preserved at %s\n' "$BUILD_ROOT" >&2
    fi
  fi
}
trap cleanup EXIT

if [[ -z "$DEPENDENCY_PREFIX" ]]; then
  echo "set BEHAVIOR_RTABMAP_DEPENDENCY_PREFIX to a persistent conda prefix" >&2
  exit 1
fi
DEPENDENCY_PREFIX=$(readlink -f "$DEPENDENCY_PREFIX")
case "$DEPENDENCY_PREFIX" in
  /tmp/*)
    echo "release dependency prefix must not be under /tmp: $DEPENDENCY_PREFIX" >&2
    exit 1
    ;;
esac
if [[ ! -x "$DEPENDENCY_PREFIX/bin/cmake" || ! -d "$DEPENDENCY_PREFIX/lib" ]]; then
  echo "incomplete RTAB-Map dependency prefix: $DEPENDENCY_PREFIX" >&2
  exit 1
fi
if [[ ! "$BUILD_JOBS" =~ ^[12]$ ]]; then
  echo "BUILD_JOBS must be 1 or 2" >&2
  exit 1
fi

COMMIT=$(sed -n 's/^commit=//p' "$MODULE_DIR/upstream.lock")
ARCHIVE_SHA256=$(sed -n 's/^sha256=//p' "$MODULE_DIR/upstream.lock")
ARCHIVE_SOURCE=${BEHAVIOR_RTABMAP_ARCHIVE:-"$WORK_ROOT/source-cache/rtabmap-$COMMIT.tar.gz"}
if [[ ! -f "$ARCHIVE_SOURCE" ]]; then
  echo "pinned upstream archive is missing: $ARCHIVE_SOURCE" >&2
  exit 1
fi
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE_SOURCE" \
  | sha256sum --check --status

EXPECTED_DEPENDENCY_SPEC_SHA256=${BEHAVIOR_RTABMAP_DEPENDENCY_SPEC_SHA256:-}
mkdir -p "$WORK_ROOT"
DEPENDENCY_SPEC_FILE=$(mktemp "$WORK_ROOT/.dependency-explicit.XXXXXX")
{
  printf '@EXPLICIT\n'
  env CONDA_NO_PLUGINS=true "$CONDA_BIN" --no-plugins list \
    --prefix "$DEPENDENCY_PREFIX" --explicit \
    | sed '/^#/d;/^@EXPLICIT$/d;/^[[:space:]]*$/d' \
    | LC_ALL=C sort
} > "$DEPENDENCY_SPEC_FILE"
DEPENDENCY_SPEC_SHA256=$(sha256sum "$DEPENDENCY_SPEC_FILE" | awk '{print $1}')
if [[ -n "$EXPECTED_DEPENDENCY_SPEC_SHA256" \
      && "$DEPENDENCY_SPEC_SHA256" != "$EXPECTED_DEPENDENCY_SPEC_SHA256" ]]; then
  echo "dependency package set does not match the verified release input" >&2
  echo "expected=$EXPECTED_DEPENDENCY_SPEC_SHA256" >&2
  echo "actual=$DEPENDENCY_SPEC_SHA256" >&2
  exit 1
fi

WORKER_SOURCE_SHA256=$(sha256sum "$SCRIPT_DIR/worker.cpp" | awk '{print $1}')
PATCH_SET_SHA256=$(
  find "$SCRIPT_DIR/patches" -maxdepth 1 -type f -name '*.patch' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
)
RELEASE_ID=${BEHAVIOR_RTABMAP_RELEASE_ID:-\
"querysubmap-v46-${COMMIT:0:12}-${WORKER_SOURCE_SHA256:0:12}-${PATCH_SET_SHA256:0:12}-${DEPENDENCY_SPEC_SHA256:0:12}"}
if [[ ! "$RELEASE_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid BEHAVIOR_RTABMAP_RELEASE_ID: $RELEASE_ID" >&2
  exit 1
fi

RELEASES_DIR="$WORK_ROOT/releases"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
if [[ -e "$RELEASE_DIR" || -L "$RELEASE_DIR" ]]; then
  echo "release already exists: $RELEASE_DIR" >&2
  exit 1
fi

BUILD_ROOT=${BEHAVIOR_RTABMAP_RELEASE_BUILD_ROOT:-}
if [[ -z "$BUILD_ROOT" ]]; then
  BUILD_ROOT=$(mktemp -d "/tmp/behavior-rtabmap-release-$RELEASE_ID.XXXXXX")
  BUILD_ROOT_OWNED=1
else
  mkdir -p "$BUILD_ROOT"
  BUILD_ROOT=$(readlink -f "$BUILD_ROOT")
fi
mkdir -p "$RELEASE_DIR/bin" "$BUILD_ROOT"
install -m 0644 "$ARCHIVE_SOURCE" "$BUILD_ROOT/rtabmap-$COMMIT.tar.gz"

env \
  PATH="$DEPENDENCY_PREFIX/bin:$PATH" \
  BEHAVIOR_RTABMAP_BUILD_ROOT="$BUILD_ROOT" \
  RTABMAP_PREFIX="$RELEASE_DIR/rtabmap-prefix" \
  RTABMAP_WORKER_BUILD_DIR="$BUILD_ROOT/worker-build-$COMMIT" \
  BEHAVIOR_RTABMAP_OUTPUT_DIR="$RELEASE_DIR/bin" \
  CMAKE_PREFIX_PATH="$DEPENDENCY_PREFIX" \
  RTABMAP_DEPENDENCY_LIB_DIR="$DEPENDENCY_PREFIX/lib" \
  CMAKE_BIN="$DEPENDENCY_PREFIX/bin/cmake" \
  BEHAVIOR_RTABMAP_WITH_PYTHON_GPU=1 \
  BEHAVIOR_RTABMAP_PYTHON_HOME="$PYTHON_HOME" \
  BUILD_JOBS="$BUILD_JOBS" \
  "$SCRIPT_DIR/build_worker.sh"

WORKER="$RELEASE_DIR/bin/behavior_rtabmap_worker"
CORE_LIBRARY=$(find "$RELEASE_DIR/rtabmap-prefix/lib" -maxdepth 1 -type f \
  -name 'librtabmap_core.so.*' -print | sort | tail -1)
if [[ ! -x "$WORKER" || -z "$CORE_LIBRARY" ]]; then
  echo "release build did not produce the worker and RTAB-Map core" >&2
  exit 1
fi

RUNTIME_LDD="$RELEASE_DIR/runtime-ldd.txt"
ldd "$WORKER" > "$RUNTIME_LDD"
if grep -q 'not found' "$RUNTIME_LDD"; then
  echo "release has unresolved shared-library dependencies" >&2
  exit 1
fi
if grep -q '/tmp/' "$RUNTIME_LDD" \
    || readelf -d "$WORKER" | grep -E '(RPATH|RUNPATH)' | grep -q '/tmp/' \
    || readelf -d "$CORE_LIBRARY" | grep -E '(RPATH|RUNPATH)' | grep -q '/tmp/'; then
  echo "release still depends on an ephemeral /tmp path" >&2
  exit 1
fi

install -m 0644 "$DEPENDENCY_SPEC_FILE" \
  "$RELEASE_DIR/dependency-explicit.txt"
cat > "$RELEASE_DIR/build-info.txt" <<EOF
release_id=$RELEASE_ID
rtabmap_commit=$COMMIT
rtabmap_archive_sha256=$ARCHIVE_SHA256
dependency_prefix=$DEPENDENCY_PREFIX
dependency_spec_sha256=$DEPENDENCY_SPEC_SHA256
worker_source_sha256=$WORKER_SOURCE_SHA256
patch_set_sha256=$PATCH_SET_SHA256
python_home=$PYTHON_HOME
EOF

CORE_RELATIVE=${CORE_LIBRARY#"$RELEASE_DIR/"}
(
  cd "$RELEASE_DIR"
  sha256sum \
    bin/behavior_rtabmap_worker \
    "$CORE_RELATIVE" \
    dependency-explicit.txt \
    build-info.txt \
    > manifest.sha256
  sha256sum --check --status manifest.sha256
)
touch "$RELEASE_DIR/.complete"

mkdir -p "$RELEASES_DIR"
CURRENT_NEXT="$WORK_ROOT/current.next.$$"
ln -s "releases/$RELEASE_ID" "$CURRENT_NEXT"
mv -T "$CURRENT_NEXT" "$WORK_ROOT/current"
CURRENT_NEXT=
RELEASE_COMPLETE=1

printf 'release=%s\nworker=%s\ndependency_spec_sha256=%s\n' \
  "$RELEASE_DIR" "$WORKER" "$DEPENDENCY_SPEC_SHA256"
