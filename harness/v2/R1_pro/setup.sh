#!/usr/bin/env bash
set -euo pipefail
robot_workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$robot_workspace"
# Meshes, geometry data and wheels are hosted outside git. Inside the
# repository, restore and verify them; a copied workspace already has them.
if [[ -f ../../../scripts/fetch_assets.py ]]; then
  python3 ../../../scripts/fetch_assets.py --group v2-r1pro
fi
export UV_PYTHON_INSTALL_DIR="$robot_workspace/.python"
export UV_CACHE_DIR="$robot_workspace/.runtime/setup-cache"
uv python install --no-bin 3.10.20
# Refresh paths after a manual copy or rename, preserving installed libraries.
uv venv --managed-python --python 3.10.20 --relocatable --allow-existing .venv
uv pip install --python .venv/bin/python --link-mode copy \
  --default-index "${ROBOT_PACKAGE_INDEX:-https://pypi.org/simple}" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match -r requirements.lock
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --link-mode copy \
  --no-deps --no-build-isolation \
  .dependencies/wheels/nvidia_curobo-0.0.0-cp310-cp310-linux_x86_64.whl .dependencies/sam2
.venv/bin/python tools/util/environment.py --configure
.venv/bin/python tools/util/environment.py --check
