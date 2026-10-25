#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Meshes, geometry data and wheels are hosted outside git. Inside the
# repository, restore and verify them; a copied workspace already has them.
if [[ -f ../../../scripts/fetch_assets.py ]]; then
  python3 ../../../scripts/fetch_assets.py --group v2-x5
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.runtime/setup-cache"
uv python install --no-bin 3.10.20
uv venv --managed-python --python 3.10.20 --relocatable --allow-existing .venv
uv pip install --python .venv/bin/python --link-mode copy -r requirements.txt
.venv/bin/python tools/util/environment.py
