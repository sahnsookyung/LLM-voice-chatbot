#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1
command -v uv >/dev/null || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
xcrun metal --version >/dev/null || { echo 'Install the Metal toolchain: xcodebuild -downloadComponent MetalToolchain'; exit 1; }
uv sync --python 3.12 --extra voice --extra models --group dev --locked
uv venv --python 3.12 --allow-existing .runtime/omlx
uv venv --python 3.12 --allow-existing .runtime/tts
uv pip install --python .runtime/omlx/bin/python 'https://github.com/jundot/omlx/releases/download/v0.7.1.dev1/omlx-0.7.1.dev1-cp312-cp312-macosx_15_0_universal2.whl'
uv pip install --python .runtime/tts/bin/python -r requirements-tts.txt
.venv/bin/python scripts/install_nemo.py
npm --prefix renderer ci
if [[ "${1:-}" == "--models" ]]; then .venv/bin/python scripts/download_models.py; fi
printf 'Setup finished. Build the app with ./scripts/build.sh, then run .venv/bin/python scripts/doctor.py.\n'
