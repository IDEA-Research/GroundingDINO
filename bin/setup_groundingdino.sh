#!/usr/bin/env bash
set -euo pipefail

# Create a uv-managed venv and install this GroundingDINO checkout editable.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

echo "Creating virtual environment with uv at $VENV_DIR ..."
uv venv "$VENV_DIR"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Ensuring pip, setuptools, and wheel are up to date..."
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

echo "Installing GroundingDINO as editable package from: $ROOT_DIR"
pip install -e "$ROOT_DIR"

echo "GroundingDINO installed successfully in editable mode."
