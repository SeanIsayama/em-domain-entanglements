#!/usr/bin/env bash
# Setup for vLLM readers on a Linux GPU machine (1x 80 GB GPU; WORKSPACE, default /workspace, holds venv, data and
# model). The published runs used the Runpod image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 on 1x A100 80GB.
# venv + pinned deps, then the data for the chosen datasets fetched under $EM_RAW_DIR/data (never inside the repo) and
# checked against the committed selection.
#
#   bash scripts/setup/setup_vllm.sh insecure,secure,educational   # public Betley data only: no passwords needed
#   bash scripts/setup/setup_vllm.sh all                           # needs EM_MO_PASSWORD and EM_WANG_PASSWORD
#
# Only the sources the chosen datasets need are downloaded, so a code-only machine never holds restricted data.
set -euo pipefail

DATASETS="${1:?datasets ('all' or a comma list of config.DATASETS)}"
WS="${WORKSPACE:-/workspace}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$WS/venvs/charinf"
export EM_RAW_DIR="${EM_RAW_DIR:-$WS/em_raw}"
export HF_HOME="$WS/hf"
mkdir -p "$HF_HOME" "$EM_RAW_DIR" "$WS/venvs"
case "$EM_RAW_DIR" in "$REPO"/*) echo "EM_RAW_DIR must not live inside the repo" >&2; exit 1;; esac

command -v uv >/dev/null || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements-vllm.txt"

source "$VENV/bin/activate"
python "$REPO/scripts/data/fetch_and_select.py" --datasets "$DATASETS" --fetch-mo --check-committed --no-write-results
cat <<MSG

Setup done. In each new shell:
  source $VENV/bin/activate && export HF_HOME=$HF_HOME EM_RAW_DIR=$EM_RAW_DIR
MSG
