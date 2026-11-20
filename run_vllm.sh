#!/usr/bin/env bash
# Reading with vLLM on the GPU machine: check the selection, read the chosen datasets with each vLLM reader (one model at a time),
# and bundle the readings. Run after scripts/setup/setup_vllm.sh and scripts/setup/verify_model.py --reader <reader>.
#
#   bash run_vllm.sh qwen_instruct                                         # all datasets
#   bash run_vllm.sh qwen_coder insecure,secure,educational
#   bash run_vllm.sh qwen_instruct,qwen_coder insecure,secure --n-examples 50 --n-samples 3
#   bash run_vllm.sh qwen_instruct all --n-examples 2 --smoke              # quick check
#
# Arguments: readers (comma list), datasets ('all' or comma list), then any scripts/generate/read.py options.
# Logs: $WORKSPACE/logs/<tag>.full.log (everything) and <tag>.log ('[count]/[time]/[ok]/[FAIL]' lines only); done
# marker <tag>.done. Then copy $EM_RAW_DIR/backup/<reader>/ to the analysis machine.
set -euo pipefail

READERS="${1:?readers (comma list of config.READERS keys)}"
DATASETS="${2:-all}"
shift $(( $# >= 2 ? 2 : 1 ))
WS="${WORKSPACE:-/workspace}"
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
source "$WS/venvs/charinf/bin/activate"
# GPU readers only: API readers (gpt41) need no GPU and run locally with scripts/generate/read.py
python -c 'import sys, config
bad = [r for r in sys.argv[1].split(",") if config.READERS.get(r, {}).get("kind") != "vllm"]
sys.exit(f"run_vllm.sh runs vLLM readers only, not {bad}; run API readers with scripts/generate/read.py" if bad else 0)' "$READERS"
export HF_HOME="$WS/hf" EM_RAW_DIR="${EM_RAW_DIR:-$WS/em_raw}"
mkdir -p "$WS/logs"
TAG="${READERS//,/+}"; SMOKE=()
for a in "$@"; do [ "$a" = "--smoke" ] && { TAG="${TAG}_smoke"; SMOKE=(--smoke); }; done
FULL="$WS/logs/$TAG.full.log"; FILT="$WS/logs/$TAG.log"
rm -f "$WS/logs/$TAG.done" "$FULL" "$FILT"
exec > >(tee -a "$FULL" | grep --line-buffered -E '^\[(count|time|ok|FAIL)\]' >> "$FILT") 2>&1
trap 'echo "[FAIL] run_vllm.sh line $LINENO"' ERR

echo "[time] start $(date -u +%FT%TZ) readers $READERS datasets $DATASETS"
python scripts/data/fetch_and_select.py --datasets "$DATASETS" --check-committed --no-write-results
python scripts/generate/read.py --readers "$READERS" --datasets "$DATASETS" "$@"
python scripts/generate/backup.py --readers "$READERS" "${SMOKE[@]}"
date -u +%FT%TZ > "$WS/logs/$TAG.done"
echo "[ok] $TAG finished $(date -u +%FT%TZ)"
