#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

CACHE_ROOT="${NODEBITS_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/nodebits}"
mkdir -p "${CACHE_ROOT}/work" "${CACHE_ROOT}/pip-cache" "${CACHE_ROOT}/matplotlib"

export TMPDIR="${CACHE_ROOT}/work"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip-cache"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${NODEBITS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${NODEBITS_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${NODEBITS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NODEBITS_NUM_THREADS:-1}"

DEVICE="${NODEBITS_DEVICE:-0}"
USE_GPU="${NODEBITS_USE_GPU:-0}"

if [[ "${USE_GPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${DEVICE}"
  MEDIUM_DEVICE_ARGS=(--device 0)
else
  export CUDA_VISIBLE_DEVICES=""
  MEDIUM_DEVICE_ARGS=(--cpu)
fi

LINK_ARGS=(
  --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4
  --predp 0.05 --gnndp 0.05 --probscale 4.3 --proboffset 2.8
  --alpha 1.0 --gnnlr 0.0043 --prelr 0.0024 --batch_size 256
  --ln --lnnn --predictor incn1cn1 --dataset Cora --epochs 1
  --runs 1 --model puregcn --hiddim 16 --mplayers 1 --testbs 1024
  --maskinput --jk --use_xlin --tailact --device 0
)

MEDIUM_ARGS=(
  --gnn gcn --dataset cora --hidden_channels 16 --epochs 1 --runs 1
  --rand_split --valid_num 500 --test_num 1000 --display_step 1
)

echo "[1/4] Link prediction pretraining smoke test"
(cd "${REPO_ROOT}/link_prediction" && python main.py "${LINK_ARGS[@]}")

echo "[2/4] Link prediction readout finetune smoke test"
(cd "${REPO_ROOT}/link_prediction" && python finetune_readout.py "${LINK_ARGS[@]}")

echo "[3/4] Medium-graph node classification smoke test"
(cd "${REPO_ROOT}/node_classification/medium_graph" && python main.py "${MEDIUM_ARGS[@]}" "${MEDIUM_DEVICE_ARGS[@]}")

echo "[4/4] Medium-graph readout finetune smoke test"
(cd "${REPO_ROOT}/node_classification/medium_graph" && python finetune.py "${MEDIUM_ARGS[@]}" "${MEDIUM_DEVICE_ARGS[@]}")

echo "Quickstart completed."
