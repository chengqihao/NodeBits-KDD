#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
DATA_DIR="${REPO_ROOT}/node_classification/medium_graph/data/geom-gcn"

CACHE_ROOT="${NODEBITS_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/nodebits}"
mkdir -p "${CACHE_ROOT}/work" "${DATA_DIR}/chameleon" "${DATA_DIR}/squirrel"
export TMPDIR="${CACHE_ROOT}/work"
export PYTHONDONTWRITEBYTECODE=1

curl -L --fail \
  -o "${DATA_DIR}/chameleon/chameleon_filtered.npz" \
  https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/chameleon_filtered.npz

curl -L --fail \
  -o "${DATA_DIR}/squirrel/squirrel_filtered.npz" \
  https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/squirrel_filtered.npz

echo "Downloaded filtered Chameleon/Squirrel datasets to ${DATA_DIR}."
