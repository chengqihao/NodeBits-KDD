#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-nodebits}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

CACHE_ROOT="${NODEBITS_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/nodebits}"
mkdir -p "${CACHE_ROOT}/work" "${CACHE_ROOT}/pip-cache" "${CACHE_ROOT}/conda-pkgs"

export TMPDIR="${CACHE_ROOT}/work"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip-cache"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda-pkgs"
export PYTHONDONTWRITEBYTECODE=1

if [[ -n "${NODEBITS_ENV_PREFIX:-}" ]]; then
  ENV_TARGET="${NODEBITS_ENV_PREFIX}"
  CONDA_RUN_ARGS=(-p "${NODEBITS_ENV_PREFIX}")
  mkdir -p "$(dirname "${NODEBITS_ENV_PREFIX}")"
  if [[ ! -x "${NODEBITS_ENV_PREFIX}/bin/python" ]]; then
    conda env create -p "${NODEBITS_ENV_PREFIX}" -f "${REPO_ROOT}/environment.yml"
  fi
else
  ENV_TARGET="${ENV_NAME}"
  CONDA_RUN_ARGS=(-n "${ENV_NAME}")
  conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}" || \
    conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml"
fi

conda run --no-capture-output "${CONDA_RUN_ARGS[@]}" python -m pip install -U pip setuptools wheel
conda run --no-capture-output "${CONDA_RUN_ARGS[@]}" python -m pip install -r "${REPO_ROOT}/requirements-cu118.txt"
conda run --no-capture-output "${CONDA_RUN_ARGS[@]}" python -m pip check

cat <<MSG

Environment '${ENV_TARGET}' is ready.
Activate it with:
  conda activate ${ENV_TARGET}

Cache directories used:
  TMPDIR=${TMPDIR}
  PIP_CACHE_DIR=${PIP_CACHE_DIR}
  CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}
MSG
