# NodeBits

## Overview

- `node_classification/`: node classification experiments on medium-scale and large-scale graphs.
- `link_prediction/`: link prediction experiments.
- `scripts/`: environment creation, quickstart smoke tests, and dataset helpers.

## Environment

The repository now uses one CUDA 11.8 environment spec for both node classification and link prediction. The smoke tests were run with Python 3.8, PyTorch 2.1.0+cu118, PyG 2.6.1, DGL 0.9.1, OGB 1.3.6, NumPy 1.24.x, SciPy 1.10.x, pandas 2.0.x, and scikit-learn 1.3.x.

Create the environment with:

```bash
bash scripts/create_env.sh nodebits
conda activate nodebits
```

The installer writes pip, conda, and build temporaries under `NODEBITS_CACHE_DIR` when it is set, otherwise under the user's standard cache directory. On shared clusters, set `NODEBITS_CACHE_DIR` to a local scratch/cache filesystem before running the installer.

If the default conda environment location is slow or quota-limited, set `NODEBITS_ENV_PREFIX` to create the environment at a custom conda prefix:

```bash
NODEBITS_ENV_PREFIX=path/to/nodebits-env bash scripts/create_env.sh
conda activate path/to/nodebits-env
```

## Quickstart

Run a fast CPU smoke test for the paths that should complete in minutes:

```bash
bash scripts/quickstart.sh
```

This runs:

- link prediction Cora pretraining for 1 epoch;
- link prediction Cora readout finetuning for 1 epoch;
- medium-graph Cora node classification for 1 epoch;
- medium-graph Cora readout finetuning for 1 epoch.

To use GPU 0 instead of CPU:

```bash
NODEBITS_USE_GPU=1 NODEBITS_DEVICE=0 bash scripts/quickstart.sh
```

## Full Experiments

Run the original full scripts after activating the environment:

```bash
cd link_prediction
bash run.sh

cd ../node_classification/medium_graph
bash run.sh

cd ../large_graph
bash run.sh
```

The full scripts are intentionally heavy: they include large datasets, many epochs, and multiple runs. OGB dataset preprocessing can be slow on networked filesystems.

Link-prediction runs write TensorBoard logs under `./rec` by default. On slow filesystems, set `NODEBITS_LOG_DIR` to place those logs on a faster local cache filesystem.

## Data Helpers

Planetoid and OGB datasets are downloaded by PyG/OGB on demand.

For filtered Chameleon and Squirrel splits used by `node_classification/medium_graph`, run:

```bash
bash scripts/download_medium_datasets.sh
```

Pokec is downloaded by the dataset loader if `data/pokec/pokec.mat` is missing.

## 🙏 Acknowledgement

We extend our sincere appreciation to the following repositories for their code and datasets:

- [NodeID](https://github.com/LUOyk1999/NodeID)
- [tunedGNN](https://github.com/LUOyk1999/tunedGNN)
- [NeuralCommonNeighbor](https://github.com/GraphPKU/NeuralCommonNeighbor)
