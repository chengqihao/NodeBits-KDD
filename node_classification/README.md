# Node Classification

## Overview

- `medium_graph/`: experiments on medium-scale graphs.
- `large_graph/`: experiments on large-scale graphs.

## Environment

Use the unified environment from the repository root:

```bash
cd ..
bash scripts/create_env.sh nodebits
conda activate nodebits
```

The pinned package list is in `../requirements-cu118.txt`.

## Quickstart

From the repository root:

```bash
bash scripts/quickstart.sh
```

This runs the Cora medium-graph classification and readout-finetuning smoke tests. It also checks the link prediction path.

## Medium Graph Datasets

Planetoid, Amazon, Coauthor, WikiCS, and OGB-style datasets are downloaded by their PyG/OGB loaders when needed. Precomputed splits for Amazon and Coauthor are included in `medium_graph/data/`.

For Chameleon and Squirrel filtered splits:

```bash
cd ..
bash scripts/download_medium_datasets.sh
```

The script downloads:

- `medium_graph/data/geom-gcn/chameleon/chameleon_filtered.npz`
- `medium_graph/data/geom-gcn/squirrel/squirrel_filtered.npz`

## Full Medium-Graph Experiments

```bash
cd medium_graph
bash run.sh
```

## Full Large-Graph Experiments

```bash
cd large_graph
bash run.sh
```

Large-graph runs download and preprocess OGB data. On networked filesystems this preprocessing can take a while before the first training log appears.
