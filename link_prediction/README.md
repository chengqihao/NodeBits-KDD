# Datasets



```
python ogbdataset.py
```

# Code Running

The commands below match `run.sh` and use 5 runs.

### Cora

```bash
python main.py   --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 \
    --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  \
    --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor incn1cn1 \
    --dataset Cora  --epochs 100 --runs 5 --model puregcn --hiddim 256 --mplayers 1  \
    --testbs 8192  --maskinput  --jk  --use_xlin  --tailact 
    
python finetune_readout.py   --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 \
    --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  \
    --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor incn1cn1 \
    --dataset Cora  --epochs 100 --runs 5 --model puregcn --hiddim 256 --mplayers 1  \
    --testbs 8192  --maskinput  --jk  --use_xlin  --tailact 
```

### Citeseer

```bash
python main.py   --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 \
    --gnndp 0.75  --probscale 6.5 --proboffset 4.4 --alpha 0.4  \
    --gnnlr 0.0085 --prelr 0.0078  --batch_size 384  --ln --lnnn --predictor incn1cn1 \
    --dataset Citeseer  --epochs 100 --runs 5 --model puregcn --hiddim 256 --mplayers 1 \
    --testbs 4096  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin

python finetune_readout.py   --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --predp 0.55 \
    --gnndp 0.75  --probscale 6.5 --proboffset 4.4 --alpha 0.4  \
    --gnnlr 0.0085 --prelr 0.0078  --batch_size 384  --ln --lnnn --predictor incn1cn1 \
    --dataset Citeseer  --epochs 100 --runs 5 --model puregcn --hiddim 256 --mplayers 1 \
    --testbs 4096  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin
```

### Pubmed

```bash
python main.py --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 \
    --gnndp 0.1 --probscale 5.3 --proboffset 0.5 --alpha 0.3 \
    --gnnlr 0.0097 --prelr 0.002 --batch_size 2048 --ln --lnnn --predictor incn1cn1 \
    --dataset Pubmed --epochs 100 --runs 5 --model puregcn --hiddim 256 --mplayers 1 \
    --testbs 8192 --maskinput --jk --use_xlin --tailact

python finetune_readout.py --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --predp 0.05 \
    --gnndp 0.1 --probscale 5.3 --proboffset 0.5 --alpha 0.3 \
    --gnnlr 0.0097 --prelr 0.002 --batch_size 2048 --ln --lnnn --predictor incn1cn1 \
    --dataset Pubmed --epochs 100 --runs 5 --model puregcn --hiddim 256 --mplayers 1 \
    --testbs 8192 --maskinput --jk --use_xlin --tailact
```

### ogbl-collab

```bash
python main.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 \
    --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05 \
    --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor incn1cn1 \
    --dataset collab  --epochs 100 --runs 5 --model gcn --hiddim 64 --mplayers 1 \
    --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 
    
python finetune_readout.py   --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 \
    --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05 \
    --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor incn1cn1 \
    --dataset collab  --epochs 100 --runs 5 --model gcn --hiddim 64 --mplayers 1 \
    --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 
```
