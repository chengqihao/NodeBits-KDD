python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 256 --epochs 2000 --lr 0.0005 --runs 2 --local_layers 4 --post_bn --device 0
python finetune_arxiv.py --lr 0.0005 --epochs 100 --weight_decay 5e-4 --runs 2 --device 0
python main-batch.py --dataset pokec --hidden_channels 256 --epochs 2000 --batch_size 550000 --lr 0.0005 --runs 2 --local_layers 7 --in_drop 0.0 --dropout 0.2 --weight_decay 0.0 --post_bn --eval_step 9 --eval_epoch 1000 --device 0
python finetune_pokec.py --dataset pokec --lr 0.0005 --epochs 100 --device 0 --runs 2 --eval_step 9 --weight_decay 0.0 --eval_epoch 10 --batch_size 550000
python product_pre.py --device 0 --seed 0
python finetune_product.py --device 0 --seed 0
python product_pre.py --device 0 --seed 1
python finetune_product.py --device 0 --seed 1
python -u -W ignore protein.py --gpu 0 --mpnn sage --n-epochs 300 --n-heads 4
python finetune_protein.py --device 0
