import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from lg_parse import parse_method, parser_add_main_args
import sys
# sys.path.append("../")
from logger_test import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
# from eval import *

@torch.no_grad()
def evaluate_arxiv(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out, quant, quant_out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        # out2 = model.evl_test(quant)
    
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    
    test_acc_quant = eval_func(
        dataset.label[split_idx['test']], quant_out[split_idx['test']])
    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss,test_acc_quant, quant, out


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
args.gnn1 = 'sage'
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, n, c, d, device)

criterion = nn.NLLLoss()
eval_func = eval_acc
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    best_test_quant = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        result = evaluate_arxiv(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-2])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            best_test_quant = result[-3]
            quant = result[-2]
            if args.save_model:
                save_model(args, model, optimizer, run)
            if not os.path.exists(f'bqid_wo_train/{args.dataset}'):
                os.makedirs(f'bqid_wo_train/{args.dataset}')
            torch.save(quant,f"bqid_wo_train/{args.dataset}/semantic_test_{args.dataset}_sage_{run}.pt")
            if not os.path.exists(f'bqid_wo_train/{args.dataset}/readout'):
                os.makedirs(f'bqid_wo_train/{args.dataset}/readout')
            torch.save(model.pred_local,f'bqid_wo_train/{args.dataset}/readout/linear_{args.dataset}_sage_{run}.pth')
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Quant Test: {100 * result[-3]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%, '
                  f'Best Quant Test: {100 * best_test_quant:.2f}%, '
                  )
    logger.print_statistics(run)
    

results = logger.print_statistics()
### Save results ###
def save_result_arxiv(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')

    filename = f'results/{args.dataset}/sage_without_train_test2.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"sage " + f"{args.lr} " + \
            f"{results[0].mean():.2f} $\pm$ {results[0].std():.2f}" + f" {results[1].mean():.2f} $\pm$ {results[1].std():.2f}  \n")
save_result_arxiv(args, results)