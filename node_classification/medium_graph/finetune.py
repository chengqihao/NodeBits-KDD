import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from parse import parser_add_main_args

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, feats=None, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(feats)
    train_acc = eval_func(dataset.label[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(dataset.label[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(dataset.label[split_idx["test"]], out[split_idx["test"]])
    
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


    return train_acc, valid_acc, test_acc, valid_loss, out


parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
args.display_step = 10
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

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

dataset.label = dataset.label.to(device)
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

def load_data(name,device):
    return torch.load(name).to(device)
def load_model(name,device):
    return torch.load(name).to(device)
# feature = load_data(f"bqid_wo_train/{args.dataset}/semantic_test_{args.dataset}_{args.gnn}.pt",device)

# readout_model = torch.load(f'bqid_wo_train/{args.dataset}/readout/linear_{args.dataset}_{args.gnn}.pth').to(device)
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

org_results = []

for run in range(args.runs):
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed', 'amazon-ratings','chameleon', 'squirrel'):
    # if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    quant_feature_path = f"bqid_wo_train/{args.dataset}/semantic_test_{args.dataset}_{args.gnn}_{run}.pt"
    readout_model_path = f'bqid_wo_train/{args.dataset}/readout/linear_{args.dataset}_{args.gnn}_{run}.pth'

    feature = load_data(quant_feature_path,device)
    readout_model = load_model(readout_model_path,device)

    feats = feature.float()
    result_org = evaluate(readout_model, dataset, split_idx, eval_func, criterion, args, feats)

    readout_model.train()

    optimizer = torch.optim.Adam(readout_model.parameters(),weight_decay=args.weight_decay, lr=args.lr)

    best_val = float('-inf')
    best_test = float('-inf')

    for epoch in range(args.epochs):
        
        # optimizer = torch.optim.Adam(readout_model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
        readout_model.train()
        optimizer.zero_grad()

        out = readout_model(feats)
        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        
        optimizer.step()

        result = evaluate(readout_model, dataset, split_idx, eval_func, criterion, args, feats)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * result[0]:.2f}%, '
                    f'Valid: {100 * result[1]:.2f}%, '
                    f'Test: {100 * result[2]:.2f}%, '
                    f'Best Valid: {100 * best_val:.2f}%, '
                    f'Best Test: {100 * best_test:.2f}%')
    logger.print_statistics(run)
    print(f'Original Quant Result: {100 * result_org[2]:.2f}%')
    org_results.append(100*result_org[2])
print(f"finetune result of {args.dataset}")
results = logger.print_statistics()
org_results = torch.tensor(org_results)
print(f'Final Quant Test: {org_results.mean():.2f} ± {org_results.std():.2f}')

def save_results(args, results,org_results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')

    filename = f'results/{args.dataset}/finetune_{args.method}_without_train.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{org_results.mean():.2f}$\pm$ {org_results.std():.2f} " + \
            f"{results.mean():.2f} $\pm$ {results.std():.2f}\n")

save_results(args, results,org_results)
