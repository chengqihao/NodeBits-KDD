import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random

from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, APPNP
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torchvision import transforms
# from outcome_correlation import *
import glob
import os
import shutil

# from logger_ import Logger
from logger_finetune_arxiv import Logger

from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
import torch.nn as nn

@torch.no_grad()
def test(model,x,y,adj, split_idx, evaluator, feats):
    model.eval()

    out = model(feats)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def load_data(name,device):
    return torch.load(name).to(device)
def load_model(name,device):
    return torch.load(name).to(device)

def main():

    parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    args.dataset = 'ogbn-arxiv'
    args.display_step = 10
    print(args)

    fix_seed()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',transform=T.ToSparseTensor(),root=f'./data/ogb')
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    x = data.x
        
    x = x.to(device)
    adj_t = data.adj_t.to(device)
    y_true = data.y.to(device)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    idxs = torch.cat([train_idx])

    org_results = []

    for run in range(args.runs):

        quant_feature_path = f"bqid_wo_train/{args.dataset}/semantic_test_{args.dataset}_sage_{run}.pt"
        readout_model_path = f'bqid_wo_train/{args.dataset}/readout/linear_{args.dataset}_sage_{run}.pth'

        feature = load_data(quant_feature_path,device)
        readout_model = load_model(readout_model_path,device)

        feats = feature.float()
        result_org = test(readout_model, x, y_true, adj_t, split_idx, evaluator, feats)

        readout_model.train()

        optimizer = torch.optim.Adam(readout_model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        criterion = nn.NLLLoss()

        best_val = float('-inf')
        best_test = float('-inf')

        for epoch in range(0, args.epochs):
            
            readout_model.train()
            optimizer.zero_grad()

            out = readout_model(feats)
            # loss = F.nll_loss(out[idxs], y_true.squeeze(1)[idxs])
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[idxs], y_true.squeeze(1)[idxs])    
            result = test(readout_model, x, y_true, adj_t, split_idx, evaluator, feats)
            train_acc, valid_acc, test_acc = result
            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]    
            if epoch%10==0:
                print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}%, '
                        f'Test: {100 * test_acc:.2f}%,'
                        f'Best Valid: {100 * best_val:.2f}%, '
                        f'Best Test: {100 * best_test:.2f}%')    
            
            loss.backward()
            optimizer.step()
            logger.add_result(run, result)
        logger.print_statistics(run)
        print(f'Original Quant Result: {100 * result_org[2]:.2f}%')
        org_results.append(100*result_org[2])
    print(f"finetune result of Arxiv")
    results = logger.print_statistics()
    org_results = torch.tensor(org_results)
    print(f'Final Quant Test: {org_results.mean():.2f} ± {org_results.std():.2f}')

    def save_results(args, results,org_results):
        if not os.path.exists(f'results/{args.dataset}'):
            os.makedirs(f'results/{args.dataset}')

        filename = f'results/{args.dataset}/finetune_sage_without_train_test2.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                f"sage " + f"{org_results.mean():.2f}$\pm$ {org_results.std():.2f} " + \
                f"{results.mean():.2f} $\pm$ {results.std():.2f}\n")

    save_results(args, results,org_results)
if __name__ == "__main__":
    main()