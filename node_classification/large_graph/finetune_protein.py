import argparse

import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger_finetune_protein import Logger
import numpy as np
import nxmetis
import networkx as nx
import numpy as np
from torch_geometric.utils import to_undirected
import pickle
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import random
import torch.optim as optim
def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, y, feats, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out = model(feats)[train_idx]
    loss = criterion(out, y[train_idx].to(torch.float))
    (loss).backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, y, feats, split_idx, evaluator):
    model.eval()

    out = model(feats)
    
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': out[split_idx['train']],
    })['rocauc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': out[split_idx['valid']],
    })['rocauc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': out[split_idx['test']],
    })['rocauc']

    return train_acc, valid_acc, test_acc

def load_data(name,device):
    return torch.load(name).to(device)
def load_model(name,device):
    return torch.load(name).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=2)

    args = parser.parse_args()
    print(f"Dataset: ogbn-protein")
    print(args)

    seed(seed=42)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    
    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    
    print(data.num_features,data.x.shape)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    org_results = []

    for run in range(args.runs):
        quant_feature_path = f"save_path/protein_{run+1}.pt"
        readout_model_path = f'save_path/protein_{run+1}_readout.pth'
        
        feature = load_data(quant_feature_path,device)
        readout_model = load_model(readout_model_path,device)

        feats = feature.float()

        result_org = test(readout_model,data.y, feats, split_idx, evaluator)

        optimizer = torch.optim.Adam(readout_model.parameters(), lr=args.lr)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

        readout_model.train()
        val_score = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(readout_model, data.y, feats, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                result = test(readout_model, data.y, feats, split_idx, evaluator)
                logger.add_result(run, result)
                val_score = result[1]
                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')
            lr_scheduler.step(val_score)
        logger.print_statistics(run)
        print(f'Original Quant Result: {100 * result_org[2]:.2f}%')
        org_results.append(100*result_org[2])
    print(f"finetune result of Protein")
    results = logger.print_statistics()
    org_results = torch.tensor(org_results)
    print(f'Final Quant Test: {org_results.mean():.2f} ± {org_results.std():.2f}')
    
    def save_results(args, results,org_results):
        if not os.path.exists(f'results/ogbn-proteins'):
            os.makedirs(f'results/ogbn-proteins')

        filename = f'results/ogbn-proteins/finetune_sage_without_train_test2.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                f"sage " + f"{org_results.mean():.2f}$\pm$ {org_results.std():.2f} " + \
                f"{results.mean():.2f} $\pm$ {results.std():.2f}\n")

    save_results(args, results,org_results)


if __name__ == "__main__":
    main()
