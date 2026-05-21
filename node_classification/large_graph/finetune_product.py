import os.path as osp
import time
import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import index_to_mask


from torch_sparse import SparseTensor, matmul

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch import Tensor,FloatTensor
from scipy.linalg import hadamard
import random
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.003)
args = parser.parse_args()
print("ogbn-products")
print(args)
fix_seed(args.seed)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = T.Compose([T.ToSparseTensor()])
root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.AddSelfLoops())
evaluator = Evaluator(name='ogbn-products')

data = dataset[0]

split_idx = dataset.get_idx_split()
for split in ['train', 'valid', 'test']:
    data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])

data_ = transform(data)
data = data.to(device)
def load_data(name,device):
    return torch.load(name).to(device)
def load_model(name,device):
    return torch.load(name).to(device)

quant_feature_path = f"bqid_wo_train/ogbn-products/semantic_test_ogbn-products_sage_{args.seed}.pt"
readout_model_path = f'bqid_wo_train/ogbn-products/readout/linear_ogbn-products_sage_{args.seed}.pth' 

feature = load_data(quant_feature_path,device)
model = load_model(readout_model_path,device)
feats = feature.float()
f_size = feats.size(0)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()

    total_loss = total_examples = 0
    optimizer.zero_grad()
    
    feats_train = feats[data.train_mask]
    y = data.y[data.train_mask]
    batch_size = 500000

    num_batches = (feats_train.size(0) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, feats.size(0))

        batch_feats = feats_train[start_idx:end_idx]

        batch_output = model(batch_feats)
    
        loss = F.cross_entropy(batch_output, y[start_idx:end_idx].view(-1))
        (loss).backward()
        optimizer.step()

    total_loss += float(loss) * int(data.train_mask.sum())
    total_examples += int(data.train_mask.sum())


    return total_loss / total_examples

@torch.no_grad()
def test(epoch):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}
    
    batch_size = 500000

    num_batches = (feats.size(0) + batch_size - 1) // batch_size

    all_outputs = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, feats.size(0))

        batch_feats = feats[start_idx:end_idx]

        batch_output = model(batch_feats)

        all_outputs.append(batch_output)

    out = torch.cat(all_outputs, dim=0).argmax(dim=-1, keepdim=True)

    for split in ['train', 'valid', 'test']:
        mask = data[f'{split}_mask']
        y_true[split].append(data.y[mask].cpu())
        y_pred[split].append(out[mask].cpu())

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return train_acc, valid_acc, test_acc

times = []
best_val = 0.0
final_train = 0.0
final_test = 0.0
org_results = 0.0
train_acc_org, val_acc_org, test_acc_org = test(0)
for epoch in range(1, 100):
    start = time.time()
    loss = train(epoch)
    train_acc, val_acc, test_acc = test(epoch)
    if val_acc > best_val:
        best_val = val_acc
        final_train = train_acc
        final_test = test_acc
    print(epoch, f'Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
    times.append(time.time() - start)

print(f'Final Train: {final_train:.4f}, Best Val: {best_val:.4f}, '
      f'Final Test: {final_test:.4f}, Org Quant Test: {test_acc_org:.4f}' )
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
import os
def save_results(args, result,org_result):
    if not os.path.exists(f'results/ogbn-products'):
        os.makedirs(f'results/ogbn-products')

    filename = f'results/ogbn-products/finetune_sage_without_train_test2.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"sage " + f"{org_result*100:.2f} " + \
            f"{result*100:.2f}\n")

save_results(args, final_test,test_acc_org)
