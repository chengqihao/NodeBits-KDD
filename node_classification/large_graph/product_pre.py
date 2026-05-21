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
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import index_to_mask
import os
import random
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def encoder_decoder(x):
    device = x.device 
    d = x.size(-1)
    x_org = x
    x_norm = torch.norm(x,p=2,dim=-1,keepdim=True)
    x = x/x_norm
    # x = default_hadamard(x)
    mu = x.mean(dim=0,keepdim=True)
    x = x - mu
    randoms = torch.mean(x.abs(),dim=0,keepdim=True)
    a_star = randoms / torch.norm(randoms, p=2, dim=1, keepdim=True)
    xq = (x>=0).to(torch.int32)

    #decoder
    x_d2 = (2*xq-1)*a_star
    # x_d2 = (2*xq-1)/(d**0.5)
    x_d2 = x_d2+mu
    # x_d2 = default_hadamard(x_d2)
    x_d2 = x_d2 * x_norm
    return x_d2

class GNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        self.conv = SAGEConv(in_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv(x, edge_index)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()

        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNBlock(
                hidden_channels,
                hidden_channels,
            )
            self.convs.append(conv)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin1(x)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = self.norm(x).relu()
        x_local = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x_local)
        if not self.training:
            quant = encoder_decoder(x_local.detach())
            return x, quant, self.lin2(quant)
        return x


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
print(args)
print("ogbn-products")
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
print(root)
fix_seed(args.seed)

dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.AddSelfLoops())
evaluator = Evaluator(name='ogbn-products')

data = dataset[0]
split_idx = dataset.get_idx_split()
for split in ['train', 'valid', 'test']:
    data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])

train_loader = RandomNodeLoader(data, num_parts=10, shuffle=True,
                                num_workers=5)
# Increase the num_parts of the test loader if you cannot fit
# the full batch graph into your GPU:
test_loader = RandomNodeLoader(data, num_parts=1, num_workers=5)

model = GNN(
    in_channels=dataset.num_features,
    hidden_channels=128,
    out_channels=dataset.num_classes,
    num_layers=5,  # You can try 1000 layers for fun
    dropout=0.5
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:03d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()

        # Memory-efficient aggregations:
        data = transform(data)
        out = model(data.x, data.adj_t)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].view(-1))
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())
        pbar.update(1)

    pbar.close()

    return total_loss / total_examples

@torch.no_grad()
def test(epoch):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}
    y_pred_quant = {"train": [], "valid": [], "test": []}
    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:03d}')
    quant_list = []
    for data in test_loader:
        # Memory-efficient aggregations
        data = transform(data)
        out, quant, quant_out = model(data.x, data.adj_t)
        out = out.argmax(dim=-1, keepdim=True)
        quant_out = quant_out.argmax(dim=-1, keepdim=True)
        for split in ['train', 'valid', 'test']:
            mask = data[f'{split}_mask']
            quant_list.append(quant[mask])
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())
            y_pred_quant[split].append(quant_out[mask].cpu())

        pbar.update(1)

    pbar.close()
    quant_list = torch.cat(quant_list, dim=0)
    # quant_list_cpu = quant_list.detach().cpu().numpy()
    # torch.save(data_dict,f"semantic_indices_ogbn-products.pt")
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

    test_acc_quant = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred_quant['test'], dim=0),
    })['acc']

    return train_acc, valid_acc, test_acc,test_acc_quant,quant_list

times = []
best_val = 0.0
final_train = 0.0
final_test = 0.0
final_test_quant = 0.0
for epoch in range(1, 1001):
    start = time.time()
    loss = train(epoch)
    train_acc, val_acc, test_acc, test_acc_quant, quant_list = test(epoch)
    if val_acc > best_val:
        best_val = val_acc
        final_train = train_acc
        final_test = test_acc
        final_test_quant = test_acc_quant
        if not os.path.exists(f'bqid_wo_train/ogbn-products'):
            os.makedirs(f'bqid_wo_train/ogbn-products')
        torch.save(quant_list,f"bqid_wo_train/ogbn-products/semantic_test_ogbn-products_sage_{args.seed}.pt")
        if not os.path.exists(f'bqid_wo_train/ogbn-products/readout'):
            os.makedirs(f'bqid_wo_train/ogbn-products/readout') 
        torch.save(model.lin2,f'bqid_wo_train/ogbn-products/readout/linear_ogbn-products_sage_{args.seed}.pth')
    print(f'Loss: {loss:.4f}, Train: {train_acc*100:.4f}, Val: {val_acc*100:.4f}, '
        f'Test: {test_acc*100:.4f}, Quant Test: {test_acc_quant*100:.4f}')
    times.append(time.time() - start)

print(f'Final Train: {final_train*100:.4f}, Best Val: {best_val*100:.4f}, '
    f'Final Test: {final_test*100:.4f}, Final Quant Test: {final_test_quant*100:.4f}')
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

def save_results(final_test,final_test_quant):
    if not os.path.exists(f'results/ogbn-products'):
        os.makedirs(f'results/ogbn-products')

    filename = f'results/ogbn-products/sage_without_train_test2.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"sage " + \
            f"{final_test*100:.4f}" + f" {final_test_quant*100:.4f} \n")
save_results(final_test,final_test_quant)
