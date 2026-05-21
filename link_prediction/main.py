import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from utils import PermIterator
import time
from ogbdataset import loaddataset
from typing import Iterable, Dict, Tuple

from torch import Tensor,FloatTensor
from scipy.linalg import hadamard
import math

_default_hadamard_cache = {}
def default_hadamard(X: Tensor) -> Tensor:
    dim = X.size(-1)
    if dim not in _default_hadamard_cache:
        _default_hadamard_cache[dim] = FloatTensor(hadamard(dim) ) / math.sqrt(dim)
    H = _default_hadamard_cache[dim].to(X.device).to(X.dtype)
    return X @ H

# def sign(x: Tensor):
#     return 2 * (x >= 0).float() - 1

@torch.no_grad()
def bsq(x):
    device = x.device 
    d = x.size(-1)
    # rad = sign(torch.randn(d, device=device))
    # x *=rad.view(1,-1)
    x = default_hadamard(x)
    mu = x.mean(dim=0,keepdim=True)
    x = x - mu
    xq = (x>=0).to(torch.uint8)

    return xq.detach()

def random_orthogonal_matrix(n):
    # 生成随机矩阵
    A = np.random.randn(n, n)
    # QR 分解
    Q, R = np.linalg.qr(A)
    # 保证行列式为 +1（避免旋转反射混合）
    Q *= np.sign(np.diag(R))
    return Q

@torch.no_grad()
def encoder_decoder(x):
    device = x.device 
    d = x.size(-1)
    x_org = x
    x_norm = torch.norm(x,p=2,dim=-1,keepdim=True)
    x = x/x_norm
    
    # Q = random_orthogonal_matrix(d)
    # Q = torch.from_numpy(Q).to(x.device, dtype=x.dtype)
    # x = x @ Q
    # x = default_hadamard(x)
    mu = x.mean(dim=0,keepdim=True)
    x = x - mu
    randoms = torch.mean(x.abs(),dim=0,keepdim=True)
    a_star = randoms / torch.norm(randoms, p=2, dim=1, keepdim=True)
    # xq = (x>=0).to(torch.int32)
    xq = (x>0).to(torch.int32)

    #decoder
    x_d2 = (2*xq-1)*a_star
    # x_d2 = (2*xq-1)/(d**0.5)
    x_d2 = x_d2+mu
    
    # x = x @ Q.T
    # x_d2 = default_hadamard(x_d2)
    x_d2 = x_d2 * x_norm
    # print((x_d2-x_org).abs())
    
    # x_d = (2*xq-1)/(d**0.5)
    # # print(x_d)
    # x_d = x_d+mu
    # # x_d = default_hadamard(x_d)
    # x_d2 = x_d * x_norm
    # # print((x_d-x_org).abs())
    return x_d2


# def residual_binary_message_quantization(input: Tensor, bit: int) -> Tuple[Tensor, Tensor, torch.Size]:
    
#     pack_q_input_list = []
#     input_norm_list = []
#     for i in range(bit):
#         input_norm = input.abs().mean(dim=-1)
#         q_input = input > 0
#         dequant_input = (q_input.to(torch.float32) * 2 - 1) * input_norm.view(-1, 1)
#         input -= dequant_input
        
#         shifts = torch.arange(7, -1, -1, device=input.device)
#         q_input = q_input.flatten()
#         remainder = q_input.numel() % 8
#         if remainder:
#             pad_len = 8 - remainder
#             q_input = torch.cat([q_input, torch.zeros(pad_len, dtype=q_input.dtype, device=q_input.device)])
        
#         pack_q_input = (q_input.view(-1, 8) << shifts).sum(dim=1).to(torch.uint8)
#         pack_q_input_list.append(pack_q_input)
#         if input.dtype == torch.float32:
#             input_norm_list.append(input_norm.to(torch.bfloat16))
#         else:
#             input_norm_list.append(input_norm)
            
#     return torch.stack(pack_q_input_list), torch.stack(input_norm_list)
    
# def residual_binary_message_dequantization(q_input: Tensor, q_scale: Tensor, input_tempin_shape: torch.Size, bit: int):
#     if q_scale.dtype == torch.bfloat16:
#         q_scale = q_scale.to(torch.float32)
    
#     total_input = torch.zeros(input_tempin_shape, device=q_input.device)
#     for i in range(bit):        
#         # need [N, D] tensor. 
#         q_input_i = q_input[i]
#         q_scale_i = q_scale[i]
#         shifts = torch.arange(7, -1, -1, device=q_input_i.device)
#         unpacked = ((q_input_i.unsqueeze(1) >> shifts) & 1).to(torch.uint8)
#         input = unpacked.view(-1)[:math.prod(input_tempin_shape)].reshape(input_tempin_shape)
#         input = input.to(torch.float32).mul_(2).sub_(1).mul_(q_scale_i.view(-1, 1))
#         total_input += input

#     return total_input.contiguous()

# @torch.no_grad()
# def encoder_decoder(x: torch.Tensor, bit: int = 3):
#     device = x.device
#     d = x.size(-1)
#     x_org = x
#     x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)   # [N,1]
#     x = x / x_norm

#     # === 量化部分 (Encoder) ===
#     # RBQ 需要保存 shape 信息
#     q_input, q_scale = residual_binary_message_quantization(x.clone(), bit=bit)
    
#     # === 解码部分 (Decoder) ===
#     x_d2 = residual_binary_message_dequantization(
#         q_input, q_scale, x.shape, bit
#     )

#     # 恢复原始 norm
#     x_d2 = x_d2 * x_norm
    
#     return x_d2
    


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None, feats = None):
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        # print(perm,perm.shape)
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h, _ = model(data.x, adj)
        
        # quantize = x + (quantize - x).detach()
        # h = feats
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        # print(neg_losss, pos_losss, total_commit_loss)
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

import pickle

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input, epoch, feats = None, dataset = None):
    model.eval()
    predictor.eval()

    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h, indices = model(data.x, adj)
    # h = feats
    
    indices = indices.detach().cpu().numpy()
    print(indices.shape)
    np.savez(f"semantic_indices_{dataset}", indices)
    
    # if epoch == 99:
    #     h = encoder_decoder(h.detach())
    
    # h = encoder_decoder(h.detach())
    h_new = encoder_decoder(h.detach())
    # h_new = encoder_decoder(h.clone())
    # h_new = h.clone()
    
    
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)


    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        h, _ = model(data.x, adj)
    
    # print(h, pos_test_edge)
    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)
                              
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']

        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
        
        
        
        
    pos_train_pred_quant = torch.cat([
        predictor(h_new, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)


    pos_valid_pred_quant = torch.cat([
        predictor(h_new, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred_quant = torch.cat([
        predictor(h_new, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        # h_new, _ = model(data.x, adj)
        h, _ = model(data.x, adj)
        h_new = encoder_decoder(h.detach())
    
    # print(h, pos_test_edge)
    pos_test_pred_quant = torch.cat([
        predictor(h_new, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred_quant = torch.cat([
        predictor(h_new, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)
                              
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits_quant = evaluator.eval({
            'y_pred_pos': pos_train_pred_quant,
            'y_pred_neg': neg_valid_pred_quant,
        })[f'hits@{K}']

        valid_hits_quant = evaluator.eval({
            'y_pred_pos': pos_valid_pred_quant,
            'y_pred_neg': neg_valid_pred_quant,
        })[f'hits@{K}']
        test_hits_quant = evaluator.eval({
            'y_pred_pos': pos_test_pred_quant,
            'y_pred_neg': neg_test_pred_quant,
        })[f'hits@{K}']

        results[f'Hits@{K}_quant'] = (train_hits_quant, valid_hits_quant, test_hits_quant)                          
    
    
    # return results, h.cpu()
    return results, h_new.cpu()


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    
    parser.add_argument('--kmeans', type=int, default=1)
    parser.add_argument('--codebook', type=int, default=16)
    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    parser.add_argument('--device', type=int, default=0)
    
    # detailed hyperparameters
    parser.add_argument('--alpha', type=float, default=1)

    # predictor used, such as CN
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', default='gcn', choices=convdict.keys())
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument('--beta', type=float, default=1)
    
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")
    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    
    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")

    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()
    return args

def load_out_t(name):
    return torch.from_numpy(np.load(name)["arr_0"])

def main():
    args = parseargs()
    print(args, flush=True)

    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    log_root = os.environ.get("NODEBITS_LOG_DIR", "./rec")
    writer = SummaryWriter(os.path.join(log_root, f"{args.model}_{args.predictor}"))
    writer.add_text("hyperparams", hpstr)

    if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name=f'ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{args.dataset}')

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load)
    
    
    data = data.to(device)

    predfn = predictor_dict[args.predictor]

    # if args.predictor in ["cn1"]:
    #     predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)

    


    ret = []
    ret_quant = []

    # set_seed(42)
    for run in range(0, args.runs):
        set_seed(run)
        # set_seed(42)
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load) # get a new split of dataset
            data = data.to(device)
            # print(data,data.y)
            # s()
        bestscore = None
        
        # build model
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        # if args.loadx:
        #     with torch.no_grad():
        #         model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
        #     model.xemb[0].weight.requires_grad_(False)
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        # if args.loadmod:
        #     keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
        #     print("unmatched params", keys, flush=True)
        #     keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
        #     print("unmatched params", keys, flush=True)
        
        print(model)
        print(predictor)
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}])
        
        
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            print(f"trn time {time.time()-t1:.2f} s", flush=True)
            if True:
                t1 = time.time()
                results, h = test(model, predictor, data, split_edge, evaluator,
                               args.testbs, args.use_valedges_as_input, epoch, dataset = args.dataset)
                if epoch == args.epochs:
                    torch.save({
                        'h': h.cpu(),
                        'predictor_state_dict': predictor.state_dict()
                    }, f'saved_h_predictor_{args.dataset}_{run}.pth')
                
                print(f"test time {time.time()-t1:.2f} s")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}
                for key, result in results.items():
                    writer.add_scalars(f"{key}_{run}", {
                        "trn": result[0],
                        "val": result[1],
                        "tst": result[2]
                    }, epoch)

                if True:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                            # if args.save_gemb:
                            #     torch.save(h, f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}.pt")
                            # if args.savex:
                            #     torch.save(model.xemb[0].weight.detach(), f"gemb/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            # if args.savemod:
                            #     torch.save(model.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pt")
                            #     torch.save(predictor.state_dict(), f"gmodel/{args.dataset}_{args.model}_{args.predictor}_{args.hiddim}_{run}.pre.pt")
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---', flush=True)
        print(f"best {bestscore}")
        if args.dataset == "collab":
            ret.append(bestscore["Hits@50"][-2:])
            ret_quant.append(bestscore["Hits@50_quant"][-2:])
        elif args.dataset == "ppa":
            ret.append(bestscore["Hits@100"][-2:])
            ret_quant.append(bestscore["Hits@100_quant"][-2:])
        elif args.dataset == "ddi":
            ret.append(bestscore["Hits@20"][-2:])
            ret_quant.append(bestscore["Hits@20_quant"][-2:])
        elif args.dataset == "citation2":
            ret.append(bestscore[-2:])
            ret_quant.append(bestscore[-2:])
        elif args.dataset in ["Pubmed", "Cora", "Citeseer"]:
            ret.append(bestscore["Hits@100"][-2:])
            ret_quant.append(bestscore["Hits@100_quant"][-2:])
        else:
            raise NotImplementedError
    ret = np.array(ret)
    ret_quant = np.array(ret_quant)
    print(ret)
    print('---', flush=True)
    print(ret_quant)
    print(f"Final result: val {np.average(ret[:, 0]):.4f} {np.std(ret[:, 0]):.4f} tst {np.average(ret[:, 1]):.4f} {np.std(ret[:, 1]):.4f}")
    print(f"Final quant result: val {np.average(ret_quant[:, 0]):.4f} {np.std(ret_quant[:, 0]):.4f} tst {np.average(ret_quant[:, 1]):.4f} {np.std(ret_quant[:, 1]):.4f}")


if __name__ == "__main__":
    main()
