import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch import Tensor,FloatTensor
from scipy.linalg import hadamard
import math

@torch.no_grad()
def encoder_decoder(x):
    device = x.device 
    d = x.size(-1)
    x_org = x
    x_norm = torch.norm(x,p=2,dim=-1,keepdim=True)
    x = x/x_norm
    mu = x.mean(dim=0,keepdim=True)
    x = x - mu
    randoms = torch.mean(x.abs(),dim=0,keepdim=True)
    a_star = randoms / torch.norm(randoms, p=2, dim=1, keepdim=True)
    xq = (x>=0).to(torch.int32)

    #decoder
    x_d2 = (2*xq-1)*a_star
    x_d2 = x_d2+mu
    x_d2 = x_d2 * x_norm
    return x_d2


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
            in_dropout=0.15, dropout=0.5, heads=1,
            pre_ln=False, post_bn=True, local_attn=False,jk=True):
        super(GAT, self).__init__()

        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn
        self.jk = jk
        ## Two initialization strategies on beta
        # self.beta = beta
        #self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        ## first layer
        # self.h_lins.append(torch.nn.Linear(in_channels, heads*hidden_channels))
        if local_attn:
            self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
        else:
            # self.local_convs.append(SAGEConv(in_channels, heads*hidden_channels,
            #     cached=False, normalize=True))
            self.local_convs.append(SAGEConv(in_channels, heads*hidden_channels))
        self.lins.append(torch.nn.Linear(in_channels, heads*hidden_channels))
        self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.post_bn:
            self.post_bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        ## following layers
        for _ in range(local_layers-1):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if local_attn:
                self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                    concat=True, add_self_loops=False, bias=False))
            else:
                self.local_convs.append(SAGEConv(heads*hidden_channels, heads*hidden_channels))
                # self.local_convs.append(SAGEConv(heads*hidden_channels, heads*hidden_channels,
                #     cached=False, normalize=True))
            
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)
        # self.linear_gnn = torch.nn.Linear(heads*hidden_channels, local_layers*3)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()
        #torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)

        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)

            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_local = x_local + x
            else:
                x_local = x

        x = self.pred_local(x_local)

        if not self.training:
            quant = encoder_decoder(x_local.detach())
            return x,quant, self.pred_local(quant)
        return x