import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch.distributions import Categorical
from torch_scatter import scatter_mean

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    return x[0][idx], x[1][idx]

class _VDropout(nn.Module):
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

class GVP(nn.Module):
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            # s=[batch, n], v=[batch, v, 3]
            v = torch.transpose(v, -1, -2)
            # v=[batch, 3, v]
            vh = self.wh(v)  
            # vh=[batch, 3, h]  
            vn = _norm_no_nan(vh, axis=-2)
            # vn=[batch, h]
            s = self.ws(torch.cat([s, vn], -1))
            # s=[batch, m]
            if self.vo: 
                v = self.wv(vh) 
                # v=[batch, 3, mu]
                v = torch.transpose(v, -1, -2)
                # v=[batch, mu, 3]
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                        # gate=[batch, mu]
                    else:
                        gate = self.wsv(s)
                        # gate=[batch, mu]
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                    # v=[batch, mu, 3]
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
                    # v=[batch, mu, 3]
        else:
            s = self.ws(x)
            # s=[batch, m]
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
            # v=[batch, mu, 3]
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s

class GVPConv(MessagePassing):

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims))
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

class GVPConvLayer(nn.Module):
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        # x=([seqs_len, 100], [seqs_len, 16, 3])
        # edge_index=[2, edges_num]
        # edge_attr=([edges_num, 32+20=52], [edges_num, 1, 3])
        if autoregressive_x is not None:
            src, dst = edge_index
            # src=[edges_num], dst=[edges_num]
            mask = src < dst
            # mask=[edges_num]
            edge_index_forward = edge_index[:, mask]
            # edge_index_forward=[2, edges_forward_num]
            edge_index_backward = edge_index[:, ~mask]
            # edge_index_backward=[2, edges_backward_num]
            edge_attr_forward = tuple_index(edge_attr, mask)
            # edge_attr_forward=([edges_forward_num, 52], [edges_forward_num, 1, 3])
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            # edge_attr_forward=([edges_backward_num, 52], [edges_backward_num, 1, 3])
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),  # ([seqs_len, 100], [seqs_len, 16, 3])
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)  # # ([seqs_len, 100], [seqs_len, 16, 3])
            )
            # dh=([seqs_len, 100], [seqs_len, 16, 3])
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            # count=[seqs_len, 1]
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
            # dh=([seqs_len, 100], [seqs_len, 16, 3])
        else:
            dh = self.conv(x, edge_index, edge_attr)              #公式3

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))        #公式4
        # x=([seqs_len, 100], [seqs_len, 16, 3])
        dh = self.ff_func(x)       #公式5其中
        # dh=([seqs_len, 100], [seqs_len, 16, 3])
        # x=([seqs_len, 100], [seqs_len, 16, 3])
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

class StructureEncoder(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):
        #node_in_dim 6*3, 512*16
        #edge_in_dim 32*1, 32*1

        super(StructureEncoder, self).__init__()
        
        self.seq_in = seq_in

        if seq_in:
            self.W_s = nn.Embedding(21, 21)
            node_in_dim = (node_in_dim[0] + 21, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, 1)
        )

        self.ln = nn.LayerNorm(node_h_dim[0])

    def forward(self, h_V, h_E, edge_index, seq=None):      
        # h_V=([seqs_len, 6], [seqs_len, 3, 3])
        # h_E=([seqs_len, 32], [seqs_len, 1, 3])
        if self.seq_in and seq is not None:
            # seq=[seqs_len]
            seq = self.W_s(seq)
            # seq=[seqs_len, 20]
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            # h_V=([seqs_len, 26], [seqs_len, 3, 3])
        h_V = self.W_v(h_V)
        # h_V=([seqs_len, hid_dim], [seqs_len, 16, 3])  hid_dim 512
        h_E = self.W_e(h_E)
        # h_E=([seqs_len, 32], [seqs_len, 1, 3])
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        # h_V=([seqs_len, hid_dim], [seqs_len, 16, 3])
        out = self.W_out(h_V)
        # out=[seqs_len, hid_dim]

        out = self.ln(out)

        return out