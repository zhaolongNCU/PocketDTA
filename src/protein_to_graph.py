import os
import numpy as np
import pickle
import torch, math
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch_geometric
import torch.nn.functional as F
import torch_cluster


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])  # D_mu=[1, D_count]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)  # D_expand=[edge_num, 1]

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)  # RBF=[edge_num, D_count]
    return RBF


letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                        'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                        'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                        'N': 2, 'Y': 18, 'M': 12, 'X': 20, '#': 21}

num_to_letter = {v:k for k, v in letter_to_num.items()}

def coord_to_graph(protein,max_seq_len,device):
    top_k = 30
    num_rbf = 16
    max_seq_len = max_seq_len
    num_positional_embeddings=16

    name = protein['name']        
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=device, dtype=torch.float32)  
        coords = coords[: max_seq_len]

        seq = torch.as_tensor([letter_to_num[a] for a in protein['seq']],device=device, dtype=torch.long)
        seq = seq[: max_seq_len]
        seq_len = torch.tensor([seq.shape[0]])
        # seq=[seq_len]
        mask = torch.isfinite(coords.sum(dim=(1,2)))  
        # mask=[seq_len]
        coords[~mask] = np.inf  
        # coords=[seq_len, 4, 3] 
        X_ca = coords[:, 1]
        # X_ca=[seq_len, 3]
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)       
        # edge_index=[2, (seq_len-infinite_num)*top_k]
        pos_embeddings = _positional_embeddings(edge_index,num_positional_embeddings,device)       
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]          
        # E_vectors=[(seq_len-infinite_num)*top_k, 3]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device=device)

        dihedrals = _dihedrals(coords)            
        orientations = _orientations(X_ca)  
        sidechains = _sidechains(coords) 

        node_s = dihedrals  # node_s=[seq_len, 6]       
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)  
        # node_v=[seq_len, 3, 3]
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        # edge_s=[(seq_len-infinite_num)*top_k, num_positional_embeddings+D_count=32]
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        # edge_v=[(seq_len-infinite_num)*top_k, 1, 3]
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))


    data = torch_geometric.data.Data(x=X_ca, seq=seq, seq_len=seq_len, name=name,
                                        node_s=node_s, node_v=node_v,
                                        edge_s=edge_s, edge_v=edge_v,
                                        edge_index=edge_index, mask=mask)
    return data

def _dihedrals(X, eps=1e-7):

    # From https://github.com/jingraham/neurips19-graph-protein-design
    # X=[seq_len, 4, 3] 
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])  # X=[seq_len*3, 3]
    dX = X[1:] - X[:-1]  # dX=[seq_len*3-1, 3]
    U = _normalize(dX, dim=-1)  # U=[seq_len*3-1, 3]
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)

    return D_features


def _positional_embeddings(edge_index, 
                            num_embeddings=None,
                            period_range=[2, 1000],num_positional_embeddings=16,device='cuda:0'):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]     
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )          
    angles = d.unsqueeze(-1) * frequency      
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1) 
    return E

def _orientations(X):

    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _sidechains(X):

    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec