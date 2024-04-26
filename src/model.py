import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn.utils.weight_norm import weight_norm

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)   #池化窗口为3

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            #print(self.h_mat.shape,v_.shape,q_.shape)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits


        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  

        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  
            return logits.transpose(2, 3).transpose(1, 2)  

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)
            logits = self.p_net(logits).squeeze(1) * self.k
        return logits


class proSeqEncoder(nn.Module):
    """protein sequence feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, max_pro_seq_len, dropout=0.3):
        super().__init__()
        self.input_dim = protein_dim     
        self.hid_dim = hid_dim           
        self.dropout = dropout           
        self.n_layers = n_layers         
        self.pos_embedding = nn.Embedding(max_pro_seq_len, protein_dim)    
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        device = protein.device
        pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(device)  
        protein = protein + self.pos_embedding(pos)

        conv_input = self.fc(protein)

        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            self.scale = self.scale.to(device)
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size, protein len, hid dim]
        conved = self.ln(conved)
        return conved


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        #self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class DTAPredictor(nn.Module):
    def __init__(self, drug_struc_encoder,target_seq_dim,target_struc_encoder,
                 drug_seq_dim,drug_dim,target_dim,gvp_output_dim,h_dim,n_heads,
                 use_drug_seq, use_drug_struc, use_target_seq, use_target_struc):
        super().__init__()
        self.use_drug_seq = use_drug_seq
        self.use_drug_struc = use_drug_struc
        self.use_target_seq = use_target_seq
        self.use_target_struc = use_target_struc
        self.h_dim = h_dim
        self.drug_seq_Linear = nn.Sequential(nn.Linear(drug_seq_dim,512),nn.ReLU(),nn.Linear(512,drug_dim))
        self.drug_struc_Linear = drug_struc_encoder
        self.target_seq_Linear = nn.Sequential(nn.Linear(target_seq_dim,256),nn.ReLU(),nn.Linear(256,target_dim))       
        #self.target_seq_ln = nn.LayerNorm(128*2)
        self.target_struc = target_struc_encoder
        self.target_struc_Linear = nn.Sequential(nn.Linear(gvp_output_dim,128),nn.ReLU(),nn.Linear(128,target_dim),)          
        self.drug_target_fusion = weight_norm(BANLayer(v_dim=drug_dim, q_dim=target_dim, h_dim=h_dim, h_out=n_heads, k=3),
                                              name='h_mat', dim=None)
        self.affinity_layer = MLPDecoder(h_dim,1024,256,1)

    def make_masks(self, proteins_len, pockets_len,compounds_len, protein_max_len, pocket_max_len, compound_max_len):
        N = len(proteins_len)  # batch size
        protein_mask = torch.zeros((N, protein_max_len))
        pocket_mask = torch.zeros((N,pocket_max_len))
        compound_mask = torch.zeros((N, compound_max_len))
        for i in range(N):
            protein_mask[i, :proteins_len[i]] = 1
            pocket_mask[i, :pockets_len[i]] = 1
            compound_mask[i, :compounds_len[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)     
        pocket_mask = pocket_mask.unsqueeze(1).unsqueeze(2)       
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)   #

        protein_mask, pocket_mask,compound_mask = protein_mask.to(proteins_len.device), pocket_mask.to(pockets_len.device),compound_mask.to(compounds_len.device)
        return protein_mask,pocket_mask,compound_mask

    def struc_data_format_change(self, sample_num, sample_len, struc_emb, pro_seq_lens, device):
        struc_emb_new = None
        seq_len_1, seq_len_2 = 0, 0
        for i in range(sample_num):
            if i == 0:
                seq_len_1, seq_len_2 = 0, pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]         
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device) 
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)   
                struc_emb_new = modal2_emb_one.unsqueeze(0)                 
            else:
                seq_len_1, seq_len_2 = seq_len_2, seq_len_2+pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device)
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)
                struc_emb_new = torch.cat((struc_emb_new, modal2_emb_one.unsqueeze(0)), dim=0)
        struc_emb = struc_emb_new   
        return struc_emb

    def forward(self, drug_seq_feat, drug_struc_feat, target_seq_feat, target_struc_feat, pockets_len,):
        device = drug_seq_feat.device

        embeddings = []

        if self.use_drug_seq:
            drug_seq_emb = self.drug_seq_Linear(drug_seq_feat)  
            embeddings.append(drug_seq_emb.unsqueeze(1)) 
        
        if self.use_drug_struc:
            drug_struc_emb = self.drug_struc_Linear(drug_struc_feat)  
            embeddings.append(drug_struc_emb)

        if embeddings:
            drug_emb = torch.cat(embeddings, dim=1)  
        else:
            raise ValueError("At least one drug feature must be used.")

        embeddings = []  

        if self.use_target_seq:

            target_seq_emb = self.target_seq_Linear(target_seq_feat)  
            embeddings.append(target_seq_emb.unsqueeze(1))  

        if self.use_target_struc:
            B = drug_seq_feat.size(0)
            max_pocket_len_batch = torch.max(pockets_len).item()
            struc_emb = self.target_struc(*target_struc_feat)  
            target_struc_emb = self.struc_data_format_change(B, max_pocket_len_batch, struc_emb, pockets_len, device)  
            target_struc_emb = self.target_struc_Linear(target_struc_emb)
            embeddings.append(target_struc_emb)


        if embeddings:
            target_emb = torch.cat(embeddings, dim=1)  
        else:
            raise ValueError("At least one target feature must be used.")


        drug_target, att = self.drug_target_fusion(drug_emb, target_emb)  


        affinity = self.affinity_layer(drug_target)

        return affinity.squeeze(-1)

class DTAPredictor_test(nn.Module):
    def __init__(self, drug_struc_encoder,target_seq_dim,target_struc_encoder,
                 drug_seq_dim,drug_dim,target_dim,gvp_output_dim,h_dim,n_heads,
                 use_drug_seq, use_drug_struc, use_target_seq, use_target_struc):
        super().__init__()
        self.use_drug_seq = use_drug_seq
        self.use_drug_struc = use_drug_struc
        self.use_target_seq = use_target_seq
        self.use_target_struc = use_target_struc
        self.h_dim = h_dim
        self.drug_seq_Linear = nn.Sequential(nn.Linear(drug_seq_dim,512),nn.ReLU(),nn.Linear(512,drug_dim))
        self.drug_struc_Linear = drug_struc_encoder
        self.target_seq_Linear = nn.Sequential(nn.Linear(target_seq_dim,256),nn.ReLU(),nn.Linear(256,target_dim))       
        #self.target_seq_ln = nn.LayerNorm(128*2)
        self.target_struc = target_struc_encoder
        self.target_struc_Linear = nn.Sequential(nn.Linear(gvp_output_dim,128),nn.ReLU(),nn.Linear(128,target_dim),)          
        self.drug_target_fusion = weight_norm(BANLayer(v_dim=drug_dim, q_dim=target_dim, h_dim=h_dim, h_out=n_heads, k=3),
                                              name='h_mat', dim=None)
        self.affinity_layer = MLPDecoder(h_dim,1024,256,1)

    def make_masks(self, proteins_len, pockets_len,compounds_len, protein_max_len, pocket_max_len, compound_max_len):
        N = len(proteins_len)  # batch size
        protein_mask = torch.zeros((N, protein_max_len))
        pocket_mask = torch.zeros((N,pocket_max_len))
        compound_mask = torch.zeros((N, compound_max_len))
        for i in range(N):
            protein_mask[i, :proteins_len[i]] = 1
            pocket_mask[i, :pockets_len[i]] = 1
            compound_mask[i, :compounds_len[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)     
        pocket_mask = pocket_mask.unsqueeze(1).unsqueeze(2)       
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)   #

        protein_mask, pocket_mask,compound_mask = protein_mask.to(proteins_len.device), pocket_mask.to(pockets_len.device),compound_mask.to(compounds_len.device)
        return protein_mask,pocket_mask,compound_mask

    def struc_data_format_change(self, sample_num, sample_len, struc_emb, pro_seq_lens, device):
        struc_emb_new = None
        seq_len_1, seq_len_2 = 0, 0
        for i in range(sample_num):
            if i == 0:
                seq_len_1, seq_len_2 = 0, pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]         
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device) 
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)   
                struc_emb_new = modal2_emb_one.unsqueeze(0)                 
            else:
                seq_len_1, seq_len_2 = seq_len_2, seq_len_2+pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device)
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)
                struc_emb_new = torch.cat((struc_emb_new, modal2_emb_one.unsqueeze(0)), dim=0)
        struc_emb = struc_emb_new   
        return struc_emb

    def forward(self, drug_seq_feat, drug_struc_feat, target_seq_feat, target_struc_feat, pockets_len,):
        device = drug_seq_feat.device

        embeddings = []

        if self.use_drug_seq:
            drug_seq_emb = self.drug_seq_Linear(drug_seq_feat)  
            embeddings.append(drug_seq_emb.unsqueeze(1)) 
        
        if self.use_drug_struc:
            drug_struc_emb = self.drug_struc_Linear(drug_struc_feat)  
            embeddings.append(drug_struc_emb)

        if embeddings:
            drug_emb = torch.cat(embeddings, dim=1)  
        else:
            raise ValueError("At least one drug feature must be used.")

        embeddings = []  

        if self.use_target_seq:

            target_seq_emb = self.target_seq_Linear(target_seq_feat)  
            embeddings.append(target_seq_emb.unsqueeze(1))  

        if self.use_target_struc:
            B = drug_seq_feat.size(0)
            max_pocket_len_batch = torch.max(pockets_len).item()
            struc_emb = self.target_struc(*target_struc_feat)  
            target_struc_emb = self.struc_data_format_change(B, max_pocket_len_batch, struc_emb, pockets_len, device)  
            target_struc_emb = self.target_struc_Linear(target_struc_emb)
            embeddings.append(target_struc_emb)


        if embeddings:
            target_emb = torch.cat(embeddings, dim=1)  
        else:
            raise ValueError("At least one target feature must be used.")


        drug_target, att = self.drug_target_fusion(drug_emb, target_emb)  


        affinity = self.affinity_layer(drug_target)

        return affinity.squeeze(-1),att


    def __call__(self, data):
        drug_seq_feat,drug_struc_feat,target_seq_feat,target_struc_feat, gpu_split,device= data

        target_struc_feat = [target_struc_feat[i] for i in gpu_split]      
        pockets_len = [i.seq_len for i in target_struc_feat]
        pockets_len = torch.stack(pockets_len).squeeze(-1)

        struc_feat = torch_geometric.data.Batch.from_data_list(target_struc_feat)     
        h_V, h_E, edge_index, seq = (struc_feat.node_s.to(device), struc_feat.node_v.to(device)), (struc_feat.edge_s.to(device), struc_feat.edge_v.to(device)), struc_feat.edge_index.to(device), struc_feat.seq.to(device)
        struc_feat = (h_V, h_E, edge_index, seq)
        label = self.forward(drug_seq_feat,drug_struc_feat,target_seq_feat,struc_feat,pockets_len,)
        return label


class PredictorGraphMVPAblation(nn.Module):
    def __init__(self, drug_struc_encoder,target_seq_dim,target_struc_encoder,
                 drug_seq_dim,drug_dim,target_dim,gvp_output_dim,h_dim,n_heads,
                 use_drug_seq, use_drug_struc, use_target_seq, use_target_struc):
        super().__init__()
        self.use_drug_seq = use_drug_seq
        self.use_drug_struc = use_drug_struc
        self.use_target_seq = use_target_seq
        self.use_target_struc = use_target_struc
        self.h_dim = h_dim
        self.drug_seq_Linear = nn.Sequential(nn.Linear(drug_seq_dim,512),nn.ReLU(),nn.Linear(512,drug_dim))
        self.drug_struc_Linear = nn.Sequential(nn.Linear(300,512),nn.ReLU(),nn.Linear(512,drug_dim))
        self.target_seq_Linear = nn.Sequential(nn.Linear(target_seq_dim,256),nn.ReLU(),nn.Linear(256,target_dim))       
        #self.target_seq_ln = nn.LayerNorm(128*2)
        self.target_struc = target_struc_encoder
        self.target_struc_Linear = nn.Sequential(nn.Linear(gvp_output_dim,128),nn.ReLU(),nn.Linear(128,target_dim),)          
        self.drug_target_fusion = weight_norm(BANLayer(v_dim=drug_dim, q_dim=target_dim, h_dim=h_dim, h_out=n_heads, k=3),
                                              name='h_mat', dim=None)
        self.affinity_layer = MLPDecoder(h_dim,1024,256,1)

    def make_masks(self, proteins_len, pockets_len,compounds_len, protein_max_len, pocket_max_len, compound_max_len):
        N = len(proteins_len)  # batch size
        protein_mask = torch.zeros((N, protein_max_len))
        pocket_mask = torch.zeros((N,pocket_max_len))
        compound_mask = torch.zeros((N, compound_max_len))
        for i in range(N):
            protein_mask[i, :proteins_len[i]] = 1
            pocket_mask[i, :pockets_len[i]] = 1
            compound_mask[i, :compounds_len[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)     #B*1*1*max_seq_len
        pocket_mask = pocket_mask.unsqueeze(1).unsqueeze(2)       #B*1*1*max_pocket_len
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)   #B*1*1*max_com_len

        protein_mask, pocket_mask,compound_mask = protein_mask.to(proteins_len.device), pocket_mask.to(pockets_len.device),compound_mask.to(compounds_len.device)
        return protein_mask,pocket_mask,compound_mask

    def struc_data_format_change(self, sample_num, sample_len, struc_emb, pro_seq_lens, device):
        struc_emb_new = None
        seq_len_1, seq_len_2 = 0, 0
        for i in range(sample_num):
            if i == 0:
                seq_len_1, seq_len_2 = 0, pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]         #提取自己长度的结构嵌入 seq_len*512
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device) #(max_seq_len-seq_len)*512pad填充
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)   #max_seq_len*512
                struc_emb_new = modal2_emb_one.unsqueeze(0)                 #1*max_seq_len*512
            else:
                seq_len_1, seq_len_2 = seq_len_2, seq_len_2+pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device)
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)
                struc_emb_new = torch.cat((struc_emb_new, modal2_emb_one.unsqueeze(0)), dim=0)
        struc_emb = struc_emb_new   #B*max_seq_len*512        B*max_pocket_len*128
        return struc_emb

    def forward(self, drug_seq_feat, drug_struc_feat, target_seq_feat, target_struc_feat, pockets_len,):
        device = drug_seq_feat.device
        # 根据传入的布尔值变量决定是否使用相应特征
        embeddings = []
        #print(self.use_drug_seq)
        #print(self.use_drug_struc)
        #print(self.use_target_seq)
        #print(self.use_target_struc)
        #self.use_drug_struc = False
        #print('before MLP ',drug_struc_feat.shape)
        if self.use_drug_seq:
            drug_seq_emb = self.drug_seq_Linear(drug_seq_feat)  # B*2048变为B*256
            embeddings.append(drug_seq_emb.unsqueeze(1))  # 添加新的维度以便后续拼接
        
        if self.use_drug_struc:
            drug_struc_emb = self.drug_struc_Linear(drug_struc_feat)  # B*36*64
            embeddings.append(drug_struc_emb)
        #print('After MLP',drug_struc_emb.shape)
        # 拼接药物的序列和结构特征
        if embeddings:
            drug_emb = torch.cat(embeddings, dim=1)  # B*(37 or less)*64
        else:
            raise ValueError("At least one drug feature must be used.")
        #print(drug_emb.shape)
        embeddings = []  # 清空列表以便重新使用

        if self.use_target_seq:

            target_seq_emb = self.target_seq_Linear(target_seq_feat)  # B*256
            embeddings.append(target_seq_emb.unsqueeze(1))  # 添加新的维度以便后续拼接

        if self.use_target_struc:
            B = drug_seq_feat.size(0)
            max_pocket_len_batch = torch.max(pockets_len).item()
            struc_emb = self.target_struc(*target_struc_feat)  # seqs_len*512
            target_struc_emb = self.struc_data_format_change(B, max_pocket_len_batch, struc_emb, pockets_len, device)  # B*max_len+1*128
            target_struc_emb = self.target_struc_Linear(target_struc_emb)
            embeddings.append(target_struc_emb)

        # 拼接蛋白质的序列和结构特征
        if embeddings:
            target_emb = torch.cat(embeddings, dim=1)  # B*(max_len+1 or less)*256
        else:
            raise ValueError("At least one target feature must be used.")

        #print(target_emb.shape)
        # 将药物和靶标的嵌入进行融合
        drug_target, att = self.drug_target_fusion(drug_emb, target_emb)  # B*5*(37 or less)*(max_len+1 or less)

        # 计算亲和力
        affinity = self.affinity_layer(drug_target)

        return affinity.squeeze(-1)

    def __call__(self, data):
        drug_seq_feat,drug_struc_feat,target_seq_feat,target_struc_feat, gpu_split,device= data
        #device = drug_seq_feat.device
        # struc_feat = struc_feat[gpu_split[0][0]: gpu_split[0][1]]
        target_struc_feat = [target_struc_feat[i] for i in gpu_split]      #B
        pockets_len = [i.seq_len for i in target_struc_feat]
        pockets_len = torch.stack(pockets_len).squeeze(-1)
        #print(pockets_len)
        #print(len(pockets_len))
        #print(max(pockets_len))
        struc_feat = torch_geometric.data.Batch.from_data_list(target_struc_feat)     #将多个图数据合并成一个大图，便于图神经网络批处理
        h_V, h_E, edge_index, seq = (struc_feat.node_s.to(device), struc_feat.node_v.to(device)), (struc_feat.edge_s.to(device), struc_feat.edge_v.to(device)), struc_feat.edge_index.to(device), struc_feat.seq.to(device)
        # torch.Size([2740, 6]) torch.Size([2740, 3, 3]) torch.Size([82200, 32]) torch.Size([82200, 1, 3]) torch.Size([2, 82200])
        # print(h_V[0].shape,h_V[1].shape,h_E[0].shape,h_E[1].shape,edge_index.shape,len(seq))
        # print(seq)
        struc_feat = (h_V, h_E, edge_index, seq)
        label = self.forward(drug_seq_feat,drug_struc_feat,target_seq_feat,struc_feat,pockets_len,)
        return label


class PredictorBANAblation(nn.Module):
    def __init__(self, drug_struc_encoder,target_seq_dim,target_struc_encoder,
                 drug_seq_dim,drug_dim,target_dim,gvp_output_dim,h_dim,n_heads,
                 use_drug_seq, use_drug_struc, use_target_seq, use_target_struc):
        super().__init__()
        self.use_drug_seq = use_drug_seq
        self.use_drug_struc = use_drug_struc
        self.use_target_seq = use_target_seq
        self.use_target_struc = use_target_struc
        self.h_dim = h_dim
        self.drug_seq_Linear = nn.Sequential(nn.Linear(drug_seq_dim,512),nn.ReLU(),nn.Linear(512,drug_dim))
        self.drug_struc_Linear = drug_struc_encoder
        self.target_seq_Linear = nn.Sequential(nn.Linear(target_seq_dim,256),nn.ReLU(),nn.Linear(256,target_dim))       

        self.target_struc = target_struc_encoder
        self.target_struc_Linear = nn.Sequential(nn.Linear(gvp_output_dim,128),nn.ReLU(),nn.Linear(128,target_dim),)          

        self.affinity_layer = MLPDecoder(drug_dim+target_dim,1024,256,1)

    def make_masks(self, proteins_len, pockets_len,compounds_len, protein_max_len, pocket_max_len, compound_max_len):
        N = len(proteins_len)  # batch size
        protein_mask = torch.zeros((N, protein_max_len))
        pocket_mask = torch.zeros((N,pocket_max_len))
        compound_mask = torch.zeros((N, compound_max_len))
        for i in range(N):
            protein_mask[i, :proteins_len[i]] = 1
            pocket_mask[i, :pockets_len[i]] = 1
            compound_mask[i, :compounds_len[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)    
        pocket_mask = pocket_mask.unsqueeze(1).unsqueeze(2)      
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)  

        protein_mask, pocket_mask,compound_mask = protein_mask.to(proteins_len.device), pocket_mask.to(pockets_len.device),compound_mask.to(compounds_len.device)
        return protein_mask,pocket_mask,compound_mask

    def struc_data_format_change(self, sample_num, sample_len, struc_emb, pro_seq_lens, device):
        struc_emb_new = None
        seq_len_1, seq_len_2 = 0, 0
        for i in range(sample_num):
            if i == 0:
                seq_len_1, seq_len_2 = 0, pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]         
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device) 
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)   
                struc_emb_new = modal2_emb_one.unsqueeze(0)                 
            else:
                seq_len_1, seq_len_2 = seq_len_2, seq_len_2+pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device)
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)
                struc_emb_new = torch.cat((struc_emb_new, modal2_emb_one.unsqueeze(0)), dim=0)
        struc_emb = struc_emb_new   
        return struc_emb

    def forward(self, drug_seq_feat, drug_struc_feat, target_seq_feat, target_struc_feat, pockets_len,):
        device = drug_seq_feat.device

        embeddings = []

        if self.use_drug_seq:
            drug_seq_emb = self.drug_seq_Linear(drug_seq_feat)  
            embeddings.append(drug_seq_emb.unsqueeze(1))  
        
        if self.use_drug_struc:
            drug_struc_emb = self.drug_struc_Linear(drug_struc_feat)  
            embeddings.append(drug_struc_emb)

        if embeddings:
            drug_emb = torch.cat(embeddings, dim=1)  
        else:
            raise ValueError("At least one drug feature must be used.")

        embeddings = []  

        if self.use_target_seq:

            target_seq_emb = self.target_seq_Linear(target_seq_feat)  # B*256
            embeddings.append(target_seq_emb.unsqueeze(1))  

        if self.use_target_struc:
            B = drug_seq_feat.size(0)
            max_pocket_len_batch = torch.max(pockets_len).item()
            struc_emb = self.target_struc(*target_struc_feat)  # seqs_len*512
            target_struc_emb = self.struc_data_format_change(B, max_pocket_len_batch, struc_emb, pockets_len, device)  # B*max_len+1*128
            target_struc_emb = self.target_struc_Linear(target_struc_emb)
            embeddings.append(target_struc_emb)
        if embeddings:
            target_emb = torch.cat(embeddings, dim=1)  # B*(max_len+1 or less)*256
        else:
            raise ValueError("At least one target feature must be used.")
        drug_emb = drug_emb.mean(1).squeeze(1)
        target_emb = target_emb.mean(1).squeeze(1)

        drug_target = torch.cat((drug_emb, target_emb),dim=1)

        affinity = self.affinity_layer(drug_target)

        return affinity.squeeze(-1)

    def __call__(self, data):
        drug_seq_feat,drug_struc_feat,target_seq_feat,target_struc_feat, gpu_split,device= data

        target_struc_feat = [target_struc_feat[i] for i in gpu_split]      
        pockets_len = [i.seq_len for i in target_struc_feat]
        pockets_len = torch.stack(pockets_len).squeeze(-1)

        struc_feat = torch_geometric.data.Batch.from_data_list(target_struc_feat)     
        h_V, h_E, edge_index, seq = (struc_feat.node_s.to(device), struc_feat.node_v.to(device)), (struc_feat.edge_s.to(device), struc_feat.edge_v.to(device)), struc_feat.edge_index.to(device), struc_feat.seq.to(device)
 
        struc_feat = (h_V, h_E, edge_index, seq)
        label = self.forward(drug_seq_feat,drug_struc_feat,target_seq_feat,struc_feat,pockets_len,)
        return label