import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

import numpy as np
# from core.encoders import *
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
# from util_spgcl import get_info_ent, get_reliab_samples
import math
import time
import pdb

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / 0.05).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * 1 # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(3):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

#Compute entropy function
def get_info_ent(p):
    info_ent_p = torch.zeros(p.shape[0])
    for i in range(info_ent_p.shape[0]):
        info_ent_p[i] = - torch.sum(p[i]*torch.log(p[i]),dim=-1)
    return info_ent_p 

def get_reliab_samples(info_ent, ratio, pacing_type):
    cand_num = info_ent.shape[0]
    if pacing_type == "logar":
        # reliab_num = min(1, 1+0.1*math.log(ratio+exp(-10))) * cand_num
        reliab_num = (1+0.1*math.log(ratio+math.exp(-10))) * cand_num
    elif pacing_type == "poly1":
        reliab_num = (ratio) * cand_num
    elif pacing_type == "poly2":
        reliab_num = (ratio)**2 * cand_num
    elif pacing_type == "poly3":
        reliab_num = (ratio)**3 * cand_num

    reliab_num = int(reliab_num)
    # print(f"reliab_num:{reliab_num}")

    reliab_idx = torch.argsort(info_ent,dim=0)[:reliab_num] # Select reliab_num samples with the lowest entropy as trusted samples/trusted positive samples pair.
    # sample_num = len(info_ent)
    reliab_mask = torch.zeros(cand_num).bool().to(info_ent.device)
    reliab_mask[reliab_idx] = True
    return reliab_mask, reliab_idx

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, num_prot):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(num_prot)
        for i, k in enumerate(num_prot):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class spgcl(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, num_prot=0, temp=0.2):
        super(spgcl, self).__init__()
        self.tau = temp
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim))

        # prototype layer
        self.prototypes = None
        if isinstance(num_prot, list):
            self.prototypes = MultiPrototypes(self.embedding_dim, num_prot)
        elif num_prot > 0:
            self.prototypes = nn.Linear(self.embedding_dim, num_prot, bias=False)

        # self.local_d = FF(self.embedding_dim)
        # self.global_d = FF(self.embedding_dim)
        # # if self.prior:
        # #     self.prior_d = PriorDiscriminator(self.embedding_dim)

        # self.criterion = torch.nn.CrossEntropyLoss()
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(edge_index.get_device())

        y, M = self.encoder(x, edge_index, batch)
        # y = F.dropout(y, p=args.dropout_rate, training=self.training)
        # y = self.proj_head(y)

        # if self.l2norm:
        # y = F.normalize(y, dim=1)
        # m = torch.nn.ReLU()
        if self.prototypes is not None:
            return y, self.prototypes(y)
            # return y, self.prototypes(y)
        else:
            return y

    def get_reliab_mask(self, prot_scores, p1, p2, prot_assign1, prot_assign2, epoch_ratio, 
                            pacing_type, pos_div_threshold, neg_div_threshold, device):
        #--------------Compute the entropy of each sample based on prototypes and determine whether each sample is reliable.
        info_ent1 = get_info_ent(p1) #info_ent1 = get_info_ent(p1)
        info_ent2 = get_info_ent(p2) #info_ent2 = get_info_ent(p2)
        info_ent = torch.cat((info_ent1,info_ent2)).to(device)
        info_ent_avg = torch.mean(info_ent)
        prot_assign = torch.cat((prot_assign1, prot_assign2))

        # print(f"[info ent] --max: {torch.max(info_ent)}, --mean: {torch.mean(info_ent)}, --min: {torch.min(info_ent)}")

        node_num = p1.shape[0]
        # cand_num = 2* node_num
        
        # time1 = time.time()
        reliab_mask, reliab_idx = get_reliab_samples(info_ent, epoch_ratio, pacing_type)
        # print(f"reliab_mask: {time.time() - time1} seconds!")

        #=======================Class similarity is not a necessary condition for positive sample pairs, 
        #============as clustering may involve cases where two or more classes are effectively merged into one. 
        #======Therefore, negative sample pairs not only depend on class dissimilarity but also on the clustering probability distribution.
        # Positive sample pairs are considered reliable only when both views from the same anchor image are reliable.
        pos_mask_bothReliab = reliab_mask[:node_num] & reliab_mask[node_num:]
        # pos_mask_bothReliab = pos_mask_bothReliab.to(device)
        # Belonging to the same class.
        assign_mask = [prot_assign1[i]==prot_assign2[i] for i in range(node_num)] #assign_mask=[prot_assign1[i]==prot_assign2[i] for i in range(node_num)]
        assign_mask = torch.tensor(assign_mask).to(device)
        # The clustering probability distributions are very similar.
        mean_p12 = (p1 + p2)/2
        pos_div = 0.5*((p1 * (p1 / mean_p12).log())+(p2 * (p2 / mean_p12).log())).sum(dim=1)
        prob_mask = pos_div <= pos_div_threshold
        # print(f"[pos div] --max: {torch.max(pos_div)}, --mean: {torch.mean(pos_div)}, --min: {torch.min(pos_div)}")
        #=======================Necessary conditions for trustworthy sample pairs: Both samples are from the same anchor image, 
        #============both clusterings are trustworthy, and they belong to the same class, 
        #======or they belong to different classes but their clustering probability distributions are very similar.
        reliab_pos_mask = pos_mask_bothReliab & (assign_mask | prob_mask) # In this case, the length of reliab_pos_mask is only node_num.
        reliab_pos_idx = torch.nonzero(reliab_pos_mask, as_tuple=True)[0]
        reliab_pos_mask = torch.cat((reliab_pos_mask, reliab_pos_mask))
        reliab_pos_idx = torch.cat((reliab_pos_idx, reliab_pos_idx+node_num))
        # print(f"reliab_pos_mask: {time.time() - time1} seconds!")

        #======================Negative sample selection
        sample_num = 2*node_num

        reliab_num = len(reliab_idx)
        reliab_pos_num = len(reliab_pos_idx)
        row_idx = reliab_pos_idx.expand(reliab_num,-1).T.flatten(start_dim=0)
        col_idx = reliab_idx.expand(reliab_pos_num,-1).flatten(start_dim=0)

        # time1 = time.time()
        
        # Negative samples should have different class labels compared to positive sample pairs.
        row_cluster_idx = prot_assign[reliab_pos_idx]
        row_cluster_idx = row_cluster_idx.expand(reliab_num,-1).T.flatten(start_dim=0)
        col_cluster_idx = prot_assign[reliab_idx]
        col_cluster_idx = col_cluster_idx.expand(reliab_pos_num,-1).flatten(start_dim=0)

        reliab_pos_idx_dual = (reliab_pos_idx + node_num) % sample_num # reliab_pos_idx_dual:The indices of the corresponding nodes in the augmented view of nodes from the same anchor image.
        row_cluster_idx_dual = prot_assign[reliab_pos_idx_dual]
        row_cluster_idx_dual = row_cluster_idx_dual.expand(reliab_num,-1).T.flatten(start_dim=0)

        cluster_mask = (row_cluster_idx != col_cluster_idx) & (row_cluster_idx_dual != col_cluster_idx) # Negative samples should have different class labels compared to both samples in the positive sample pair.

        # Compute the Jensen-Shannon divergence between each pair of trustworthy positive samples and each pair of trustworthy samples.
        row_scores = torch.cat((prot_scores[row_idx, row_cluster_idx], prot_scores[row_idx, col_cluster_idx])).reshape(2,-1).T
        col_scores = torch.cat((prot_scores[col_idx, row_cluster_idx], prot_scores[col_idx, col_cluster_idx])).reshape(2,-1).T
        row_probs = F.softmax(row_scores / self.tau,dim=1)
        col_probs = F.softmax(col_scores / self.tau,dim=1)
        row_col_probs = (row_probs + col_probs)/2
        res_div = 0.5*((row_probs * (row_probs / row_col_probs).log())+(col_probs * (col_probs / row_col_probs).log())).sum(dim=1)
        res_div = res_div * cluster_mask

        neg_div = torch.zeros(sample_num,sample_num).to(device)
        neg_div[row_idx, col_idx] = res_div#.to(device)
        
        # pdb.set_trace()
        # Negative sample pairs must come from different anchor images.
        anchor_mask = torch.ones_like(neg_div).to(device)
        # anchor_mask = anchor_mask - torch.diag(anchor_mask) - torch.diag(anchor_mask, node_num) - torch.diag(anchor_mask, -node_num)
        _ = torch.diagonal(anchor_mask, 0).zero_()
        _ = torch.diagonal(anchor_mask, node_num).zero_()
        _ = torch.diagonal(anchor_mask, -node_num).zero_()
        neg_div = neg_div * anchor_mask

        neg_div_ = neg_div > 0
        # print(f"[neg div] --max: {torch.max(neg_div[neg_div_])}, --mean: {torch.sum(pos_div)/torch.sum(neg_div_)}, --min: {torch.min(neg_div[neg_div_])}")

        # The Jensen-Shannon divergence for negative sample pairs must be higher than a certain threshold.
        reliab_neg_mask = neg_div >= neg_div_threshold

        # print(f"get_neg_div: {time.time() - time1} seconds!")

        L_cluster_compactness = torch.mean(info_ent)

        # pdb.set_trace()
        return reliab_neg_mask, reliab_pos_mask, reliab_pos_idx, L_cluster_compactness
    
    def semi_clustering_consist_loss(self, prot_scores1, prot_scores2, q1, q2):
        L_cluster_consistency = 0
        L_cluster_consistency = -0.5 * (torch.mean(torch.sum(q1 * F.log_softmax(prot_scores2 / self.tau,dim=1),dim=1)) + torch.mean(torch.sum(q2 * F.log_softmax(prot_scores1 / self.tau,dim=1),dim=1))) 
        # L_cluster_consistency -= torch.mean(torch.sum(q1 * F.log_softmax(prot_scores2 / self.tau,dim=1),dim=1)) 
        # L_cluster_consistency -= torch.mean(torch.sum(q2 * F.log_softmax(prot_scores1 / self.tau,dim=1),dim=1))

        # pdb.set_trace()
        return L_cluster_consistency

    def semi_contrastive_loss(self, z, reliab_neg_mask, reliab_pos_idx, device):
        # Calculate the contrastive loss, i.e., L_contrastive.
        # z: z1||z2
        # reliab_pos_idx = torch.cat(reliab_pos_idx)
        # reliab_pos_num = reliab_pos_idx.size(0)//2     
        sample_num = z.shape[0] # sample_num = node_num *2
        node_num = sample_num // 2
        reliab_pos_idx_dual = (reliab_pos_idx + node_num) % sample_num # reliab_pos_idx_dual:The indices of corresponding nodes in the augmented view from the same anchor image.

        # pdb.set_trace() 
        reliab_neg_mask = reliab_neg_mask[reliab_pos_idx].to(device) # reliab_pos_num * 2N
        # z= F.normalize(z, dim=1) # Normalization has already been performed in the encoder, so it is omitted here.
        z_pos = z[reliab_pos_idx]
        #--1)Evaluation based on the distances of all sample pairs.
        # Calculation of the pairwise sample dissimilarity matrix sample_dist.
        # import pdb
        # pdb.set_trace()
        sample_dist = 1 - torch.mm(z_pos, z.t().contiguous()) # reliab_pos_num * 2N
        # Calculate the mean and standard deviation of each row individually.
        mu=torch.mean(sample_dist,dim=1)
        std = torch.std(sample_dist,dim=1)
        # Compute the weight matrix reweight for negative samples.

        # pdb.set_trace()
        
        reweight = torch.exp(-torch.pow(sample_dist - mu.unsqueeze(1), 2)/(2 * torch.pow(std,2).unsqueeze(1))).to(device)
        reweight= reweight * reliab_neg_mask
        #--2)Evaluate based on trustworthy negative sample pairs.
        
        reweight_normalize = (sample_num - 2) / (torch.sum(reweight,dim=1) + math.exp(-10)) ########Negative sample balancing
        reweight = reweight * reweight_normalize.reshape(reweight.shape[0],1)
        # Compute the similarity between each pair of negative samples.
        sim_matrix  = torch.exp(torch.mm(z_pos, z.t().contiguous()) / self.tau).to(device)
        # Weighting the similarity matrix.
        sim_matrix = sim_matrix * reweight #(sim_matrix * reweight)*(reliab_neg_mask)
        
        # Compute the similarity of positive sample pairs.
        pos_sim = torch.exp(torch.sum(z_pos * z[reliab_pos_idx_dual],dim=-1)/self.tau).to(device)
        
        # Calculate the contrastive loss.
        # L_contrastive = -(torch.log((pos_sim /( pos_sim + sim_matrix.sum(dim=-1) + exp(-10)))[reliab_pos_idx])).mean()
        L_contrastive = -torch.log((pos_sim /( pos_sim + sim_matrix.sum(dim=-1) + math.exp(-10))))

        
        # Exclude parts without trustworthy negative samples from the final loss calculation.
        reliab_neg_nums = torch.sum(reliab_neg_mask, dim=1)
        nonzero_neg_idx = torch.nonzero(reliab_neg_nums, as_tuple=True)[0]
        L_contrastive = L_contrastive[nonzero_neg_idx].mean()
        if torch.isnan(L_contrastive):
            L_contrastive[L_contrastive != L_contrastive] = 0
        
        return L_contrastive

    def loss(self, z1, z2, prot_scores1, prot_scores2, beta, 
        pacing_type, pos_div_threshold, neg_div_threshold, epoch, epochs):
       
        device = z1.device

        epoch_ratio = epoch/epochs
        node_num = z1.shape[0]
        z = torch.cat((z1, z2))

        prot_scores = torch.cat((prot_scores1, prot_scores2))
 
        p1 = F.softmax(prot_scores1 / self.tau,dim=1)
        p2 = F.softmax(prot_scores2 / self.tau,dim=1)
        prot_assign1 = torch.argmax(p1,dim=1)
        prot_assign2 = torch.argmax(p2,dim=1)
        prot_assign = torch.cat((prot_assign1,prot_assign2))
        prot_scores = torch.cat((prot_scores1,prot_scores2))

        with torch.no_grad():
            q1 = distributed_sinkhorn(prot_scores1)[-node_num:]
            q2 = distributed_sinkhorn(prot_scores2)[-node_num:]

        # Compute the Sinkhorn divergence.
        ## Filtering of trustworthy samples, trustworthy positive sample pairs, and trustworthy negative sample pairs.
        reliab_neg_mask, reliab_pos_mask, reliab_pos_idx, L_cluster_compactness = self.get_reliab_mask(prot_scores, p1, p2, prot_assign1, prot_assign2, epoch_ratio, 
                                                                                                        pacing_type, pos_div_threshold, neg_div_threshold, device)
        # print("===================== get_reliab_mask done! ====================")

        # time1 = time.time()
        L_cluster_consistency = self.semi_clustering_consist_loss(prot_scores1, prot_scores2, q1, q2).to(device)
        # print(f"-------L_cluster_consistency: {time.time() - time1} seconds")
        
        # time1 = time.time()
        L_contrastive = self.semi_contrastive_loss(z, reliab_neg_mask, reliab_pos_idx, device).to(device)
        # print(f"-------L_contrastive: {time.time() - time1} seconds")

        loss = 0
        loss += L_contrastive + beta * L_cluster_consistency + beta * L_cluster_compactness
        return loss



class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)

        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
    

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        old_x = x
        x = self.x_embedding1(old_x[:,0]) + self.x_embedding2(old_x[:,1])
        # print("self.x_embedding1(x[:,0]): {}".format(self.x_embedding1(old_x[:,0])))
        # print("self.x_embedding2(x[:,0]): {}".format(self.x_embedding2(old_x[:,1])))

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

class GNN_Virtualnode(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

        self.linear_pred_atoms = torch.nn.Linear(emb_dim, 121)
        self.linear_pred_bonds = torch.nn.Linear(emb_dim, 6)

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [x]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))

class GNN_graphpred_Virtualnode(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN_Virtualnode(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)


        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr, batch)
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # return self.graph_pred_linear(self.pool(node_representation, batch) + virtualnode_embedding)
        return self.gnn.linear_pred_atoms(self.pool(node_representation, batch) + virtualnode_embedding)

        # return self.graph_pred_linear(self.)

if __name__ == "__main__":
    pass

