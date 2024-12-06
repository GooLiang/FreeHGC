import numpy as np
import torch
from time import perf_counter

def get_receptive_fields_dense(cur_neighbors, selected_node, adj, weighted_score):
    # t = perf_counter()
    receptive_vector=((cur_neighbors+adj[selected_node])!=0)+0. ##+0 bool转int
    # receptive_vector = torch.where((cur_neighbors+adj[selected_node]) !=0, 1, 0)
    # print(perf_counter()-t)
    # t = perf_counter()
    # count= sum(receptive_vector[0]) #weighted_score.dot(receptive_vector) ##内积 eq9  #    assert count!=sum(receptive_vector)
    count = weighted_score.dot(receptive_vector)
    # print(perf_counter()-t)
    return count

def get_current_neighbors_dense(cur_nodes, adj):
    if np.array(cur_nodes).shape[0]==0:
        return 0
    neighbors=(adj[list(cur_nodes)].sum(dim=0)!=0)+0  ##dim=0
    return neighbors

def get_max_nnd_node_dense(idx_used,high_score_nodes,adj, jaccard_score, weighted_score_B):   ###high_score_nodes == idx_avaliable_temp
    max_receptive_node = 0
    max_total_score = 0
    # t = perf_counter()
    cur_neighbors=get_current_neighbors_dense(idx_used, adj)
    # print("time 2:", perf_counter()-t)
    # t = perf_counter()
    for node in high_score_nodes:  ###S集合
        receptive_field=get_receptive_fields_dense(cur_neighbors,int(node), adj, weighted_score_B) ##eq10
        total_score = 0.01 * receptive_field/adj.size(1) + (1-jaccard_score[node])  #distance_score/adj.size(0)
        if total_score > max_total_score:
            max_total_score = total_score
            max_receptive_node = node
    # print("time 3:", perf_counter()-t)
    return max_receptive_node, max_total_score


def PPR(A):
    pagerank_prob=0.95  #0.85 -1   0.65best 68
    pr_prob = 1 - pagerank_prob
    A_hat   = A + torch.eye(A.size(0)).to(A.device)
    D       = torch.diag(torch.sum(A_hat,1))
    D       = D.inverse().sqrt()
    A_hat   = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)).to(A.device) - (1 - pr_prob) * A_hat).inverse())
    Pi = Pi.cpu()
    return Pi

def calc_ppr(adj_dict, k, device):
    # ppr={}
    adj = adj_dict.clone().to_dense()
    adj_T = adj.t()
    A = torch.zeros(adj.shape[0]+adj.shape[1], adj.shape[0]+adj.shape[1]).to(device)
    A[:adj.shape[0],adj.shape[0]:] = adj  ##[:4057, 4057:]
    A[adj.shape[0]:,:adj.shape[0]] = adj_T
    PPRM=PPR(A)
    ppr=(PPRM[:adj.shape[0],adj.shape[0]:])
    return ppr
