import numpy as np
import torch
from time import perf_counter

def get_receptive_fields_dense(cur_neighbors, selected_node, adj, weighted_score):
    # t = perf_counter()
    receptive_vector_pre=((cur_neighbors)!=0)+0. ##+0 bool转int
    receptive_vector=((cur_neighbors+adj[selected_node])!=0)+0. ##+0 bool转int
    # receptive_vector = torch.where((cur_neighbors+adj[selected_node]) !=0, 1, 0)
    # print(perf_counter()-t)
    # t = perf_counter()
    # count= sum(receptive_vector[0]) #weighted_score.dot(receptive_vector) ##内积 eq9  #    assert count!=sum(receptive_vector)
    receptive_field_previous = weighted_score.dot(receptive_vector_pre)
    receptive_field = weighted_score.dot(receptive_vector)
    expand_ratio = (receptive_field - receptive_field_previous)/receptive_field
    current_score = weighted_score.dot((adj[selected_node]!=0)+0.)
    # print(perf_counter()-t)
    return receptive_field, expand_ratio, current_score

def get_receptive_fields_sparse(cur_neighbors, selected_node, adj):
    # t = perf_counter()
    # receptive_vector=((cur_neighbors+adj[selected_node])!=0)+0. ##+0 bool转int
    selected_node_vector = adj[selected_node].storage.col().tolist() ##当前节点的所有邻居
    receptive_vector = list(set(cur_neighbors + selected_node_vector))
    # receptive_vector = torch.where((cur_neighbors+adj[selected_node]) !=0, 1, 0)
    # print(perf_counter()-t)
    # t = perf_counter()
    # count= sum(receptive_vector[0]) #weighted_score.dot(receptive_vector) ##内积 eq9  #    assert count!=sum(receptive_vector)
    count = len(receptive_vector)
    count2 = len(selected_node_vector)
    expand_ratio = (count - len(cur_neighbors))/(count+1e-12)
    # print(perf_counter()-t)
    return count, count2, expand_ratio

def get_current_neighbors_dense(cur_nodes, adj):
    if np.array(cur_nodes).shape[0]==0:
        return torch.zeros(adj.shape[1]).to(adj.device)
    neighbors=(adj[list(cur_nodes)].sum(dim=0)!=0)+0  ##dim=0
    return neighbors

def get_current_neighbors_sparse(cur_nodes, adj):
    if np.array(cur_nodes).shape[0]==0:
        return []
    neighbors=list(set(adj[list(cur_nodes)].storage.col().tolist()))
    return neighbors

def get_max_nnd_node_dense(idx_used,high_score_nodes,adj, jaccard_score, weighted_score_B):   ###high_score_nodes == idx_avaliable_temp
    max_receptive_node = 0
    max_total_score = 0
    max_expand_ratio = 0
    # t = perf_counter()
    cur_neighbors=get_current_neighbors_dense(idx_used, adj)
    # print("time 2:", perf_counter()-t)
    # t = perf_counter()
    for node in high_score_nodes:  ###S集合
        receptive_field, expand_ratio, current_score = get_receptive_fields_dense(cur_neighbors,int(node), adj, weighted_score_B) ##eq10
        total_score = receptive_field + (1-jaccard_score[node])  #/adj.size(1)
        ## receptive_field: 0.1 0.001
        if total_score > max_total_score:
            max_total_score = total_score
            max_node_score = current_score
            max_receptive_node = node
            max_expand_ratio = expand_ratio
    # print("time 3:", perf_counter()-t)
    return max_receptive_node, max_total_score, max_node_score, max_expand_ratio

def get_max_nnd_node_sparse(idx_used,high_score_nodes,adj, jaccard_score):   ###high_score_nodes == idx_avaliable_temp
    max_receptive_node = 0
    max_total_score = 0
    max_expand_ratio = 0
    # t = perf_counter()
    cur_neighbors=get_current_neighbors_sparse(idx_used, adj)
    # print("time 2:", perf_counter()-t)
    # t = perf_counter()
    for node in high_score_nodes:  ###S集合
        receptive_field, current_score, expand_ratio = get_receptive_fields_sparse(cur_neighbors,int(node), adj) ##eq10
        total_score = receptive_field + (1-jaccard_score[node])  #/adj.size(1)
        ## receptive_field: 0.1 0.001
        if total_score > max_total_score:
            max_total_score = total_score
            max_node_score = current_score
            max_receptive_node = node
            max_expand_ratio = expand_ratio
    # print("time 3:", perf_counter()-t)
    return max_receptive_node, max_total_score, max_node_score, max_expand_ratio


def PPR(A, pr):
    pagerank_prob=pr  #0.85 -1   0.65best 68
    pr_prob = 1 - pagerank_prob
    A_hat   = A + torch.eye(A.size(0)).to(A.device)
    D       = torch.diag(torch.sum(A_hat,1))
    D       = D.inverse().sqrt()
    A_hat   = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)).to(A.device) - (1 - pr_prob) * A_hat).inverse())
    Pi = Pi.cpu()
    return Pi

def calc_ppr(adj_dict, k, pr, device):
    # ppr={}
    adj = adj_dict.clone()
    adj.storage._value = None
    adj = adj.to_dense()
    adj_T = adj.t()
    A = torch.zeros(adj.shape[0]+adj.shape[1], adj.shape[0]+adj.shape[1]).to(device)

    ###待优化
    # adj[adj!=0]=1
    # adj_T[adj_T!=0]=1
    if k[0]==k[-1]:  ###这个情况不存在，因为只选其他顶点类型
        A = adj
    ###待优化

    A[:adj.shape[0],adj.shape[0]:] = adj  ##[:4057, 4057:]
    A[adj.shape[0]:,:adj.shape[0]] = adj_T
    PPRM=PPR(A, pr)
    ppr=(PPRM[:adj.shape[0],adj.shape[0]:])
    return ppr