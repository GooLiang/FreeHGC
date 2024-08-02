from torch import optim
import os
import numpy as np
import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import scipy.sparse as sp
import gc
import torch.nn.functional as F
import time

from torch_sparse import SparseTensor
from torch_sparse import remove_diag, to_torch_sparse, set_diag
from copy import deepcopy
from data_loader_hgb import data_loader


def square_feat_map(z, c=1):
    from sklearn.preprocessing import PolynomialFeatures
    polf = PolynomialFeatures(include_bias=True)
    x = polf.fit_transform(z)
    coefs = np.ones(len(polf.powers_))
    coefs[0] = c
    coefs[(polf.powers_ == 1).sum(1) == 2] = np.sqrt(2)
    coefs[(polf.powers_ == 1).sum(1) == 1] = np.sqrt(2*c) 
    return x * coefs

def run_cluster(H, c, k):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    if k == 0:
        k = 1
    H = StandardScaler(with_std=False).fit_transform(H) ##传入特征
    svd = TruncatedSVD(k)
    svd.fit(H.T)
    U = svd.components_.T

    Z = square_feat_map(U, c=c)
    r = Z.sum(0)
    D = Z @ r 
    Z_hat = Z / D[:,None]**.5
    
    svd = TruncatedSVD(k+1)
    svd.fit(Z_hat.T)
    Q = svd.components_.T[:,1:]
    P = KMeans(k).fit_predict(Q)  
    return P, Q

def list_to_sp_mat(li, n):
    data = [x[2] for x in li]
    i = [x[0] for x in li]
    j = [x[1] for x in li]
    return sp.coo_matrix((data, (i,j)), shape=(n, n)).tocsr()

def adj_mask_sparse(A, B):
    assert A.size(0)==B.size(0) and A.size(1)==B.size(1)
    rowA, colA, _ = A.coo()
    rowB, colB, _ = B.coo()
    indexA = rowA * A.size(1) + colA
    indexB = rowB * A.size(1) + colB
    ###针对freebase mask的问题，但是没有完全解决，A.masked_select_nnz会OOM for mag###
    # chunk_size = 100000
    # mask_list = []
    # for i in range(0, len(indexA), chunk_size):
    #     chunk = indexA[i:i+chunk_size]
    #     mask = ~torch.isin(chunk, indexB)
    #     mask_list.append(mask)
    # mask = torch.cat(mask_list)
    ######################################################################
    nnz_mask = ~(torch.isin(indexA, indexB))
    A = A.masked_select_nnz(nnz_mask)
    return A

def adj_mask(a, b, rede):
    # mask = b.to_dense().gt(0)
    # a = a.to_dense().masked_fill(mask, 0)
    # mask = b.to_dense()
    # a = a.to_dense()*(mask<=0).float()
    a_nnz = a.nnz()
    b_nnz = b.nnz()
    # print(a_nnz,b_nnz)
    mask = b.to(torch.bool).to_dense()
    a = a.to_dense()
    a = torch.where(mask, 0, a)
    a = a.to_sparse().coalesce()
    a = SparseTensor(row=a.indices()[0], col=a.indices()[1], value = a.values(), sparse_sizes=a.size())
    # print("new", a.nnz())
    #print("de-redundancy", ((a_nnz-a.nnz())/a_nnz)*100)
    if a_nnz!=0:
        rede = rede + ((a_nnz-a.nnz())/a_nnz)*100
    ###第三种实际速度应该是最快的###
    #a = torch.masked_fill(a, mask, 0) 跟torch.where速度基本一样
    return a, rede

def adj_mask_freebase(a, b):
    mask = b.to(torch.bool).to_dense()
    a = a.to_dense()
    a = torch.where(mask, 0, a)
    ###第三种实际速度应该是最快的###
    #a = torch.masked_fill(a, mask, 0) 跟torch.where速度基本一样
    #### for Freebase case############
    a = a.to_sparse()
    a = SparseTensor(row=a.indices()[0], col=a.indices()[1], value = a.values(), sparse_sizes=a.size())
    ##################################
    return a

def generate_mask_subgraph(s):
    def dfs(graph, node, path, paths):
        path.append(node)
        if node == len(graph) - 1:
            paths.append(path[:])
        else:
            for neighbor in graph[node]:
                dfs(graph, neighbor, path, paths)
        path.pop()
    # 构建图
    graph = [[] for _ in range(len(s))]
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            graph[i].append(j)

    # 进行 DFS 遍历
    paths = []
    dfs(graph, 0, [], paths)
    paths_list = set()
    # 输出结果
    for path in paths:
        if len(path)<len(s):
            paths_list.add(''.join(s[i] for i in path))
    #print("generate subgraph to mask", paths_list)
    return paths_list

def hg_propagate(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False):
    for hop in range(1, max_hops):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)

            for k in list(new_g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    new_g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g


def hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_types, num_hops, max_length, extra_metapath, threshold_metalen, prop_device, enhance, prop_feats=False, echo=True):
    store_device = 'cpu'
    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]
    features_list_dict = deepcopy(features_list_dict_cp)
    adj_dict = {k: v.clone().to(prop_device) for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for k,v in features_list_dict.items():
        features_list_dict[k] = v.to(prop_device)

    for k,v in adj_dict.items():
        #print('Generating ...', k)
        features_list_dict[k] = (v.to(prop_device) @ features_list_dict[k[-1]].to(prop_device))
        ############去掉@ features_list_dict[k[-1]]保持与其他metapath一样的sparse格式。
    # compute k-hop feature
    rede_buffer = []
    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in adj_dict.items(): ###遍历所有metapath，每轮都会将新hop的metapaths更新到adjk_dict
            #metapath_types = list(rtype_r)
            if len(rtype_r) == hop:   ###只计算metapath==hop的
                dtype_r, stype_r = rtype_r[0], rtype_r[-1] #stype, _, dtype = g.to_canonical_etype(etype)
                for rtype_l, adj_l in adjs_g.items():   ### 固定的：dict_keys(['PP', 'PA', 'AP', 'PC', 'CP'])
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:  #rtype_l @ rtype_r
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):   ###如果hop为3，避免生成APPP这种类型
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)    ##每次左乘A  两个sparse  都到gpu上可以做
                                features_list_dict[name] = new_adjs[name].to(prop_device).matmul(features_list_dict[stype_r])  #稀疏乘密集  rtype_r[-1] == stype_r   features_list_dict需要转成稀疏
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = (adj_l.matmul(adj_r.to(prop_device)))  #.to('cpu')    ##每次左乘A  两个sparse
                                    features_list_dict[name] = (new_adjs[name].to(prop_device).matmul(features_list_dict[stype_r]))   #.to('cpu')  #rtype_r[-1] == stype_r   features_list_dict需要转成稀疏
                                    #features_list_dict[name] = (new_adjs[name].to(prop_device)).to('cpu')  #rtype_r[-1] == stype_r   features_list_dict需要转成稀疏

                        else:
                            if echo: print(f'Warning: {name} already exists')
        adj_dict.update(new_adjs)
    removes = []
    for k in features_list_dict.keys():
        if k[0] == tgt_types[0]: continue
        else:
            removes.append(k)
    for k in removes:
        features_list_dict.pop(k)
    if echo and len(removes): print('remove features', removes)

    removes_adj = []
    for k in adj_dict.keys():
        if k[0] == tgt_types[0]: continue
        else:
            removes_adj.append(k)
    for k in removes_adj:
        adj_dict.pop(k)
    if echo and len(removes_adj): print('remove adjs', removes)

    extra_features_buffer = {}

    del new_adjs

    gc.collect()
    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()
    return features_list_dict, adj_dict, extra_features_buffer

def hg_propagate_sparse_pyg_freebase(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_device, enhance, prop_feats=False, echo=False):
    store_device = 'cpu'
    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone().to(prop_device) for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}
    count = 0
    hop = 2
    new_adjs = {}
    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items(): ###遍历所有metapath
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]  ###拆分metapath
                for rtype_l, adj_l in adjs_g.items():   ### 聚合所有的stype
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)    ##每次左乘A
                            else:
                                with torch.no_grad():
                                    #new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device))#.to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)
    removes = []
    for k in label_feats.keys():
        metapath_types = list(k)
        if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
        if len(metapath_types) <= hop:
            removes.append(k)
    for k in removes:
        label_feats.pop(k)
    if echo and len(removes): print('remove', removes)

    extra_features_buffer = {}

    del new_adjs
    gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    label_feats_new = {}
    for k in list(label_feats.keys()):
        label_feats_new[k] = label_feats.pop(k).to(store_device)

    return label_feats_new, extra_features_buffer


def hg_propagate_sparse_pyg_mask(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone().to(prop_device) for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}
    for k in adjs.keys():
        print('Generating ...', k)

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items(): ###遍历所有metapath
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]  ###拆分metapath
                for rtype_l, adj_l in adjs_g.items():   ### 聚合所有的stype
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                                if hop >= 1:
                                    print("label mask", name)
                                    mask_adj_list = generate_mask_subgraph(name)
                                    for mask in mask_adj_list:
                                        print(mask)
                                        if mask in label_feats:
                                            new_adjs[name] = adj_mask_sparse(new_adjs[name], label_feats[mask])  ##这里返回的是sparse
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device))#.to(store_device)
                                    if hop >= 1:
                                        print("label mask", name)
                                        mask_adj_list = generate_mask_subgraph(name)
                                        for mask in mask_adj_list:
                                            print(mask)
                                            if mask in label_feats:
                                                new_adjs[name] = adj_mask_sparse(new_adjs[name], label_feats[mask].to(prop_device)) ##这里返回的是sparse
                                        new_adjs[name] = new_adjs[name].to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats

def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}
    for k in adjs.keys():
        print('Generating ...', k)

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items(): ###遍历所有metapath
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]  ###拆分metapath
                for rtype_l, adj_l in adjs_g.items():   ### 聚合所有的stype
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats

def load_dataset(args):
    dl = data_loader(f'/home/public/lyx/SeHGNN_new/SeHGNN/data/{args.dataset}/{args.dataset}')
    # use one-hot index vectors for nods with no attributes
    # === feats ===
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            features_list.append(torch.FloatTensor(th))
    # if args.tf_idf:
    #     from sklearn.feature_extraction.text import TfidfTransformer
    #     features_list = [torch.FloatTensor(TfidfTransformer(norm='l2').fit_transform(features).todense()) for features in features_list]
    # else:
    #     from sklearn.preprocessing import normalize
    #     features_list = [torch.FloatTensor(normalize(features, 'l2')) for features in features_list]
        
    idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

    val_ratio = 0.2
    train_nid = np.nonzero(dl.labels_train['mask'])[0]   ###统计label不为0的
    np.random.shuffle(train_nid)  ###每次都是随种子打乱的train_nid
    split = int(train_nid.shape[0]*val_ratio)
    val_nid = train_nid[:split]
    train_nid = train_nid[split:]
    train_nid = np.sort(train_nid)
    val_nid = np.sort(val_nid)
    test_nid = np.nonzero(dl.labels_test['mask'])[0]
    test_nid_full = np.nonzero(dl.labels_test_full['mask'])[0]

    init_labels[train_nid] = dl.labels_train['data'][train_nid]
    init_labels[val_nid] = dl.labels_train['data'][val_nid]
    init_labels[test_nid] = dl.labels_test['data'][test_nid]
    if args.dataset != 'IMDB':
        init_labels = init_labels.argmax(axis=1)

    print(len(train_nid), len(val_nid), len(test_nid), len(test_nid_full))
    init_labels = torch.LongTensor(init_labels)

    # === adjs ==
    # print(dl.nodes['attr'])
    # for k, v in dl.nodes['attr'].items():
    #     if v is None: print('none')
    #     else: print(v.shape)
    adjs = [] if args.dataset != 'Freebase' else {}
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        #adj = torch.sparse.FloatTensor(indices, values, shape)
        if args.dataset == 'Freebase':
            name = f'{dst_type_idx}{src_type_idx}'
            assert name not in adjs
            adjs[name] = adj
        else:
            adjs.append(adj)
            print(adj)
            
    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs
        total_edges = AP.nnz() + PT.nnz() + PV.nnz()
        node_type_nodes = {}
        node_type_nodes['A'] = A.shape[0]
        node_type_nodes['P'] = P.shape[0]
        node_type_nodes['T'] = T.shape[0]
        node_type_nodes['V'] = V.shape[0]

        edge_type_ratio = {}
        edge_type_ratio['AP'] = AP.nnz()/total_edges
        edge_type_ratio['PT'] = PT.nnz()/total_edges
        edge_type_ratio['PV'] = PV.nnz()/total_edges

        features_list_dict = {}
        features_list_dict['A'] = A
        features_list_dict['P'] = P
        features_list_dict['T'] = T
        features_list_dict['V'] = V
        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('T', 'T-P', 'P'),
            ('V', 'V-P', 'P'),
            ('P', 'P-T', 'T'),
            ('P', 'P-V', 'V'),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)
        g.nodes['A'].data['A'] = A
        g.nodes['P'].data['P'] = P
        g.nodes['T'].data['T'] = T
        g.nodes['V'].data['V'] = V
        ########### test cos########################
    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1) # each movie has single director
        edge_type_ratio = 0
        node_type_nodes = {}
        node_type_nodes['M'] = M.shape[0]
        node_type_nodes['D'] = D.shape[0]
        node_type_nodes['A'] = A.shape[0]
        node_type_nodes['K'] = K.shape[0]
        
        features_list_dict = {}
        features_list_dict['M'] = M
        features_list_dict['D'] = D
        features_list_dict['A'] = A
        features_list_dict['K'] = K
        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('D', 'D-M', 'M'),
            ('M', 'M-D', 'D'),
            ('A', 'A-M', 'M'),
            ('M', 'M-A', 'A'),
            ('K', 'K-M', 'M'),
            ('M', 'M-K', 'K'),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        g.nodes['M'].data['M'] = M
        g.nodes['D'].data['D'] = D
        g.nodes['A'].data['A'] = A
        if args.num_hops > 2:
            g.nodes['K'].data['K'] = K

    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        P, A, C, K = features_list
        features_list_dict = {}

        features_list_dict['P'] = P
        features_list_dict['A'] = A
        features_list_dict['C'] = C
        if args.ACM_keep_F:
            features_list_dict['K'] = K
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())  ##设置为对称矩阵 PP.is_symmetric()
        PP = PP.coalesce()  ###这里作用是去重
        PP = PP.set_diag()  ###对角线全部置1  ??
        total_edges = PP.nnz() + PA.nnz() + PC.nnz() + PK.nnz()
        node_type_nodes = {}
        node_type_nodes['P'] = P.shape[0]
        node_type_nodes['A'] = A.shape[0]
        node_type_nodes['C'] = C.shape[0]
        if args.ACM_keep_F:
            node_type_nodes['K'] = K.shape[0]

        edge_type_ratio = {}
        edge_type_ratio['PP'] = PP.nnz()/total_edges
        edge_type_ratio['PA'] = PA.nnz()/total_edges
        edge_type_ratio['PC'] = PC.nnz()/total_edges
        if args.ACM_keep_F:
            edge_type_ratio['PK'] = PK.nnz()/total_edges

        adjs = [PP] + adjs[2:]

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('A', 'A-P', 'P'),
            ('P', 'P-A', 'A'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
        ]
        if args.ACM_keep_F:
            etypes += [
                ('K', 'K-P', 'P'),
                ('P', 'P-K', 'K'),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        g = dgl.heterograph(new_edges)
                    
        g.nodes['P'].data['P'] = P # [3025, 1902]
        g.nodes['A'].data['A'] = A # [5959, 1902]
        g.nodes['C'].data['C'] = C # [56, 1902]
        if args.ACM_keep_F:
            g.nodes['K'].data['K'] = K # [1902, 1902]

    elif args.dataset == 'Freebase':
        # 0*: 40402  2/4/7 <-- 0 <-- 0/1/3/5/6
        #  1: 19427  all <-- 1
        #  2: 82351  4/6/7 <-- 2 <-- 0/1/2/3/5
        #  3: 1025   0/2/4/6/7 <-- 3 <-- 1/3/5
        #  4: 17641  4 <-- all
        #  5: 9368   0/2/3/4/6/7 <-- 5 <-- 1/5
        #  6: 2731   0/4 <-- 6 <-- 1/2/3/5/6/7
        #  7: 7153   4/6 <-- 7 <-- 0/1/2/3/5/7

        _0, _1, _2, _3, _4, _5, _6, _7 = features_list
        features_list_dict = {}
        # features_list_dict['0'] = _0
        # features_list_dict['1'] = _1
        # features_list_dict['2'] = _2
        # features_list_dict['3'] = _3
        # features_list_dict['4'] = _4
        # features_list_dict['5'] = _5
        # features_list_dict['6'] = _6
        # features_list_dict['7'] = _7
        
        _0_dim = _0.shape[1]
        _1_dim = _1.shape[1]
        _2_dim = 128
        __2_dim = _2.shape[1]
        _3_dim = _3.shape[1]
        _4_dim = _4.shape[1]
        _5_dim = _5.shape[1]
        _6_dim = _6.shape[1]
        _7_dim = _7.shape[1]
        if _2_dim < _0_dim:
            print(f"Randomly project paper feature from dimension {_0_dim} to {_2_dim}")
            rand_weight_02 = torch.Tensor(_0_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_12 = torch.Tensor(_1_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_22 = torch.Tensor(__2_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_32 = torch.Tensor(_3_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_42 = torch.Tensor(_4_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_52 = torch.Tensor(_5_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_62 = torch.Tensor(_6_dim, _2_dim).uniform_(-0.5, 0.5)
            rand_weight_72 = torch.Tensor(_7_dim, _2_dim).uniform_(-0.5, 0.5)

            # _2_feat = g.nodes['2'].data.pop('2')
            # g.nodes['2'].data['feat'] = _2_feat.to(device)
            features_list_dict['0'] = torch.matmul(_0, rand_weight_02)
            features_list_dict['1'] = torch.matmul(_1, rand_weight_12)
            features_list_dict['2'] = torch.matmul(_2, rand_weight_22)
            features_list_dict['3'] = torch.matmul(_3, rand_weight_32)
            features_list_dict['4'] = torch.matmul(_4, rand_weight_42)
            features_list_dict['5'] = torch.matmul(_5, rand_weight_52)
            features_list_dict['6'] = torch.matmul(_6, rand_weight_62)
            features_list_dict['7'] = torch.matmul(_7, rand_weight_72)
            
        node_type_nodes = {}
        node_type_nodes['0'] = _0.shape[0]
        node_type_nodes['1'] = _1.shape[0]
        node_type_nodes['2'] = _2.shape[0]
        node_type_nodes['3'] = _3.shape[0]
        node_type_nodes['4'] = _4.shape[0]
        node_type_nodes['5'] = _5.shape[0]
        node_type_nodes['6'] = _6.shape[0]
        node_type_nodes['7'] = _7.shape[0]
        
        adjs['00'] = adjs['00'].to_symmetric()
        g = None
        edge_type_ratio = {}
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
    elif args.dataset == 'ACM':
        if args.ACM_keep_F:
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP, 'PK':PK, 'KP':KP}
        else:
            adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
    elif args.dataset == 'Freebase':
        new_adjs = {}
        for rtype, adj in adjs.items():
            dtype, stype = rtype
            if dtype != stype:
                new_name = f'{stype}{dtype}'
                assert new_name not in adjs
                new_adjs[new_name] = adj.t()    ###为每个类型添加对称关系
        adjs.update(new_adjs)  ### adjs + new_adjs
        g = None
    else:
        assert 0

    return g, adjs, features_list_dict, node_type_nodes, edge_type_ratio, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full
