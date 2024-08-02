from torch import optim
import os
import numpy as np
import torch
import dgl
import dgl.function as fn
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag, spspmm
from utils_ogbn import *
from copy import deepcopy

def hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_types, num_hops, max_length, extra_metapath, prop_device, prop_feats=False, echo=True):
    store_device = 'cpu'
    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]
    features_list_dict = deepcopy(features_list_dict_cp)
    adj_dict = {k: v.clone().to(store_device) for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(store_device) for k, v in adjs.items()}

    for k,v in features_list_dict.items():
        features_list_dict[k] = v.to(prop_device)

    for k,v in adj_dict.items():
        #print('Generating ...', k)
        features_list_dict[k] = (v.to(prop_device) @ features_list_dict[k[-1]].to(prop_device)).to(store_device)
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
                                    if name != 'PFP': ###bug with sparse matmul, for fairness, all the baseline use this.
                                        new_adjs[name] = (adj_l.to(prop_device).matmul(adj_r.to(prop_device))).to(store_device)  #.to('cpu')    ##每次左乘A  两个sparse
                                        features_list_dict[name] = (new_adjs[name].to(prop_device).matmul(features_list_dict[stype_r].to(prop_device))).to(store_device)   #.to('cpu')  #rtype_r[-1] == stype_r   features_list_dict需要转成稀疏
                                        #features_list_dict[name] = (new_adjs[name].to(prop_device)).to('cpu')  #rtype_r[-1] == stype_r   features_list_dict需要转成稀疏
                                        torch.cuda.empty_cache()
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
    # del adj_dict
    gc.collect()
    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()
    return features_list_dict, adj_dict, extra_features_buffer

def hg_propagate_dgl(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False):
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

###################################################
def load_dataset(args):
    if args.dataset == 'ogbn-mag':
        # train/val/test 629571/64879/41939
        return load_mag(args)
    if args.dataset == 'aminer':
        # train/val/test 629571/64879/41939
        return load_aminer(args)
    else:
        assert 0
        

import torch_geometric.transforms as T 
from torch_geometric.datasets import AMiner
def load_aminer(args):
    root = '/home/public/lyx/hgcond-main/datahetero'
    dataset = AMiner(root+'/AMiner', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data['author'].x = torch.load(root+'/AMiner/raw/author.tensor')
    data['paper'].x = torch.load(root+'/AMiner/raw/paper.tensor')
    data['venue'].x = torch.load(root+'/AMiner/raw/venue.tensor')

    torch.manual_seed(1)
    index_labeled_rand = data['author'].y_index[torch.randperm(data['author'].y_index.shape[0])]
    y = torch.ones(data['author'].x.shape[0], dtype=torch.long)*-1
    y[data['author'].y_index] = data['author'].y
    init_labels = y
    n = index_labeled_rand.shape[0]
    train_nid = index_labeled_rand[:int(n*0.3)]
    val_nid = index_labeled_rand[int(n*0.3):int(n*0.6)]
    test_nid = index_labeled_rand[int(n*0.6):]


    adjs = []
    for i, etype in enumerate(data.edge_types):
        key = etype[2][0].upper() + etype[0][0].upper()
        adj = data[etype].pop('adj_t')
        adjs.append(adj)
        print(key)
        
    # etypes = [ # src->tgt
    #     ('A', 'A-P', 'P'),
    #     ('P', 'P-A', 'A'),
    #     ('V', 'V-P', 'P'),
    #     ('P', 'P-V', 'V'),
    # ]
    new_g = False
    AP, PA, VP, PV = adjs
    num_nodes = AP.size(0)
    adjs = {'AP': AP, 'PA': PA, 'VP': VP, 'PV': PV}
    node_type_nodes = {}
    node_type_nodes['A'] = data['author'].x.shape[0]
    node_type_nodes['P'] = data['paper'].x.shape[0]
    node_type_nodes['V'] = data['venue'].x.shape[0]

    features_list_dict = {}
    features_list_dict['A'] = data['author'].x
    features_list_dict['P'] = data['paper'].x
    features_list_dict['V'] = data['venue'].x
    n_classes = int(data['author'].y.max()+1)
    return new_g, num_nodes, adjs, node_type_nodes, features_list_dict, init_labels, n_classes, train_nid, val_nid, test_nid


from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
def load_mag(args, symmetric=True):
    dataset = DglNodePropPredDataset(name=args.dataset)#, root=args.root)
    splitted_idx = dataset.get_idx_split()

    g, init_labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx['train']['paper']
    val_nid = splitted_idx['valid']['paper']
    test_nid = splitted_idx['test']['paper']

    features = g.nodes['paper'].data['feat']
    if len(args.extra_embedding):
        # print(f'Use extra embeddings generated with the {args.extra_embedding} method')
        # # path = os.path.join(args.emb_path, f'{args.extra_embedding}_nars')
        # path = '/home/public/lyx/SeHGNN_new/SeHGNN/data/complex_nars/emb_backup'
        # author_emb = torch.load(os.path.join(path, 'author.pt'), map_location=torch.device('cpu')).float()
        # topic_emb = torch.load(os.path.join(path, 'field_of_study.pt'), map_location=torch.device('cpu')).float()
        # institution_emb = torch.load(os.path.join(path, 'institution.pt'), map_location=torch.device('cpu')).float()

        print(f'Use extra embeddings generated with the Line method') 
        import pickle
        nrl_cache_path = '/home/public/lyx/SeHGNN_new/SeHGNN/ogbn/cache/mag.p'
        with open(nrl_cache_path, "rb") as f:
            nrl_embedding_dict = pickle.load(f)
        author_emb = torch.Tensor(nrl_embedding_dict['author'])
        topic_emb = torch.Tensor(nrl_embedding_dict['field_of_study'])
        institution_emb = torch.Tensor(nrl_embedding_dict['institution'])
    else:
        author_emb = torch.Tensor(g.num_nodes('author'), args.embed_size).uniform_(-0.5, 0.5)
        topic_emb = torch.Tensor(g.num_nodes('field_of_study'), args.embed_size).uniform_(-0.5, 0.5)
        institution_emb = torch.Tensor(g.num_nodes('institution'), args.embed_size).uniform_(-0.5, 0.5)

    features_list_dict = {}
    # P_ = P.to_sparse()
    # A_ = A.to_sparse()
    # C_ = C.to_sparse()
    # features_list_dict['P'] = SparseTensor(row=P_.indices()[0], col=P_.indices()[1], value = P_.values(), sparse_sizes=P_.size())
    # features_list_dict['A'] = SparseTensor(row=A_.indices()[0], col=A_.indices()[1], value = A_.values(), sparse_sizes=A_.size())
    # features_list_dict['C'] = SparseTensor(row=C_.indices()[0], col=C_.indices()[1], value = C_.values(), sparse_sizes=C_.size())
    features_list_dict['P'] = features
    features_list_dict['A'] = author_emb
    features_list_dict['I'] = institution_emb
    features_list_dict['F'] = topic_emb

    node_type_nodes = {}
    node_type_nodes['P'] = features.shape[0]
    node_type_nodes['A'] = author_emb.shape[0]
    node_type_nodes['I'] = institution_emb.shape[0]
    node_type_nodes['F'] = topic_emb.shape[0]

    g.nodes['paper'].data['feat'] = features
    g.nodes['author'].data['feat'] = author_emb
    g.nodes['institution'].data['feat'] = institution_emb
    g.nodes['field_of_study'].data['feat'] = topic_emb

    init_labels = init_labels['paper'].squeeze()
    n_classes = int(init_labels.max()) + 1

    # for k in g.ntypes:
    #     print(k, g.ndata['feat'][k].shape)
    for k in g.ntypes:
        print(k, g.nodes[k].data['feat'].shape)

    adjs = []
    adjs_b = []
    edge_key = {}
    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=dst, col=src)
        adjs.append(adj)
        print(g.to_canonical_etype(etype), adj)
        # edge_key[etype] = g.to_canonical_etype(etype)[0].capitalize()[0]+g.to_canonical_etype(etype)[2].capitalize()[0]
        # if edge_key[etype][0]+edge_key[etype][1] == 'PP':
        #     adj2 = SparseTensor(row=dst, col=src)
        #     adj2 = adj2.to_symmetric()
        #     k = torch.cat((adj2.storage.col().unsqueeze(0), adj2.storage.row().unsqueeze(0)), 0)
        #     values = torch.ones(len(adj2.storage.col()))
        #     adj = torch.sparse_coo_tensor(k, values, [node_type_nodes[edge_key[etype][1]], node_type_nodes[edge_key[etype][0]]]).coalesce()
        # else:
        #     k = torch.cat((dst.unsqueeze(0), src.unsqueeze(0)), 0)
        #     values = torch.ones(len(dst))
        #     adj = torch.sparse_coo_tensor(k, values, [node_type_nodes[edge_key[etype][1]], node_type_nodes[edge_key[etype][0]]]).coalesce()
        # adjs.append(adj)
        # print(g.to_canonical_etype(etype), adj)
    # F --- *P --- A --- I
    # paper : [736389, 128]
    # author: [1134649, 256]
    # institution [8740, 256]
    # field_of_study [59965, 256]

    new_edges = {}
    ntypes = set()

    etypes = [ # src->tgt
        ('A', 'A-I', 'I'),
        ('A', 'A-P', 'P'),
        ('P', 'P-P', 'P'),
        ('P', 'P-F', 'F'),
    ]

    if symmetric:
        adjs[2] = adjs[2].to_symmetric()   ####ajds[2]应该是P-P关系
        assert torch.all(adjs[2].get_diag() == 0)

    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        dst, src, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype == dtype:
            new_edges[(stype, rtype, dtype)] = (np.concatenate((src, dst)), np.concatenate((dst, src)))
        else:
            new_edges[(stype, rtype, dtype)] = (src, dst)
            new_edges[(dtype, rtype[::-1], stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)

    new_g = dgl.heterograph(new_edges)
    new_g.nodes['P'].data['P'] = g.nodes['paper'].data['feat']
    new_g.nodes['A'].data['A'] = g.nodes['author'].data['feat']
    new_g.nodes['I'].data['I'] = g.nodes['institution'].data['feat']
    new_g.nodes['F'].data['F'] = g.nodes['field_of_study'].data['feat']
    
    IA, PA, PP, FP = adjs
    AI = IA.t()
    AP = PA.t()
    PF = FP.t()
    num_nodes = g.num_nodes('paper')
    # new_g = None
    adjs = {'IA': IA, 'AI': AI, 'PA': PA, 'AP': AP, 'PF': PF, 'FP': FP, 'PP': PP}
    IA = None
    PA = None
    PP = None
    FP = None
    AI = None
    AP = None
    PF = None
    return new_g, num_nodes, adjs, node_type_nodes, features_list_dict, init_labels, n_classes, train_nid, val_nid, test_nid
