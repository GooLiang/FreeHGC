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
import argparse

from torch_sparse import SparseTensor
from torch_sparse import remove_diag, to_torch_sparse, set_diag
from copy import deepcopy
from data_loader_hgb import data_loader

def FreeHGC_convert(args):
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

    idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

    val_ratio = 0.2
    train_nid = np.nonzero(dl.labels_train['mask'])[0]   ###统计label不为0的
    np.random.shuffle(train_nid)  ###每次都是打乱的train_nid
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
        if args.num_hops > 2 or args.two_layer:
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
        PP = PP.set_diag()  ###对角线全部置1
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
        features_list_dict['0'] = _0
        features_list_dict['1'] = _1
        features_list_dict['2'] = _2
        features_list_dict['3'] = _3
        features_list_dict['4'] = _4
        features_list_dict['5'] = _5
        features_list_dict['6'] = _6
        features_list_dict['7'] = _7
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
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
        ## convert condense graph to other models##
        ###########################################
        candidate = {}
        idx_selected = []
        candidate['A'] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_A.pt')
        idx_selected.append(candidate['A'])
        for key in ['P', 'T', 'V']:
            # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key}.pt')                    
            candidate[key] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}_type_{key}.pt')                    
            idx_selected.append(candidate[key])
        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGB/{args.dataset}/selected_hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
        
        new_adjs = {}
        for key, value in adjs.items():
            new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
        ### selected node ###
        selected_node = candidate['A']
        ### shift ###
        shift_idx = {'A':0, 'P':1, 'T':2, 'V':3}
        idx_shift = np.zeros(len(candidate.keys())+1, dtype=np.int32)
        idx_shift_RGCN = {}
        for i,key in enumerate(candidate.keys()):
            idx_shift[i+1] = idx_shift[i] + len(candidate[key]) 
            idx_shift_RGCN[i] = idx_shift[i]
        
        link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'} 
        link_type_dic_HAN = {'ap': 'AP', 'pa': 'PA', 'pc': 'PT', 'cp': "TP",'pt': 'PV', 'tp': 'VP'}
        links = range(len(link_type_dic))
    elif args.dataset == 'ACM':
        adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP, 'PK': PK, 'KP': KP}

        ## convert condense graph to other models##
        ###########################################
        candidate = {}
        idx_selected = []
        candidate['P'] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_P.pt')
        idx_selected.append(candidate['P'])
        for key in ['A', 'C', 'K']:
            # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key}.pt')                    
            candidate[key] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}_type_{key}.pt')                     
            idx_selected.append(candidate[key])
        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGB/{args.dataset}/selected_hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
        new_adjs = {}
        for key, value in adjs.items():
            new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]

        ### random ###
        idx_selected = {}
        candidate['P'] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/random/{args.dataset}/rrate_{args.reduction_rate}_type_P.pt')
        idx_selected[0] = candidate['P']
        for i, key in enumerate(['A', 'C', 'K']):
            # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key}.pt')                    
            candidate[key] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/random/{args.dataset}/rrate_{args.reduction_rate}_type_{key}.pt')    
            idx_selected[i+1] = candidate[key]
        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/RSHN/{args.dataset}/selected_hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
        
        new_adjs_random = {}
        for key, value in adjs.items():
            new_adjs_random[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
        ### random ###
            
        ### shift ###
        shift_idx = {'P': 0, 'A':1, 'C':2, 'K':3}
        idx_shift = np.zeros(len(candidate.keys())+1, dtype=np.int32)
        idx_shift_RGCN = {}
        for i,key in enumerate(candidate.keys()):
            idx_shift[i+1] = idx_shift[i] + len(candidate[key]) 
            idx_shift_RGCN[i] = idx_shift[i]

        if args.ACM_keep_F:
            links = dl.links['data'].keys()
            link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
            link_type_dic_HAN = {'pp': 'PP', '-pp': '-PP', 'pa': 'PA', 'ap': 'AP', 'ps': 'PC', 'sp': 'CP', 'pt': 'PK', 'tp': 'KP'}
        if not args.ACM_keep_F:
            link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp'}
            links = range(len(link_type_dic))
            link_type_dic_HAN = {'pp': 'PP', '-pp': '-PP', 'pa': 'PA', 'ap': 'AP', 'ps': 'PC', 'sp': 'CP'}

    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
        ## convert condense graph to other models##
        ###########################################
        candidate = {}
        idx_selected = []
        candidate['M'] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_M.pt')
        idx_selected.append(candidate['M'])
        for key in ['D', 'A', 'K']:
            # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key}.pt')                    
            candidate[key] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}_type_{key}.pt')                     
            idx_selected.append(candidate[key])
        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGB/{args.dataset}/selected_hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')

        new_adjs = {}
        for key, value in adjs.items():
            new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
        ### selected node ###
        selected_node = candidate['A']
        ### shift ###
        shift_idx = {'M':0, 'D':1, 'A':2, 'K':3}
        idx_shift = np.zeros(len(candidate.keys())+1, dtype=np.int32)
        idx_shift_RGCN = {}
        for i,key in enumerate(candidate.keys()):
            idx_shift[i+1] = idx_shift[i] + len(candidate[key])
            idx_shift_RGCN[i] = idx_shift[i]

        link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
        link_type_dic_HAN = {'md': 'MD', 'dm': 'DM', 'ma': 'MA', 'am': "AM",'mk': 'MK', 'km': 'KM'}
        links = range(len(link_type_dic))
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

        ## convert condense graph to other models##
        ###########################################
        candidate = {}
        idx_selected = []
        candidate['0'] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_0.pt')
        idx_selected.append(candidate['0'])
        for key in ['1', '2', '3', '4', '5', '6' ,'7']:
            # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key}.pt')                    
            candidate[key] = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}_type_{key}.pt')                     
            idx_selected.append(candidate[key])
        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGB/{args.dataset}/selected_hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
        new_adjs = {}
        for key, value in adjs.items():
            new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]

        ### shift ###
        shift_idx = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7}
        idx_shift = np.zeros(len(candidate.keys())+1, dtype=np.int32)
        idx_shift_RGCN = {}
        for i,key in enumerate(candidate.keys()):
            idx_shift[i+1] = idx_shift[i] + len(candidate[key])
            idx_shift_RGCN[i] = idx_shift[i]
        
        link_type_dic = {0: '00', 1: '01', 2: '03', 3: '05', 4: '06',
                        5: '11',
                        6: '20', 7: '21', 8: '22', 9: '23', 10: '25',
                        11: '31', 12: '33', 13: '35',
                        14: '40', 15: '41', 16: '42', 17: '43', 18: '44', 19: '45', 20: '46', 21: '47',
                        22: '51', 23: '55',
                        24: '61', 25: '62', 26: '63', 27: '65', 28: '66', 29: '67',
                        30: '70', 31: '71', 32: '72', 33: '73', 34: '75', 35: '77',
                        36: '-00', 37: '10', 38: '30', 39: '50', 40: '60',
                        41: '-11',
                        42: '02', 43: '12', 44: '-22', 45: '32', 46: '52',
                        47: '13', 48: '-33', 49: '53',
                        50: '04', 51: '14', 52: '24', 53: '34', 54: '-44', 55: '54', 56: '64', 57: '74',
                        58: '15', 59: '-55',
                        60: '16', 61: '26', 62: '36', 63: '56', 64: '-66', 65: '76',
                        66: '07', 67: '17', 68: '27', 69: '37', 70: '57', 71: '-77',
                        }
        links = range(int(len(link_type_dic)/2))
        link_type_dic_HAN = {value:value for key,value in link_type_dic.items()}

        
    else:
        assert 0
    ###########################################
    ## convert condense graph to other models##

    data_dic = {}
    for key, adj in new_adjs.items():
        row = adj.storage.row()+idx_shift[shift_idx[key[0]]]
        col = adj.storage.col()+idx_shift[shift_idx[key[1]]]
        # data_dic_HAN[key] = (row.numpy(), col.numpy())  ###origin
        data_dic[key[::-1]] = (col.numpy(), row.numpy())  ###这里是为了跟HAN对齐
        if key[0] == key[1]:
            data_dic['-'+key[::-1]] = (row.numpy(), col.numpy())  ###这里是为了跟HAN对齐

    # data_dic_random = {}
    # for key, adj in new_adjs_random.items():
    #     row = adj.storage.row()+idx_shift[shift_idx[key[0]]]
    #     col = adj.storage.col()+idx_shift[shift_idx[key[1]]]
    #     # data_dic_HAN[key] = (row.numpy(), col.numpy())  ###origin
    #     data_dic_random[key[::-1]] = (col.numpy(), row.numpy())  ###这里是为了跟HAN对齐
    #     if key[0] == key[1]:
    #         data_dic_random['-'+key[::-1]] = (row.numpy(), col.numpy())  ###这里是为了跟HAN对齐
            
    ### 注意，HAN跟HGB都把K类型算进去了，我们如果要对齐也需要算K   6.5:针对ACM？
    data_dic_HAN = {}
    data_dic_hgb = {}
    data_dic_RSHN = {}
    for link_type in links:
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic_HAN[(src_type, link_type_dic[link_type], dst_type)] = data_dic[link_type_dic_HAN[link_type_dic[link_type]]]
        # reverse
        if args.dataset == 'Freebase' and link_type_dic[link_type + 36][0] != '-':
            data_dic_HAN[(dst_type, link_type_dic[link_type + 36], src_type)] = dl.links['data'][link_type].T.nonzero()
        #################################
        ######### for HGB #########
        n = idx_shift[-1]
        i = data_dic[link_type_dic_HAN[link_type_dic[link_type]]][0]
        j = data_dic[link_type_dic_HAN[link_type_dic[link_type]]][1]
        assert len(i) == len(j)
        data = list(np.ones(len(i)))
        data_dic_hgb[link_type] = sp.coo_matrix((data, (i,j)), shape=(n, n)).tocsr()  # <==> dl.links['data']
        # ######### for RSHN ########
        # n = idx_shift[-1]
        # i = data_dic_random[link_type_dic_HAN[link_type_dic[link_type]]][0]
        # j = data_dic_random[link_type_dic_HAN[link_type_dic[link_type]]][1]
        # assert len(i) == len(j)
        # data = list(np.ones(len(i)))
        # data_dic_RSHN[link_type] = sp.coo_matrix((data, (i,j)), shape=(n, n)).tocsr()  # <==> dl.links['data']      
        # torch.save(data_dic_RSHN, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/RSHN/{args.dataset}/random_rrate_{args.reduction_rate}.pt')

    torch.save(data_dic_HAN, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HAN/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
    torch.save(data_dic_hgb, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/RSHN/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
    torch.save(data_dic_hgb, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGB/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
    # adjM = sum(data_dic_hgb.values())   ### type for HGB

    ######### for RGCN #########
    from scipy import sparse
    data_dic_RGCN = {}
    for etype in dl.links['meta']:
        etype_info = dl.links['meta'][etype]
        metrix = data_dic_hgb[etype]  ### data_dic_hgb[etype]
        data_dic_RGCN[(etype_info[0], 'link', etype_info[1])] = (
            sparse.find(metrix)[0]-idx_shift_RGCN[etype_info[0]], sparse.find(metrix)[1]-idx_shift_RGCN[etype_info[1]])   #idx_shift
    torch.save(data_dic_RGCN, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/RGCN/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
    ######### for RGCN #########
    
    ######## for HGT ###########
    nodes_count = {}
    i = 0
    for key,value in candidate.items():
        print(i, nodes_count)
        nodes_count[str(i)] = len(value)
        i += 1
    torch.save(nodes_count, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGT/{args.dataset}/node_count_hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
    
    edge_dict = {}
    for i, meta_path in dl.links['meta'].items():
        edge_dict[(str(meta_path[0]), str(meta_path[0]) + '_' + str(meta_path[1]), str(meta_path[1]))] = (torch.tensor(data_dic_hgb[i].tocoo().row - idx_shift_RGCN[meta_path[0]]), torch.tensor(data_dic_hgb[i].tocoo().col - idx_shift_RGCN[meta_path[1]]))
    torch.save(edge_dict, f'/home/public/lyx/FreeHGC/hgb/convert_condense_graph/HGT/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}.pt')
    ###########################################
    ## convert condense graph to other models##
    return 0

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FreeHGC")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--root", type=str, default="../data/")
    parser.add_argument("--data-dir", type=str, default=None, help="path to dataset, only used for OAG")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)   ##50000
    parser.add_argument("--eval-batch-size", type=int, default=25000,  ##250000
                        help="evaluation batch size, -1 for full batch")
    parser.add_argument("--ff-layer-1", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--ff-layer-2", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--ff-layer-mid", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--ff-layer-mid-out", type=int, default=2,
                        help="number of feed-forward layers")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-preprocess", action="store_true",
                        help="Preprocess on CPU")
    parser.add_argument("--r-length", type=int, default=2)
    parser.add_argument("--in-feats", type=int, default=512)
    parser.add_argument("--micro", type=bool, default=True)
    parser.add_argument('--patience', type=int, default=50, help='Patience.')
    parser.add_argument("--residual", type=bool, default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument("--load-feature", type=bool, default=False)
    parser.add_argument("--embed-size", type=int, default=256,
                    help="inital embedding size of nodes with no attributes")
    parser.add_argument("--input-dropout", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="att dropout of attention scores")
    parser.add_argument("--SGA", action='store_true', default=False)
    parser.add_argument("--enhance", action='store_true', default=False)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--r", nargs='+', type=float, default=[0.0],
                        help="the seed used in the training")
    parser.add_argument("--sum-metapath", action='store_true', default=False)
    parser.add_argument("--bns", action='store_true', default=False)
    parser.add_argument("--mean", action='store_true', default=False)
    parser.add_argument("--transformer", action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default="Freebase")
    parser.add_argument("--num-hops", type=int, default=2, help="number of hops")
    parser.add_argument('--method', type=str, default='FreeHGC', choices=['kcenter', 'herding', 'herding_class','random', 'FreeHGC'])
    parser.add_argument("--reduction-rate", type=float, default=0.9)
    parser.add_argument("--pr", type=float, default=0.15)  ###ACM 0.85, DBLP 0.95?
    parser.add_argument("--ACM-keep-F", type=bool, default=True)
    #parser.add_argument("--infeat", type=int, default=512)
    args = parser.parse_args()

    print(args)
    FreeHGC_convert(args)
