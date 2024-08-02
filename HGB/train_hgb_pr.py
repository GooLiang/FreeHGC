import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import logging
import uuid
from model_hgb import *
from model_SeHGNN import *
import sys
from data_hgb import *
from utils_hgb import *
from core_set_methods import *
import random
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from copy import deepcopy
import torch
from data_selection import *
from collections import Counter
from pprfile import topk_ppr_matrix

def main(args):
    if args.seed is not None:
        set_random_seed(args.seed)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
    print("seed", args.seed)
    # Load dataset
    g, adjs, features_list_dict, node_type_nodes, edge_type_ratio, labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full \
        = load_dataset(args)
    evaluator = get_evaluator(args.dataset)
    # g = None
    import os
    if args.dataset == 'Freebase':
        if not os.path.exists('./Freebase_adjs'):
            os.makedirs('./Freebase_adjs')        

    # =======
    # neighbor aggregation
    # =======
    if args.dataset == 'DBLP':
        tgt_type = 'A'
        extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        extra_metapath = []
    elif args.dataset == 'Freebase':
        tgt_type = '0'
        extra_metapath = []
    else:
        assert 0
    max_length = args.num_hops + 1
    #r_list = [0.]
    r_list = args.r
    threshold_metalen = args.threshold
    r = 0.0
    print(r)
    feats_r_ensemble = []
    coreset_feats_r_ensemble = []
    extra_feats_r_ensemble = []
    label_feats_r_ensemble = []
    features_list_dict_type = {}
    features_list_dict_cp = features_list_dict
    with torch.no_grad():
        #############normalization---wait to modify#########################
        if args.dataset != "Freebase":
            prop_device = 'cuda:{}'.format(args.gpu)  ###对于dblp需要用gpu,IMDB一样可以用
            ###core-set###
            if args.method == 'kcenter':
                agent = KCenter(labels, train_nid, args, device)
                start = time.time() 
                for tgt_node_key in node_type_nodes:  ###求所有节点类型的特征
                    features_list_dict_type[tgt_node_key], adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_node_key, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True) 
                end = time.time()
                print("time for feature propagation", end - start) 
            if args.method == 'herding':
                agent = Herding(labels, train_nid, args, device)
                start = time.time()
                for tgt_node_key in node_type_nodes:  ###求所有节点类型的特征
                    features_list_dict_type[tgt_node_key], adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_node_key, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)    
                end = time.time()
                print("time for feature propagation", end - start)          
            if args.method == 'herding_class':
                agent = Herding_class(labels, train_nid, args, device)
            if args.method == 'random':
                agent = Random(labels, train_nid, args, device)
                start = time.time()
                features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)
                end = time.time()
            if args.method == 'FreeHGC':
                num_class_dict = Base(labels, train_nid, args, device).num_class_dict
                # real_reduction_rate = sum(num_class_dict.values())/node_type_nodes[tgt_type]
                start = time.time()
                # g = hg_propagate(g, tgt_type, args.num_hops, args.num_hops + 1, extra_metapath, echo=False)             
                features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)
                end = time.time()
                print("time for feature propagation", end - start)

            if args.method == 'herding' or args.method == 'kcenter':
                if args.method == 'herding':
                    flag = False
                if args.method == 'kcenter':
                    flag = True
                dis_dict_sum = {}
                dis_dict_sum[tgt_type] = {}
                
                embeds = torch.load(f'/home/public/lyx/FreeHGC/hgb/embeds/{args.dataset}_embeds.pt')
                dis_dict_sum[tgt_type] = agent.select(embeds)
                
                real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                for key_A, key_B in features_list_dict_type.items():
                    if key_A != tgt_type:
                        reduce_nodes = int(real_reduction_rate * node_type_nodes[key_A])  #args.reduction_rate
                        if reduce_nodes == 0:
                            reduce_nodes = int(0.1 * node_type_nodes[key_A])
                        dis_dict_sum[key_A] = {}
                        for key in key_B:
                            dis_dict_sum[key_A][key] = agent.select_other_types_top(features_list_dict_type[key_A][key], reduce_nodes)

                candidate = {}
                ### topk for target node types###
                idx_selected = dis_dict_sum[tgt_type]
                candidate[tgt_type] = idx_selected
                if not flag:
                    torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/condense_graph/herding/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{tgt_type}.pt')                    
                else:
                    torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/condense_graph/kcenter/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{tgt_type}.pt')                    

                ### topk for target node types###


                # ### topk for other node types ###
                # real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                # for key, value in features_list_dict_cp.items():
                #     reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                #     if key != tgt_type:
                #         if reduce_nodes == 0:
                #             candidate[key] = agent.select_other_types(features_list_dict_cp[key], int(0.1 * node_type_nodes[key]))
                #         else:
                #             candidate[key] = agent.select_other_types(features_list_dict_cp[key], reduce_nodes)
                # x = 1
                # ### topk for other node types ###


                ### topk for other node types pro ###
                score_train_idx_sum = defaultdict(list)
                real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                for key_A,key_B in dis_dict_sum.items():
                    if key_A != tgt_type:
                        score_train_idx_sum[key_A] = None
                        for key in key_B:
                            score_train_idx = dict(zip(dis_dict_sum[key_A][key].keys(), [v+1e-12 for k,v in dis_dict_sum[key_A][key].items()]))
                            score_train_idx_sum[key_A] = dict(Counter(score_train_idx_sum[key_A]) + Counter(score_train_idx))
                ###### 对每个class_id的所有meta的结果score_train_idx_sum求topk
                # score_train_idx = torch.argsort(torch.tensor(list(score_train_idx_sum[class_id].values())), descending=True)
                        reduce_nodes = int(real_reduction_rate * node_type_nodes[key_A])  #args.reduction_rate
                        if reduce_nodes == 0:
                            reduce_nodes = int(0.1 * node_type_nodes[key_A])
                        _, score_train_idx = torch.topk(torch.tensor(list(score_train_idx_sum[key_A].values())), reduce_nodes, largest = flag)
                        score_train_idx = torch.tensor(list(score_train_idx_sum[key_A].keys()))[score_train_idx]
                        candidate[key_A] = score_train_idx.numpy()
                ### topk for other node types pro ###

            elif args.method == 'random':
                candidate = {}
                for key, value in node_type_nodes.items():
                    if key == tgt_type:
                        candidate[key] = agent.select()
                        idx_selected = candidate[key]
                    else:
                        real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                        reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                        if reduce_nodes == 0:
                            reduce_nodes = int(0.1 * node_type_nodes[key])
                        candidate[key] = agent.select_other_types(np.arange(value), reduce_nodes)
                    torch.save(candidate[key], f'/home/public/lyx/FreeHGC/hgb/condense_graph/random/{args.dataset}/rrate_{args.reduction_rate}_type_{key}.pt')                    

            elif args.method == 'FreeHGC':
                import copy
                budget = int(args.reduction_rate * len(train_nid))  #wrong node_type_nodes[tgt_type]
                ###############################
                ###condense target node type###
                key_counter = {}
                ppr = {}
                ppr_sum = {}
                for key in adj_dict.keys():
                        key_counter.setdefault(key[-1], []).append(key)
                        
                ###condense target node type###
                ###############################
                ppr = {}
                ppr_sum = {}
                key_A = tgt_type
                ppr[key_A] = {}
                ppr_sum[key_A] = 0
                candidate = {}
                for key in key_counter[tgt_type]:
                    ppr[key_A][key]= calc_ppr(adj_dict[key], key, args.pr, device)  #[score_train_idx]待验证
                    ppr_sum[key_A] += ppr[key_A][key] ##V1: 不同metapath直接相加，这里可以考虑优化
                    # ppr_sum[key_A] += ppr[key_A][key] ##V2(6.5): 直接这么做acc很差
                    # ppr[key_A][key] = None
                ppr_sum[key_A] = torch.sum(ppr_sum[key_A], dim = 0)
                
                ### 不考虑class
                reduce_nodes = sum(num_class_dict.values())
                _, candidate[key_A] = torch.topk(ppr_sum[key_A], k = reduce_nodes)   
                idx_selected = candidate[key_A]
                ### 不考虑class
                
                # ### 考虑class
                # a = []
                # labels_train = labels[train_nid]
                # for class_id, cnt in num_class_dict.items():
                #     idx = train_nid[labels_train==class_id]
                #     _, id = torch.topk(ppr_sum[key_A][idx], k = cnt) 
                #     a.append(idx[id])
                # idx_selected = np.sort(np.concatenate(a))
                # candidate[key_A] = idx_selected
                
                # ## PPR: condense other node types ###
                # key_counter = {}
                # ppr = {}
                # ppr_sum = {}
                # for key in adj_dict.keys():
                #     if key[-1] != tgt_type:
                #         key_counter.setdefault(key[-1], []).append(key)
                # for key_A, key_B in key_counter.items():
                #     ppr[key_A] = {}
                #     ppr_sum[key_A] = 0
                #     for key in key_B:
                #         ppr[key_A][key]= calc_ppr(adj_dict[key], key, args.pr, device)  #[score_train_idx]待验证
                #         ppr_sum[key_A] += ppr[key_A][key][idx_selected] ##V1: 不同metapath直接相加，这里可以考虑优化
                #         # ppr_sum[key_A] += ppr[key_A][key] ##V2(6.5): 直接这么做acc很差
                #         # ppr[key_A][key] = None
                #     ppr_sum[key_A] = torch.sum(ppr_sum[key_A], dim = 0)
                # # torch.save(ppr_sum, f'/home/public/lyx/FreeHGC/hgb/tuning_graph/{args.dataset}/ppr_sum_{args.num_hops}_{args.reduction_rate}_pr_{args.pr}.pt')
                # # ppr_sum = torch.load(f'/home/public/lyx/FreeHGC/hgb/tuning_graph/{args.dataset}/ppr_sum_{args.num_hops}_{args.reduction_rate}_pr_{args.pr}.pt')
                
                # real_reduction_rate = len(idx_selected)/node_type_nodes[tgt_type]
                # print("real_reduction_rate: ", real_reduction_rate)
                # for key, value in ppr_sum.items():
                #     reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                #     if reduce_nodes == 0:
                #         _, candidate[key] = torch.topk(ppr_sum[key], k = int(0.1 * node_type_nodes[key]))
                #     else:
                #         _, candidate[key] = torch.topk(ppr_sum[key], k = reduce_nodes)
                    # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}_type_{key}.pt')                    

                ## PPR: condense other node types ###
                
                # ### random 
                # for key, value in ppr_sum.items():
                #     reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                #     if reduce_nodes == 0:
                #         nodes = torch.arange(len(ppr_sum[key]))
                #         selected = np.random.permutation(nodes)
                #         candidate[key] = selected[:int(0.1 * node_type_nodes[key])]
                #     else:
                #         nodes = torch.arange(len(ppr_sum[key]))
                #         selected = np.random.permutation(nodes)
                #         candidate[key] = selected[:reduce_nodes]
                # ### random
                
            # new_adjs = {}
            # for key, value in adjs.items():
            #     new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
            # for key, value in features_list_dict_cp.items():
            #     features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]

            ### only target ###
            candidate = {}
            candidate[tgt_type] = idx_selected
            new_adjs = {}
            for key, value in adjs.items():
                if key[0] == tgt_type and key[1] != tgt_type:
                    new_adjs[key] = adjs[key][candidate[tgt_type], :]
                elif key[1] == tgt_type and key[0] != tgt_type:
                    new_adjs[key] = adjs[key][:, candidate[tgt_type]]
                elif key[0] == key[1]:
                    new_adjs[key] = adjs[key][candidate[tgt_type]][candidate[tgt_type]]
                else:
                    new_adjs[key] = adjs[key]
            features_list_dict_cp[tgt_type] = features_list_dict_cp[tgt_type][candidate[tgt_type], :]
            ### only target ###


            # def relation_condition(key):
            #     if args.dataset == 'ACM':
            #         key_A = ['PA', 'PC', 'PK']
            #     if args.dataset == 'DBLP':
            #         key_A = ['PT', 'PV']
            #     if args.dataset == 'IMDB':
            #         key_A = ['MK', 'MA', 'MD']
            #     if key in  key_A:
            #         return True
            #     else:
            #         return False
                 
            # if args.dataset == 'DBLP':
            #     new_adjs = {}
            #     for key, value in adjs.items():
            #         if key[0] == tgt_type or key[1] == tgt_type:
            #             new_adjs[key] = adjs[key][candidate[key[0]], candidate[key[1]]]
            #         elif key[0] == 'P':
            #             new_adjs[key] = adjs[key][candidate[key[0]], :]
            #         elif key[1] == 'P':
            #             new_adjs[key] = adjs[key][:, candidate[key[1]]]
                        
            #     for key, value in features_list_dict_cp.items():
            #         if key == 'A' or key =='P':
            #             features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]

            # if args.dataset == 'ACM' or args.dataset == 'IMDB':
            #     new_adjs = {}
            #     for key, value in adjs.items():
            #         if key[0] == tgt_type and key[1] == tgt_type:
            #             new_adjs[key] = adjs[key][candidate[key[0]], candidate[key[1]]]
            #         elif key[0] == tgt_type:
            #             new_adjs[key] = adjs[key][candidate[key[0]], :]
            #         elif key[1] == tgt_type:
            #             new_adjs[key] = adjs[key][:, candidate[key[1]]]
                        
            #     for key, value in features_list_dict_cp.items():
            #         if  key == tgt_type:
            #             features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]
                                 
            # new_index = {}
            # edge_count = {}
            # features_new = {}
            # for key_A, val in new_adjs.items():
            #     ### 只有2类结构的判断###
            #     if relation_condition(key_A):
            #         a_list = val.storage._row.tolist()
            #         b_list = val.storage._col.tolist()
            #         # Construct the dictionary
            #         result_dict = {}
            #         for key, value in zip(a_list, b_list):
            #             result_dict.setdefault(key, []).append(value)
            #             # if key in result_dict:
            #             #     result_dict[key].append(value)
            #             # else:
            #             #     result_dict[key] = [value]
                        
            #         pool = []
            #         new_index[key_A[1]] = {}
            #         edge_count[key_A[1]] = {}
            #         i = 0
            #         features_new[key_A[1]] = []
            #         for key, value in result_dict.items():
            #             if value in pool:
            #                 new_index[key_A[1]][key] = new_index[key_A[1]][pool.index(value)]
            #             else:
            #                 new_index[key_A[1]][key] = i
            #                 edge_count[key_A[1]][i] = len(value)
            #                 features_new[key_A[1]].append(torch.sum(features_list_dict_cp[key_A[1]][value], dim = 0)/len(value))   ### /长度是V1版本
            #                 i = i + 1
            #                 pool.append(value)
            #         row = list(new_index[key_A[1]].keys())
            #         col = list(new_index[key_A[1]].values())
            #         # torch.save(row, f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_back_new_type_row_{key_A}.pt')
            #         # torch.save(col, f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_back_new_type_col_{key_A}.pt')
            #         new_adjs[key_A] = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=(max(row)+1, i))
            #         new_adjs[key_A[::-1]] = SparseTensor(row=torch.LongTensor(col), col=torch.LongTensor(row), sparse_sizes=(i, max(row)+1))
            #         # new_adjs[key_A].storage.row = torch.tensor(list(edge_count[key_A[1]].values()))
            #         features_list_dict_cp[key_A[1]] = torch.stack(features_new[key_A[1]])
                    

            train_nid = np.arange(len(idx_selected))
            labels_train = labels[idx_selected].to(device)

            ###core-set###
            start = time.time()
            coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)                    
            # features_list_dict, adj_dict, extra_features_buffer = coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer
            end = time.time()
            print("Core-set: time for feature propagation", end - start)
            
            feats = {}
            feats_core = {}
            feats_extra = {}

            keys = list(coreset_features_list_dict.keys())
            print(f'For tgt {tgt_type}, feature keys {keys}')
            keys_extra = list(extra_features_buffer.keys())
            print(f'For tgt {tgt_type}, extra feature keys {keys_extra}')
            for k in keys:
                if args.method == 'FreeHGC' or args.method == 'random':
                    feats[k] = features_list_dict.pop(k)
                else:
                    feats[k] = features_list_dict_type[tgt_type].pop(k)
                feats_core[k] = coreset_features_list_dict.pop(k)
            for k in keys_extra:
                feats_extra[k] = extra_features_buffer.pop(k)
            data_size = {k: v.size(-1) for k, v in feats.items()}
            data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}




        elif args.dataset == "Freebase":
            prop_device = 'cuda:{}'.format(args.gpu)
            if args.method == 'kcenter':
                agent = KCenter(labels, train_nid, args, device)
                start = time.time()
                for tgt_node_key in node_type_nodes:  ###求所有节点类型的特征
                    features_list_dict_type[tgt_node_key], extra_features_buffer = hg_propagate_sparse_pyg_freebase(adjs, tgt_node_key, args.num_hops, max_length, extra_metapath, prop_device, args.enhance, prop_feats=True, echo=True)
                end = time.time()
                print("time for feature propagation", end - start) 
            if args.method == 'herding':
                agent = Herding(labels, train_nid, args, device)
                start = time.time()
                for tgt_node_key in node_type_nodes:  ###求所有节点类型的特征
                    features_list_dict_type[tgt_node_key], extra_features_buffer = hg_propagate_sparse_pyg_freebase(adjs, tgt_node_key, args.num_hops, max_length, extra_metapath, prop_device, args.enhance, prop_feats=True, echo=True)
                # torch.save(features_list_dict_type, f'/home/public/lyx/FreeHGC/hgb/condense_graph/herding/{args.dataset}/hops_{args.num_hops}_aggregate.pt')
                # features_list_dict_type = torch.load(f'/home/public/lyx/FreeHGC/hgb/condense_graph/herding/{args.dataset}/hops_{args.num_hops}_aggregate.pt')
                end = time.time()
                print("time for feature propagation", end - start)
            if args.method == 'herding' or args.method == 'kcenter':
                if args.method == 'herding':
                    flag = False
                if args.method == 'kcenter':
                    flag = True
                dis_dict_sum = {}
                dis_dict_sum[tgt_type] = {}
                for key in features_list_dict_type[tgt_type]:
                    dis_dict_sum[tgt_type][key] = agent.select_top(features_list_dict_type[tgt_type][key])
                print("finish target")
                
                real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                for key_A, key_B in features_list_dict_type.items():
                    if key_A != tgt_type:
                        reduce_nodes = int(real_reduction_rate * node_type_nodes[key_A])  #args.reduction_rate
                        if reduce_nodes == 0:
                            reduce_nodes = int(0.1 * node_type_nodes[key_A])
                        dis_dict_sum[key_A] = {}
                        for key in key_B:
                            dis_dict_sum[key_A][key] = agent.select_other_types_top(features_list_dict_type[key_A][key], reduce_nodes)
                        print("finish key", key_A)
                if not flag:
                    torch.save(dis_dict_sum, f'/home/public/lyx/FreeHGC/hgb/condense_graph/herding/dic_{args.dataset}_hops_{args.num_hops}_rrate_{args.reduction_rate}.pt')                    
                else:
                    torch.save(dis_dict_sum, f'/home/public/lyx/FreeHGC/hgb/condense_graph/kcenter/dic_{args.dataset}_hops_{args.num_hops}_rrate_{args.reduction_rate}.pt')  

                candidate = {}
                ### topk for target node types###
                idx_selected = []
                score_train_idx_sum = defaultdict(list)
                for class_id, class_budget in agent.num_class_dict.items():
                    score_train_idx_sum[class_id] = None
                    for key,value in dis_dict_sum[tgt_type].items():        
                        score_train_idx = dis_dict_sum[tgt_type][key][class_id]
                        score_train_idx_sum[class_id] = dict(Counter(score_train_idx_sum[class_id]) + Counter(score_train_idx))
                    ###### 对每个class_id的所有meta的结果score_train_idx_sum求topk
                    # score_train_idx = torch.argsort(torch.tensor(list(score_train_idx_sum[class_id].values())), descending=True)
                    _, score_train_idx = torch.topk(torch.tensor(list(score_train_idx_sum[class_id].values())), class_budget, largest = flag)
                    score_train_idx = torch.tensor(list(score_train_idx_sum[class_id].keys()))[score_train_idx]
                    idx_selected.append(score_train_idx)
                idx_selected = torch.cat(idx_selected, dim = 0).numpy()
                candidate[tgt_type] = idx_selected
                if not flag:
                    torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/condense_graph/herding/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{tgt_type}.pt')                    
                else:
                    torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/condense_graph/kcenter/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{tgt_type}.pt')            
                print("{}: rate {}: real_reduction_rate {}".format(args.dataset, args.reduction_rate, real_reduction_rate))
                ### topk for target node types###


                # ### topk for other node types ###
                # real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                # for key, value in features_list_dict_cp.items():
                #     reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                #     if key != tgt_type:
                #         if reduce_nodes == 0:
                #             candidate[key] = agent.select_other_types(features_list_dict_cp[key], int(0.1 * node_type_nodes[key]))
                #         else:
                #             candidate[key] = agent.select_other_types(features_list_dict_cp[key], reduce_nodes)
                # x = 1
                # ### topk for other node types ###


                ### topk for other node types pro ###
                score_train_idx_sum = defaultdict(list)
                real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                for key_A,key_B in dis_dict_sum.items():
                    if key_A != tgt_type:
                        score_train_idx_sum[key_A] = None
                        for key in key_B:
                            score_train_idx = dis_dict_sum[key_A][key]
                            score_train_idx_sum[key_A] = dict(Counter(score_train_idx_sum[key_A]) + Counter(score_train_idx))
                ###### 对每个class_id的所有meta的结果score_train_idx_sum求topk
                # score_train_idx = torch.argsort(torch.tensor(list(score_train_idx_sum[class_id].values())), descending=True)
                        reduce_nodes = int(real_reduction_rate * node_type_nodes[key_A])  #args.reduction_rate
                        if reduce_nodes == 0:
                            reduce_nodes = int(0.1 * node_type_nodes[key_A])
                        _, score_train_idx = torch.topk(torch.tensor(list(score_train_idx_sum[key_A].values())), reduce_nodes, largest = flag)
                        score_train_idx = torch.tensor(list(score_train_idx_sum[key_A].keys()))[score_train_idx]
                        candidate[key_A] = score_train_idx.numpy()
                ### topk for other node types pro ###
                    if not flag:
                        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/condense_graph/herding/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key_A}.pt')              
                    else:
                        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/hgb/condense_graph/kcenter/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{key_A}.pt')
                
            if args.method == 'random':
                agent = Random(labels, train_nid, args, device)
                start = time.time()
                features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)
                end = time.time()
                feats = features_list_dict
                feats_core = {}
                feats_extra = {}
                label_feats = {}
                labels_train = labels.to(device)
                data_size = {k: v.size(-1) for k, v in feats.items()}
                data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}
            if args.method == 'FreeHGC':
                import copy
                num_class_dict = Base(labels, train_nid, args, device).num_class_dict
                start = time.time()
                features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)
                end = time.time()
                print("time for feature propagation", end - start)

            #     ###############################
            #     ###condense target node type###
            #     key_counter = {}
            #     ppr = {}
            #     ppr_sum = {}
            #     for key in adj_dict.keys():
            #             key_counter.setdefault(key[-1], []).append(key)
                
                key_counter = {}
                ppr = {}
                ppr_sum = {}
                for key in adj_dict.keys():
                        key_counter.setdefault(key[-1], []).append(key)
                        
                ### ppr condense target type ###
                ppr = {}
                ppr_sum = {}
                key_A = tgt_type
                ppr[key_A] = {}
                ppr_sum[key_A] = 0
                for key in key_counter[tgt_type]:
                    idx=np.arange(adj_dict[key].size(0))
                    flag = key[0] == key[-1]
                    ppr[key_A][key] = topk_ppr_matrix(adj_dict[key], flag, alpha=args.alpha , eps=1e-4 , idx=idx, topk=0, normalization='sym')
                    ppr_sum[key_A] += ppr[key_A][key] ##V1: 不同metapath直接相加，这里可以考虑优化
                    # ppr_sum[key_A] += ppr[key_A][key] ##V2(6.5): 直接这么做acc很差
                    # ppr[key_A][key] = None
                ppr_sum[key_A] = ppr_sum[key_A].sum(axis = 0)
                ### ppr condense target type ###

                ## 考虑class
                a = []
                labels_train = labels[train_nid]
                for class_id, cnt in num_class_dict.items():
                    idx = train_nid[labels_train==class_id]
                    _, id = torch.topk(torch.tensor(np.asarray(ppr_sum[tgt_type]).squeeze(0)[idx]), k = cnt) 
                    a.append(idx[id])
                idx_selected = np.sort(np.concatenate(a))
                ## 考虑class
                real_reduction_rate = len(idx_selected)/node_type_nodes[tgt_type]
                print("{}: rate {}: real_reduction_rate {}".format(args.dataset, args.reduction_rate, real_reduction_rate))

                ### 不考虑class
                # reduce_nodes = sum(num_class_dict.values())
                # _, idx_selected = torch.topk(torch.tensor(np.asarray(ppr_sum[tgt_type]).squeeze(0)[train_nid]), k = reduce_nodes)   
                # idx_selected = np.sort(idx_selected.squeeze(0).numpy())
                ### 不考虑class
        
                ### PPR: condense other node types ###
                key_counter = {}
                ppr = {}
                ppr_sum = {}

                for key in adj_dict.keys():
                    if key[-1] != tgt_type:
                        key_counter.setdefault(key[-1], []).append(key)

                for key_A, key_B in key_counter.items():
                    ppr[key_A] = {}
                    ppr_sum[key_A] = 0
                    for key in key_B:
                        print(key_B,": ", key)
                        # idx=np.arange(adj_dict[key].size(0)+adj_dict[key].size(1))
                        idx=np.arange(adj_dict[key].size(0))
                        ppr[key_A][key] = topk_ppr_matrix(adj_dict[key], False, alpha=args.alpha , eps=1e-4 , idx=idx, topk=0, normalization='sym')
                        # ppr[key_A][key]= calc_ppr(adj_dict[key], key, device)  #[score_train_idx]待验证
                        ppr_sum[key_A] += ppr[key_A][key][idx_selected] ##不同metapath直接相加，这里可以考虑优化
                    ppr_sum[key_A] = ppr_sum[key_A].sum(axis = 0)  ###暂时注释是为了保存tuning结果
                torch.save(ppr_sum, f'/home/public/lyx/FreeHGC/hgb/tuning_graph/{args.dataset}/de-normalized/ppr_sum_alpha_{args.alpha}.pt')
                ppr_sum = torch.load(f'/home/public/lyx/FreeHGC/hgb/tuning_graph/{args.dataset}/de-normalized/ppr_sum_alpha_{args.alpha}.pt')
            
                candidate = {}
                candidate[tgt_type] = torch.tensor(idx_selected)

                real_reduction_rate = len(idx_selected)/node_type_nodes[tgt_type]
                for key, value in ppr_sum.items():
                    reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                    if reduce_nodes == 0:
                        _, candidate[key] = torch.topk(torch.tensor(np.asarray(ppr_sum[key]).squeeze()), k = int(0.1 * node_type_nodes[key]))
                    else:
                        _, candidate[key] = torch.topk(torch.tensor(np.asarray(ppr_sum[key]).squeeze()), k = reduce_nodes)
                    torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/hgb/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.alpha}_type_{key}.pt')                    
                ### PPR: condense other node types ###
                
                
                new_adjs = {}
                feats = {}
                feats_core = {}
                feats_extra = {}
                feats['0'] = features_list_dict_cp[tgt_type]
                for key, value in adjs.items():
                    new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
                # features_list_dict_cp = deepcopy(adj_dict)
                for key, value in features_list_dict_cp.items():
                    features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]
                    
                train_nid = np.arange(len(idx_selected))
                labels_train = labels[idx_selected].to(device)
                
                ###core-set###
                start = time.time()
                coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)
                # coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)                    
                # features_list_dict, adj_dict, extra_features_buffer = coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer
                end = time.time()
                print("Core-set: time for feature propagation", end - start)
                
                
                keys = list(coreset_features_list_dict.keys())
                print(f'For tgt {tgt_type}, feature keys {keys}')
                keys_extra = list(extra_features_buffer.keys())
                print(f'For tgt {tgt_type}, extra feature keys {keys_extra}')
                
                feats_core['0'] = features_list_dict_cp[tgt_type]
                for k in keys:
                    if args.method == 'FreeHGC' or args.method == 'random':
                        feats[k] = features_list_dict.pop(k)
                    else:
                        feats[k] = features_list_dict_type[tgt_type].pop(k)
                    feats_core[k] = coreset_features_list_dict.pop(k)
                ##feats[k], _ = v.sample_adj(init2sort, -1, False) # faster, 50% time acceleration
                # data_size = dict(dl.nodes['count'])
                data_size = {k: v.size(-1) for k, v in feats.items()}
                data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}
               
                del adjs
                del new_adjs
                features_list_dict_cp = None
        label_feats = {}
        if args.model == 'HGAMLP':
            feats_r_ensemble.append(feats)
            coreset_feats_r_ensemble.append(feats_core)
            extra_feats_r_ensemble.append(feats_extra)
            label_feats_r_ensemble.append(label_feats)
        elif args.model == 'SeHGNN':
            feats_r_ensemble = feats
            coreset_feats_r_ensemble = feats_core
            extra_feats_r_ensemble = feats_extra
            label_feats_r_ensemble = label_feats

        # =======
        # labels propagate alongside the metapath
        # =======
        num_nodes = dl.nodes['count'][0]
    ################
    g = None
    ################

    if args.dataset == 'IMDB':
        labels = labels.float().to(device)
        labels_train = labels_train.float().to(device)
    labels = labels.to(device)
    
    # Set up logging
    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.INFO)
    #logging.info(str(args))

    # _, num_feats, in_feats = feats_r_ensemble[0][0].shape
    r_len = len(r_list)
    #logging.info(f"new input size: {num_feats} {in_feats}")

    # =======
    import os
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print('checkpt_file', checkpt_file)

    # Create model
    in_feats = 512
    if args.model == 'HGAMLP':
        model = GlobalMetaAggregator(feats.keys(), feats_extra.keys(), label_feats.keys(), data_size, data_size_extra, in_feats,
                                    r_len, tgt_type, args.input_dropout, args.dropout, args.num_hidden, 
                                    num_classes, args.ff_layer_2, args.att_drop, args.sum_metapath, args.bns)
    elif args.model == 'SeHGNN':
        model = SeHGNN(in_feats, args.num_hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
            args.dropout, args.input_dropout, args.att_drop, args.label_drop,
            args.ff_layer_1, args.ff_layer_2, args.act, args.residual, bns=args.bns, data_size=data_size,
            remove_transformer=args.remove_transformer, independent_attn=args.independent_attn)
    #model = GlobalMetaAggregator(data_size, in_feats, 1, r_len, tgt_type, args.input_dropout, args.dropout, args.num_hidden, num_classes, args.ff_layer_2, args.ff_layer_2)
    logging.info("# Params: {}".format(get_n_params(model)))
    model.to(device)
    print(model)
    if len(labels.shape) == 1:
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    # Start training

    best_epoch = 0
    best_val_loss = 1000000
    best_test_loss = 0
    best_val_micro = 0
    best_test_micro = 0
    best_val_macro = 0
    best_test_macro = 0
    time_sum = 0
    for epoch in range(1, args.num_epochs + 1):
            start = time.time()
            if args.model == 'HGAMLP':
                train_micro, train_macro = train(model, coreset_feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels_train, train_nid, loss_fcn, optimizer, args.batch_size, args.dataset)
            elif args.model == 'SeHGNN':
                train_micro, train_macro = train_SeHGNN(model, coreset_feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels_train, train_nid, loss_fcn, optimizer, args.batch_size, args.dataset)
                # train_micro, train_macro = train_SeHGNN(model, feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels_train, train_nid, loss_fcn, optimizer, args.batch_size, args.dataset)

            end = time.time()
            #start = time.time()
            #if epoch % args.eval_every == 0:
            with torch.no_grad():                    
                if args.model == 'HGAMLP':
                    val_micro, test_micro, val_macro, test_macro, loss_train, loss_val, loss_test = test(
                        model, feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels, train_nid, val_nid, test_nid, loss_fcn, evaluator, args.eval_batch_size, args.micro, args.dataset)  
                elif args.model == 'SeHGNN':
                    val_micro, test_micro, val_macro, test_macro, loss_train, loss_val, loss_test = test_SeHGNN(
                        model, feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels, train_nid, val_nid, test_nid, loss_fcn, evaluator, args.eval_batch_size, args.micro, args.dataset)  
            #end = time.time()
            if epoch>1:
                time_sum = time_sum + (end - start)
            if epoch % 1 == 0:
                log = "Epoch {}, Times(s): {:.4f}".format(epoch, end - start)
                if args.micro == True:
                    log += ", mac,mic: Tra({:.4f} {:.4f}), Val({:.4f} {:.4f}), Tes({:.4f} {:.4f}) Val_loss({:.4f})".format(train_macro, train_micro, val_macro, val_micro, test_macro, test_micro, loss_val)
                logging.info(log)
            #if (args.dataset != 'Freebase' and args.dataset != 'IMDB' and loss_val <= best_val_loss) or (args.dataset == 'Freebase' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])) or (args.dataset == 'IMDB' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])): #:
            if (args.dataset != 'IMDB' and loss_val <= best_val_loss) or (args.dataset == 'IMDB' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])) or (args.dataset == 'Freebase' and sum([val_micro, val_macro]) >= sum([best_val_micro, best_val_macro])): #:
                best_epoch = epoch
                best_val_loss = loss_val
                best_test_loss = loss_test
                best_val_micro = val_micro
                best_val_macro = val_macro
                best_test_micro = test_micro
                best_test_macro = test_macro
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')     ###只存val_loss最小的模型
            if args.dataset == 'ACM' or args.dataset == 'DBLP':
                if epoch - best_epoch > args.patience:
                    time_sum = time_sum/(epoch-1)
                    break
                # elif epoch - best_epoch <= args.patience:
                #     count = count + 1
                #     print(count)
            elif args.dataset == 'IMDB':
                #args.patience = 20
                if epoch - best_epoch > args.patience: break
            elif args.dataset == 'Freebase':
                # args.patience = 20
                if epoch - best_epoch > args.patience: break
    if args.micro:
        print(f'Best Epoch {best_epoch} at {checkpt_file.split("/")[-1]}\n\t with val loss {best_val_loss:.4f} and test loss {best_test_loss:.4f}')
        logging.info("macro: Best Val {:.4f}, Best Test {:.4f}".format(best_val_macro, best_test_macro))
        logging.info("micro: Best Val {:.4f}, Best Test {:.4f}".format(best_val_micro, best_test_micro))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FreeHGC")
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--num-hops", type=int, default=3,
                        help="number of hops")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
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
    parser.add_argument("--load-feature", type=bool, default=False)
    parser.add_argument("--embed-size", type=int, default=256,
                    help="inital embedding size of nodes with no attributes")
    parser.add_argument("--input-dropout", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.5,
                        help="att dropout of attention scores")
    parser.add_argument("--SGA", action='store_true', default=False)
    parser.add_argument("--enhance", action='store_true', default=False)
    parser.add_argument("--ACM-keep-F", action='store_true', default=False)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--r", nargs='+', type=float, default=[0.0],
                        help="the seed used in the training")
    parser.add_argument("--sum-metapath", action='store_true', default=False)
    parser.add_argument("--bns", action='store_true', default=False)
    parser.add_argument("--mean", action='store_true', default=False)
    parser.add_argument("--transformer", action='store_true', default=False)
    parser.add_argument('--method', type=str, default='FreeHGC', choices=['kcenter', 'herding', 'herding_class','random', 'FreeHGC'])
    parser.add_argument("--reduction-rate", type=float, default=0.1)
    parser.add_argument("--pr", type=float, default=0.2)  ###ACM 0.85, DBLP 0.95?
    parser.add_argument("--alpha", type=float, default=0.7)
    ####SeHGNN####
    parser.add_argument("--label-drop", type=float, default=0., help="label feature dropout of model")
    parser.add_argument("--act", type=str, default='relu', help="the activation function of the model")
    parser.add_argument("--remove-transformer", action='store_true', default=False)
    parser.add_argument("--independent-attn", action='store_true', default=False)
    parser.add_argument("--residual", type=bool, default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument('--model', type=str, default='SeHGNN', choices=['HGAMLP', 'SeHGNN'])

    args = parser.parse_args()
    print(args)
    main(args)