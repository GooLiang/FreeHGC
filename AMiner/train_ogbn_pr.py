import os
import gc
import time
import uuid
import argparse
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from utils_ogbn import *
from data_ogbn import *
from data_selection_ogbn import *
# from FreeHGC.hgb.utils import *
from core_set_methods import *
from pprfile import *
from model_ogbn import *
# from hgb.model_SeHGNN import *

def main(args):
    if args.seed >= 0:
        set_random_seed(args.seed)
    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
    print("seed", args.seed)

    g, num_nodes, adjs, node_type_nodes, features_list_dict, init_labels, num_classes, train_nid, val_nid, test_nid = load_dataset(args)
    evaluator = get_ogb_evaluator(args.dataset)
    # =======
    # rearange node idx (for feats & labels)
    # =======
    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)
    # num_nodes = g.num_nodes('P')
    if total_num_nodes < num_nodes:
        flag = torch.ones(num_nodes, dtype=bool)
        flag[train_nid] = 0
        flag[val_nid] = 0
        flag[test_nid] = 0
        extra_nid = torch.where(flag)[0]
        print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
    else:
        extra_nid = torch.tensor([], dtype=torch.long)

    init2sort = torch.cat([train_nid, val_nid, test_nid, extra_nid])
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]

    features_list_dict_type = {}
    features_list_dict_cp = features_list_dict
    
    # =======
    # features propagate alongside the metapath
    # =======
    prop_tic = datetime.datetime.now()

    if args.dataset == 'ogbn-mag': # multi-node-types & multi-edge-types
        tgt_type = 'P'
    else:
        tgt_type = 'A'
        
    extra_metapath = []
    max_length = args.num_hops + 1
    prop_device = 'cuda:{}'.format(args.gpu)  ###对于dblp需要用gpu,IMDB一样可以用


    print(f'Current num hops = {args.num_hops}')

    if args.method == 'FreeHGC':
        num_class_dict = Base(init_labels, train_nid, args, device).num_class_dict
        print("real reduction rate", len(train_nid)*args.reduction_rate/num_nodes)
        # compute k-hop feature
        features_list_dict_cp  = deepcopy(features_list_dict)
        # g = hg_propagate(g, tgt_type, args.num_hops, max_hops, extra_metapath, echo=False)
        start = time.time()
        features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, prop_device, prop_feats=True, echo=True)
        end = time.time()
        print("time for feature propagation", end - start)
    
        import copy
        budget = int(args.reduction_rate * len(train_nid))  #wrong node_type_nodes[tgt_type]
        ###############################
        ###condense target node type###
        key_counter = {}
        ppr = {}
        ppr_sum = {}
        for key in adj_dict.keys():
                key_counter.setdefault(key[-1], []).append(key)


        ppr = {}
        ppr_sum = {}
        key_A = tgt_type
        ppr[key_A] = {}
        ppr_sum[key_A] = 0
        for key in key_counter[tgt_type]:
            idx=np.arange(adj_dict[key].size(0)+adj_dict[key].size(1))
            flag = key[0] == key[-1]
            ppr[key_A][key] = topk_ppr_matrix(adj_dict[key], flag, alpha=args.alpha , eps=1e-4 , idx=idx, topk=0, normalization='sym')
            ppr_sum[key_A] += ppr[key_A][key]
            
        ppr_sum[key_A] = ppr_sum[key_A].sum(axis = 0)
        torch.save(ppr_sum[key_A], f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/hop_{args.num_hops}/{args.dataset}_tgt_ppr_alpha_{args.alpha}.pt')                                
        ppr_sum[tgt_type] = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/hop_{args.num_hops}/{args.dataset}_tgt_ppr_alpha_{args.alpha}.pt')                                

        
        ## 考虑class
        a = []
        labels_train = init_labels[train_nid]
        for class_id, cnt in num_class_dict.items():
            idx = train_nid[labels_train==class_id]
            _, id = torch.topk(torch.tensor(np.asarray(ppr_sum[tgt_type]).squeeze(0)[idx]), k = cnt) 
            a.append(idx[id])
        idx_selected = np.sort(np.concatenate(a))
        torch.save(idx_selected, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/hop_{args.num_hops}/{args.dataset}_cond_{args.reduction_rate}_ppr_idx_selected_{args.alpha}.pt')  
        idx_selected = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/hop_{args.num_hops}/{args.dataset}_cond_{args.reduction_rate}_ppr_idx_selected_{args.alpha}.pt')
        ## 考虑class
        
        
        ### 不考虑class
        # reduce_nodes = sum(num_class_dict.values())
        # _, idx_selected = torch.topk(torch.tensor(np.asarray(ppr_sum[tgt_type]).squeeze(0)[train_nid]), k = reduce_nodes)   
        # idx_selected = np.sort(idx_selected.squeeze(0).numpy())
        ### 不考虑class
        
        real_reduction_rate = len(idx_selected)/node_type_nodes[tgt_type]
        print("real reduction rate", real_reduction_rate)
        
        candidate = {}
        candidate[tgt_type] = idx_selected
        
        key_counter = {}
        for key in adj_dict.keys():
            # if key[-1] != tgt_type:
            key_counter.setdefault(key[-1], []).append(key)

        # ## PPR: condense other node types ###
        ppr = {}
        ppr_sum = {}
        for key_A, key_B in key_counter.items():
            if key_A != tgt_type:
                ppr[key_A] = {}
                ppr_sum[key_A] = 0
                for key in key_B:
                    print(key_B,": ", key)
                    idx=np.arange(adj_dict[key].size(0)+adj_dict[key].size(1))
                    ppr[key_A][key] = topk_ppr_matrix(adj_dict[key], False, alpha=args.alpha , eps=1e-4 , idx=idx, topk=0, normalization='sym')
                    ppr_sum[key_A] += ppr[key_A][key][idx_selected]
                ppr_sum[key_A] = ppr_sum[key_A].sum(axis = 0)

        torch.save(ppr_sum, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/hop_{args.num_hops}/{args.dataset}_cond_{args.reduction_rate}_ppr_sum_alpha_{args.alpha}.pt')                                
        ppr_sum = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/hop_{args.num_hops}/{args.dataset}_cond_{args.reduction_rate}_ppr_sum_alpha_{args.alpha}.pt')
        # ## PPR: condense other node types ###

        for key, value in ppr_sum.items():
            reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
            if reduce_nodes == 0:
                _, candidate[key] = torch.topk(torch.tensor(np.asarray(ppr_sum[key]).squeeze()), k = int(0.1 * node_type_nodes[key]))
            else:
                _, candidate[key] = torch.topk(torch.tensor(np.asarray(ppr_sum[key]).squeeze()), k = reduce_nodes)
            torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/ogbn/condense_graph/hop_{args.num_hops}/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_alpha_{args.alpha}_type_{key}.pt')             
        
        if args.dataset == 'aminer':
            new_adjs = {}
            for key, value in adjs.items():
                if key == 'PA' or key == 'AP':
                    new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
                elif key == 'VP':
                    new_adjs[key] = adjs[key][:, candidate[key[1]]]
                elif key == 'PV':
                    new_adjs[key] = adjs[key][candidate[key[0]], :]
                    
            for key, value in features_list_dict_cp.items():
                if key == 'A':
                    features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]
        
        ### LOOSE VERSION ###
        # def relation_condition(key):
        #     if args.dataset == 'ogbn-mag':
        #         key_A = ['PF', 'AI']   ### PA用2，PF,AI用3
        #     if args.dataset == 'aminer':
        #         key_A = ['PF', 'AI']   ### PA用2，AV用3
        #     if key in  key_A:
        #         return True
        #     else:
        #         return False
            
        # new_index = {}
        # edge_count = {}
        # features_new = {}
        # for key_A, val in new_adjs.items():
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
        #                 a = list(new_index[key_A[1]].keys())[pool.index(value)]
        #                 new_index[key_A[1]][key] = new_index[key_A[1]][a]
        #                 pool.append(value)
        #             else:
        #                 new_index[key_A[1]][key] = i
        #                 edge_count[key_A[1]][i] = len(value)
        #                 features_new[key_A[1]].append(torch.sum(features_list_dict_cp[key_A[1]][value], dim = 0)/len(value))   ### /长度是V1版本
        #                 i = i + 1
        #                 pool.append(value)
        #         row = list(new_index[key_A[1]].keys())
        #         col = list(new_index[key_A[1]].values())
        #         new_adjs[key_A] = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=(max(row)+1, i))
        #         new_adjs[key_A[::-1]] = SparseTensor(row=torch.LongTensor(col), col=torch.LongTensor(row), sparse_sizes=(i, max(row)+1))
        #         # new_adjs[key_A].storage.row = torch.tensor(list(edge_count[key_A[1]].values()))
        #         features_list_dict_cp[key_A[1]] = torch.stack(features_new[key_A[1]])
        
        # torch.save(new_adjs, f'/home/public/lyx/FreeHGC/ogbn/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_new_adjs.pt')
        # torch.save(features_list_dict_cp, f'/home/public/lyx/FreeHGC/ogbn/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_new_features.pt')
        # new_adjs =  torch.load(f'/home/public/lyx/FreeHGC/ogbn/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_new_adjs.pt')
        # features_list_dict_cp = torch.load(f'/home/public/lyx/FreeHGC/ogbn/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_new_features.pt')
        ### LOOSE VERSION ###

        # sum_node = new_adjs['PF'].size(0) + new_adjs['PF'].size(1) +new_adjs['AI'].size(0) + new_adjs['AI'].size(1)

        ### origin ###
        new_adjs = {}
        for key, value in adjs.items():
            new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
        for key, value in features_list_dict_cp.items():
            features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]
        ### origin ###
            

        # ### only target ###
        # new_adjs = {}
        # for key, value in adjs.items():
        #     if key[0] == tgt_type and key[1] != tgt_type:
        #         new_adjs[key] = adjs[key][candidate[tgt_type], :]
        #     elif key[1] == tgt_type and key[0] != tgt_type:
        #         new_adjs[key] = adjs[key][:, candidate[tgt_type]]
        #     elif key[0] == key[1]:
        #         new_adjs[key] = adjs[key][candidate[tgt_type]][:, candidate[tgt_type]]
        #     else:
        #         new_adjs[key] = adjs[key]
        # features_list_dict_cp[tgt_type] = features_list_dict_cp[tgt_type][candidate[tgt_type], :]
        # ### only target ###
        
        ###core-set###
        start = time.time()
        coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, prop_device, prop_feats=True, echo=True)
        # coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)                    
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
        
        train_nid = np.arange(len(idx_selected))
        labels_train = init_labels[idx_selected].to(device)

            
    if args.method == 'random':
        agent = Random(init_labels, train_nid, args, device)
        start = time.time()
        features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, prop_device, prop_feats=True, echo=True)
        end = time.time()
        
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
                
        new_adjs = {}
        for key, value in adjs.items():
            new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]
        for key, value in features_list_dict_cp.items():
            features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]
                        
        ###core-set###
        start = time.time()
        coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, prop_device, prop_feats=True, echo=True)
        # coreset_features_list_dict, coreset_adj_dict, coreset_extra_features_buffer = hg_propagate_sparse_pyg_A(new_adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)                    
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
        
        train_nid = np.arange(len(idx_selected))
        labels_train = init_labels[idx_selected].to(device)
        
    if args.method == 'kcenter':
        agent = KCenter(init_labels, train_nid, args, device)
        start = time.time()
        for tgt_node_key in node_type_nodes:  ###求所有节点类型的特征
            # compute k-hop feature
            new_g = hg_propagate_dgl(g.clone(), tgt_node_key, args.num_hops, max_length, extra_metapath, echo=True)
            feats = {}
            keys = list(new_g.nodes[tgt_node_key].data.keys())
            print(f'Involved feat keys {keys}')
            for k in keys:
                feats[k] = new_g.nodes[tgt_node_key].data.pop(k)
            features_list_dict_type[tgt_node_key] = feats
        end = time.time()
        print("time for feature propagation", end - start)
    if args.method == 'herding':
        agent = Herding(init_labels, train_nid, args, device)
        start = time.time()
        for tgt_node_key in node_type_nodes:  ###求所有节点类型的特征
            # compute k-hop feature
            new_g = hg_propagate_dgl(g.clone(), tgt_node_key, args.num_hops, max_length, extra_metapath, echo=True)
            feats = {}
            keys = list(new_g.nodes[tgt_node_key].data.keys())
            print(f'Involved feat keys {keys}')
            for k in keys:
                feats[k] = new_g.nodes[tgt_node_key].data.pop(k)
            features_list_dict_type[tgt_node_key] = feats
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
            torch.save(dis_dict_sum, f'/home/public/lyx/FreeHGC/ogbn/condense_graph/herding/dic_{args.dataset}_hops_{args.num_hops}_rrate_{args.reduction_rate}.pt')                    
        else:
            torch.save(dis_dict_sum, f'/home/public/lyx/FreeHGC/ogbn/condense_graph/kcenter/dic_{args.dataset}_hops_{args.num_hops}_rrate_{args.reduction_rate}.pt')  
        return 0

    if args.dataset == 'ogbn-mag':
        g = clear_hg(g, echo=False)


    feats = {k: v[init2sort] for k, v in feats.items()}

    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # train_loader = torch.utils.data.DataLoader(
    #     torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
    # eval_loader = full_loader = []
    all_loader = torch.utils.data.DataLoader(
        torch.arange(num_nodes), batch_size=args.batch_size, shuffle=False, drop_last=False)

    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
    labels_cuda = labels.long().to(device)

    checkpt_file = checkpt_folder + uuid.uuid4().hex
    print(checkpt_file)

    for stage in range(0,1):
        epochs = args.stages[stage]

        if len(args.reload):
            pt_path = f'output/ogbn-mag/{args.reload}_{stage-1}.pt'
            assert os.path.exists(pt_path)
            print(f'Reload raw_preds from {pt_path}', flush=True)
            raw_preds = torch.load(pt_path, map_location='cpu')

        # =======
        # Expand training set & train loader
        # =======
        # ### 原图 ###
        # train_loader = torch.utils.data.DataLoader(
        #     torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
        # ### 原图 ###
        
        ### 小图 ###
        train_loader = torch.utils.data.DataLoader(
            torch.arange(len(idx_selected)), batch_size=args.batch_size, shuffle=True, drop_last=False)
        ### 小图 ###
        
        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        label_emb = torch.zeros((num_nodes, num_classes))
        label_emb_corset = torch.zeros((len(idx_selected), num_classes))
        label_feats = {k: v[init2sort] for k, v in label_feats.items()}
        label_emb = label_emb[init2sort]

        if stage == 0:
            label_feats = {}

        # =======
        # Eval loader
        # =======
        eval_loader = []
        for batch_idx in range((num_nodes-trainval_point-1) // args.batch_size + 1):
            batch_start = batch_idx * args.batch_size + trainval_point
            batch_end = min(num_nodes, (batch_idx+1) * args.batch_size + trainval_point)

            batch_feats = {k: v[batch_start:batch_end] for k,v in feats.items()}
            batch_label_feats = {k: v[batch_start:batch_end] for k,v in label_feats.items()}
            batch_labels_emb = label_emb[batch_start:batch_end]
            eval_loader.append((batch_feats, batch_label_feats, batch_labels_emb))

        # =======
        # Construct network
        # =======
        model = SeHGNN_mag(args.dataset,
            data_size, args.embed_size,
            args.hidden, num_classes,
            len(feats), len(label_feats), tgt_type,
            dropout=args.dropout,
            input_drop=args.input_drop,
            att_drop=args.att_drop,
            label_drop=args.label_drop,
            n_layers_1=args.n_layers_1,
            n_layers_2=args.n_layers_2,
            n_layers_3=args.n_layers_3,
            act=args.act,
            residual=args.residual,
            bns=args.bns, label_bns=args.label_bns,
            # label_residual=stage > 0,
            )
        model = model.to(device)
        if stage == args.start_stage:
            print(model)
            print("# Params:", get_n_params(model))

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        best_epoch = 0
        best_val_acc = 0
        best_test_acc = 0
        count = 0

        for epoch in range(epochs):
            gc.collect()
            torch.cuda.empty_cache()
            start = time.time()
            loss, acc = train(model, train_loader, loss_fcn, optimizer, evaluator, device, feats_core, label_feats, labels_train, label_emb_corset, scalar=scalar)  ###labels_train应该是优化好的
            # loss, acc = train(model, train_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, scalar=scalar)  ###训练全图
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(epoch, end-start, loss, acc*100)
            torch.cuda.empty_cache()

            if epoch % args.eval_every == 0:
                with torch.no_grad():
                    model.eval()
                    raw_preds = []

                    start = time.time()
                    for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
                        batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                        batch_label_feats = {k: v.to(device) for k,v in batch_label_feats.items()}
                        batch_labels_emb = batch_labels_emb.to(device)
                        raw_preds.append(model(batch_feats, batch_label_feats, batch_labels_emb).cpu())
                    raw_preds = torch.cat(raw_preds, dim=0)

                    loss_val = loss_fcn(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
                    loss_test = loss_fcn(raw_preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes]).item()

                    preds = raw_preds.argmax(dim=-1)
                    val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
                    test_acc = evaluator(preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes])

                    end = time.time()
                    log += f'Time: {end-start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
                    log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc*100, test_acc*100)

                if val_acc > best_val_acc:
                    best_epoch = epoch
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                    torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
                    count = 0
                else:
                    count = count + args.eval_every
                    if count >= args.patience:
                        break
                log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100)
            print(log, flush=True)

        print("Best Epoch Stage {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100))

        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
        # raw_preds = gen_output_torch(model, feats, label_feats, label_emb, all_loader, device)
        # torch.save(raw_preds, checkpt_file+f'_{stage}.pt')


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[1],
                        help="the seed used in the training")
    # parser.add_argument("--seeds", type=int, default=None,
    #                     help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="aminer")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default='/home/public/lyx/SeHGNN_new/SeHGNN/data/')
    parser.add_argument("--stages", nargs='+',type=int, default=[300],
                        help="The epoch setting for each stage.")
    ## For pre-processing
    parser.add_argument("--emb_path", type=str, default='../data/')
    parser.add_argument("--extra-embedding", type=str, default='',
                        help="the name of extra embeddings")
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-1", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--n-layers-2", type=int, default=2,
                        help="number of layers of the downstream task")
    parser.add_argument("--n-layers-3", type=int, default=2,
                        help="number of layers of residual label connection")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,
                        help="label feature dropout of model")
    parser.add_argument("--residual", action='store_true', default=True,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=True,
                        help="whether to process the input features")
    parser.add_argument("--label-bns", action='store_true', default=False,
                        help="whether to process the input label features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="the threshold of multi-stage learning, confident nodes "
                           + "whose score above this threshold would be added into the training set")
    parser.add_argument("--gama", type=float, default=0.5,
                        help="parameter for the KL loss")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--reload", type=str, default='')
    parser.add_argument("--sum-meta", action='store_true', default=False)
    parser.add_argument('--method', type=str, default='FreeHGC', choices=['kcenter', 'herding', 'herding_class','random', 'FreeHGC'])
    parser.add_argument("--reduction-rate", type=float, default=0.05) ## mag 0.2
    parser.add_argument("--alpha", type=float, default=0.15)  #default=0.15
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    # assert args.dataset.startswith('ogbn')
    print(args)

    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
 
        main(args)
