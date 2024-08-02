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


    feats_r_ensemble = []
    coreset_feats_r_ensemble = []
    extra_feats_r_ensemble = []
    label_feats_r_ensemble = []
    features_list_dict_type = {}
    features_list_dict_cp = features_list_dict
    
    # =======
    # features propagate alongside the metapath
    # =======
    prop_tic = datetime.datetime.now()

    if args.dataset == 'ogbn-mag': # multi-node-types & multi-edge-types
        tgt_type = 'P'
        extra_metapath = []
        max_length = args.num_hops + 1
        prop_device = 'cuda:{}'.format(args.gpu)  ###对于dblp需要用gpu,IMDB一样可以用


        print(f'Current num hops = {args.num_hops}')

        if args.method == 'FreeHGC':
            num_class_dict = Base(init_labels, train_nid, args, device).num_class_dict

            # compute k-hop feature
            features_list_dict_cp  = deepcopy(features_list_dict)
            # g = hg_propagate(g, tgt_type, args.num_hops, max_hops, extra_metapath, echo=False)
            start = time.time()
            features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, prop_device, prop_feats=True, echo=True)
            ##PFP可以用另一种方法得到，dgl只有对角线的特征为1
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

            ## jaccord score ###
            jaccard_score_buffer_sum = {}
            for key_A, key_B in key_counter.items():
                jaccard_score_buffer_sum[key_A] = {}
                if len(key_B) == 1:
                    jaccard_score_buffer_sum[key_A][''.join(key_B)] = torch.zeros(adj_dict[''.join(key_B)].size(0))
                if len(key_B)>1:
                    for i, key in enumerate(key_B):
                        print("1: ", key)
                        # a = adj_dict[key].to_dense()  #((adj_dict[key].to_dense()!=0)+0).cpu().numpy()

                        sorted_indices = torch.argsort(adj_dict[key].storage.row())
                        sorted_row = adj_dict[key].storage.row()[sorted_indices]
                        sorted_col = adj_dict[key].storage.col()[sorted_indices]
                        unique_row_A, counts = torch.unique(sorted_row, return_counts=True)  #unique_row_A, 具体哪个节点
                        groups_A = torch.split(sorted_col, counts.tolist()) ##groups_A: unique_row_A所对应的每个节点所具有的边
                        a_now = dict(zip(unique_row_A.numpy(), groups_A))
                        jaccard_score_buffer_sum[key_A][key] = 0
                        for key_compare in key_B:
                            if key_compare != key:
                                print("2: ",key_compare)
                                sum_tmp = torch.zeros(adj_dict[key_compare].size(0))
                                sorted_indices = torch.argsort(adj_dict[key_compare].storage.row())
                                sorted_row = adj_dict[key_compare].storage.row()[sorted_indices]
                                sorted_col = adj_dict[key_compare].storage.col()[sorted_indices]
                                unique_row_B, counts = torch.unique(sorted_row, return_counts=True)
                                groups_B = torch.split(sorted_col, counts.tolist()) ##每个节点所具有的边
                                b = dict(zip(unique_row_B.numpy(), groups_B))
                                groups_A_B = list(set(unique_row_A.numpy()).intersection(set(unique_row_B.numpy()))) #共同有交集的节点
                                a = [a_now[key].numpy() for key in groups_A_B]
                                b = [b[key].numpy() for key in groups_A_B]
                                k1 = [len(set(a[i])&set(b[i])) for i in range(len(a))]
                                k2 = [len(set(a[i])|set(b[i])) for i in range(len(a))]
                                tmp = torch.tensor(k1)/torch.tensor(k2)
                                tmp = torch.where(torch.isnan(tmp), torch.full_like(tmp, 1), tmp) ### debug去掉nan
                                sum_tmp[groups_A_B] = tmp
                                jaccard_score_buffer_sum[key_A][key] += sum_tmp
                    jaccard_score_buffer_sum[key_A][key] = jaccard_score_buffer_sum[key_A][key]/(len(key_B) - 1)
        ### jaccord score ###

            torch.save(jaccard_score_buffer_sum, f'/home/public/lyx/FreeHGC/ogbn/jaccard_score_buffer_sum.pt')                    
            jaccard_score_buffer_sum = torch.load(f'/home/public/lyx/FreeHGC/ogbn/jaccard_score_buffer_sum.pt')

            idx_train_nnd_metapth = defaultdict(list)
            score_train_nnd_metapth = defaultdict(list)
            score_train_nnd_current = defaultdict(list)
            score_train_nnd_ratio = defaultdict(list)
            labels_train = init_labels[train_nid]
            for key, adj in adj_dict.items():
                idx_train_nnd_metapth[key] = {}
                score_train_nnd_metapth[key] = {}
                score_train_nnd_current[key] = {}
                score_train_nnd_ratio[key] = {}
                print("============== new =============")
                # weighted_score_B = torch.ones(adj.size(1), dtype=torch.float32).to(device)
                for class_id, cnt in num_class_dict.items():
                    idx = train_nid[labels_train==class_id]
                    idx_train_nnd = []
                    score_train_nnd = []
                    score_train_nnd_1 = []
                    score_train_nnd_2 = []
                    idx_avaliable_temp = copy.deepcopy(list(idx))
                    # adj = adj.to_dense()
                    t = perf_counter()
                    for count in range(cnt):
                        max_receptive_node, max_total_score, max_node_score, max_expand_ratio = get_max_nnd_node_sparse(idx_train_nnd,idx_avaliable_temp, adj, jaccard_score_buffer_sum[key[-1]][key])
                        idx_train_nnd.append(max_receptive_node)
                        score_train_nnd.append(max_total_score.item())  ###socre_version1
                        score_train_nnd_1.append(max_node_score)  ##socre_version2: more reasonable
                        score_train_nnd_2.append(max_expand_ratio)  ##socre_version2: more more reasonable
                        idx_avaliable_temp.remove(max_receptive_node)
                    print("time budget:", perf_counter()-t)
                    idx_train_nnd_metapth[key][class_id] = idx_train_nnd
                    score_train_nnd_metapth[key][class_id] = score_train_nnd
                    score_train_nnd_current[key][class_id] = score_train_nnd_1
                    score_train_nnd_ratio[key][class_id] = score_train_nnd_2
                torch.save(idx_train_nnd_metapth, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}/idx_train_nnd_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')                    
                torch.save(score_train_nnd_metapth, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}/score_train_nnd_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
                torch.save(score_train_nnd_current, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}/score_train_nnd_current_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')                    
                torch.save(score_train_nnd_ratio, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}/score_train_nnd_ratio_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
            idx_train_nnd_metapth = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}/normalized/idx_train_nnd_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
            score_train_nnd_metapth = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}/normalized/score_train_nnd_ratio_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
            # return 0
            # idx_selected = []
            # score_train_idx_sum = defaultdict(list)
            # for class_id, class_budget in num_class_dict.items():
            #     score_train_idx_sum[class_id] = None
            #     for key,value in idx_train_nnd_metapth.items():
            #         # score_train_idx = dict(zip(value[class_id], [i+1e-12 for i in score_train_nnd_metapth[key][class_id]]))
            #         score_train_idx = dict(zip(value[class_id], score_train_nnd_metapth[key][class_id]))  ###score_train_nnd_metapth
            #         score_train_idx_sum[class_id] = dict(Counter(score_train_idx_sum[class_id]) + Counter(score_train_idx))
            #     ###### 对每个class_id的所有meta的结果score_train_idx_sum求topk
            #     # score_train_idx = torch.argsort(torch.tensor(list(score_train_idx_sum[class_id].values())), descending=True)
            #     _, score_train_idx = torch.topk(torch.tensor(list(score_train_idx_sum[class_id].values())), class_budget)
            #     score_train_idx = torch.tensor(list(score_train_idx_sum[class_id].keys()))[score_train_idx]
            #     idx_selected.append(score_train_idx)
            # idx_selected = torch.cat(idx_selected, dim = 0).numpy()
            # assert set(idx_selected) < set(train_nid.numpy())
            # assert Counter(init_labels[idx_selected].numpy()) == num_class_dict
            # torch.save(idx_selected, f'/home/public/lyx/FreeHGC/ogbn/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_type_{tgt_type}.pt')

            # # # ###condense target node type###


            ### PPR: condense other node types ###
            # key_counter = {}
            # ppr = {}
            # ppr_sum = {}
            
            # for key in adj_dict.keys():
            #     # if key[-1] != tgt_type:
            #     key_counter.setdefault(key[-1], []).append(key)

            # for key_A, key_B in key_counter.items():
            #     if key_A == tgt_type:
            #         ppr[key_A] = {}
            #         ppr_sum[key_A] = 0
            #         for key in key_B:
            #             print(key_B,": ", key)
            #             idx=np.arange(adj_dict[key].size(0)+adj_dict[key].size(1))
            #             ppr[key_A][key] = topk_ppr_matrix(adj_dict[key], alpha=args.alpha , eps=1e-4 , idx=idx, topk=0, normalization='sym')
            #             # ppr[key_A][key]= calc_ppr(adj_dict[key], key, device)  #[score_train_idx]待验证
            #             ppr_sum[key_A] += ppr[key_A][key] ##不同metapath直接相加，这里可以考虑优化
            #             # ppr[key_A][key] = None
            #         ppr_sum[key_A] = ppr_sum[key_A].sum(axis = 0)
            # topk_num = sum(num_class_dict.values())
            # candidate_nodes = np.asarray(ppr_sum[tgt_type]).squeeze()[train_nid]
            # _, idx_selected = torch.topk(torch.tensor(candidate_nodes), k = topk_num)
            # idx_selected = train_nid[idx_selected]
            # torch.save(idx_selected, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}_ppr_idx_selected_{args.alpha}.pt')  
            idx_selected = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}_ppr_idx_selected_{args.alpha}.pt')
            idx_selected = idx_selected.numpy()
            for key_A, key_B in key_counter.items():
                if key_A != tgt_type:
                    ppr[key_A] = {}
                    ppr_sum[key_A] = 0
                    for key in key_B:
                        print(key_B,": ", key)
                        idx=np.arange(adj_dict[key].size(0)+adj_dict[key].size(1))
                        ppr[key_A][key] = topk_ppr_matrix(adj_dict[key], alpha=args.alpha , eps=1e-4 , idx=idx, topk=0, normalization='sym')
                        # ppr[key_A][key]= calc_ppr(adj_dict[key], key, device)  #[score_train_idx]待验证
                        ppr_sum[key_A] += ppr[key_A][key][idx_selected] ##不同metapath直接相加，这里可以考虑优化
                        # ppr[key_A][key] = None
                    ppr_sum[key_A] = ppr_sum[key_A].sum(axis = 0)
                    # ppr_sum[key_A] = torch.sum(ppr_sum[key_A], dim = 0)
            torch.save(ppr_sum, f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}_ppr_sum_alpha_{args.alpha}.pt')                                
            ppr_sum = torch.load(f'/home/public/lyx/FreeHGC/ogbn/tuning_graph/{args.dataset}_ppr_sum_alpha_{args.alpha}.pt')
            ### PPR: condense other node types ###
            
            candidate = {}
            candidate[tgt_type] = idx_selected

            real_reduction_rate = len(idx_selected)/node_type_nodes[tgt_type]
            for key, value in ppr_sum.items():
                reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                if reduce_nodes == 0:
                    _, candidate[key] = torch.topk(torch.tensor(np.asarray(ppr_sum[key]).squeeze()), k = int(0.1 * node_type_nodes[key]))
                else:
                    _, candidate[key] = torch.topk(torch.tensor(np.asarray(ppr_sum[key]).squeeze()), k = reduce_nodes)
                # torch.save(candidate[key].numpy(), f'/home/public/lyx/FreeHGC/ogbn/condense_graph/{args.dataset}/hops_{args.num_hops}_rrate_{args.reduction_rate}_pr_{args.pr}_type_{key}.pt')             
            
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

        g = clear_hg(g, echo=False)
    else:
        assert 0

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

    for stage in range(args.start_stage, len(args.stages)):
        epochs = args.stages[stage]

        if len(args.reload):
            pt_path = f'output/ogbn-mag/{args.reload}_{stage-1}.pt'
            assert os.path.exists(pt_path)
            print(f'Reload raw_preds from {pt_path}', flush=True)
            raw_preds = torch.load(pt_path, map_location='cpu')

        # =======
        # Expand training set & train loader
        # =======

        # train_loader = torch.utils.data.DataLoader(
        #     torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            torch.arange(len(idx_selected)), batch_size=args.batch_size, shuffle=True, drop_last=False)
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
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default='/home/public/lyx/SeHGNN_new/SeHGNN/data/')
    parser.add_argument("--stages", nargs='+',type=int, default=[300, 300],
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
    parser.add_argument("--n-layers-3", type=int, default=4,
                        help="number of layers of residual label connection")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--label-drop", type=float, default=0.,
                        help="label feature dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=False,
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
    parser.add_argument("--reduction-rate", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.15)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    assert args.dataset.startswith('ogbn')
    print(args)

    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
 
        main(args)
