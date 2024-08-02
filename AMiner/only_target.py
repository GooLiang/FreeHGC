from openbox import sp, Optimizer
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import logging
import uuid
from model_hgb import *
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
from sparse_tools import SparseAdjList
from data_selection import *
from collections import Counter

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.gpu < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}"
    print("seed", args.seed)
    # Load dataset
    g, adjs, features_list_dict, node_type_nodes, edge_type_ratio, labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full \
        = load_dataset(args)
    evaluator = get_evaluator(args.dataset)
    g = None
    import os
    if args.dataset == 'Freebase':
        if not os.path.exists('./Freebase_adjs'):
            os.makedirs('./Freebase_adjs')
            
        num_tgt_nodes = dl.nodes['count'][0]

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
    r = 0.0
    print(r)
    feats_r_ensemble = []
    coreset_feats_r_ensemble = []
    extra_feats_r_ensemble = []
    label_feats_r_ensemble = []
    features_list_dict_cp = features_list_dict
    with torch.no_grad():
        #############normalization---wait to modify#########################
        if args.dataset != "Freebase":
            prop_device = 'cuda:{}'.format(args.gpu)  ###对于dblp需要用gpu,IMDB一样可以用
            threshold_metalen = args.threshold
            start = time.time()
            features_list_dict, adj_dict, extra_features_buffer = hg_propagate_sparse_pyg_A(adjs, features_list_dict_cp, tgt_type, args.num_hops, max_length, extra_metapath, threshold_metalen, prop_device, args.enhance, prop_feats=True, echo=True)                    
            end = time.time()
            print("time for feature propagation", end - start)

            ###core-set###
            
            if args.method == 'kcenter':
                agent = KCenter(labels, train_nid, args, device)
            if args.method == 'herding':
                agent = Herding(labels, train_nid, args, device)
            if args.method == 'herding_class':
                agent = Herding_class(labels, train_nid, args, device)
            if args.method == 'random':
                agent = Random(labels, train_nid, args, device)
            if args.method == 'FreeHGC':
                num_class_dict = Base(labels, train_nid, args, device).num_class_dict
            idx_selected_dict = {}
            if args.method == 'herding' or args.method == 'kcenter':
                if args.method == 'herding':
                    flag = False
                if args.method == 'kcenter':
                    flag = True
                ### topk ###
                dis_dict_sum = {}
                for key in features_list_dict:
                    dis_dict = agent.select_top(features_list_dict[key])
                    dis_dict_sum[key] = dis_dict
                idx_selected = []
                score_train_idx_sum = defaultdict(list)
                for class_id, class_budget in agent.num_class_dict.items():
                    score_train_idx_sum[class_id] = None
                    for key,value in dis_dict_sum.items():        
                        score_train_idx = dis_dict_sum[key][class_id]
                        score_train_idx_sum[class_id] = dict(Counter(score_train_idx_sum[class_id]) + Counter(score_train_idx))
                    ###### 对每个class_id的所有meta的结果score_train_idx_sum求topk
                    # score_train_idx = torch.argsort(torch.tensor(list(score_train_idx_sum[class_id].values())), descending=True)
                    _, score_train_idx = torch.topk(torch.tensor(list(score_train_idx_sum[class_id].values())), class_budget, largest = flag)
                    score_train_idx = torch.tensor(list(score_train_idx_sum[class_id].keys()))[score_train_idx]
                    idx_selected.append(score_train_idx)
                idx_selected = torch.cat(idx_selected, dim = 0).numpy()
                ### topk ###

                # ## 并差集 ###
                # for key in features_list_dict:
                #     idx_selected = agent.select(features_list_dict[key])
                #     idx_selected_dict[key] = idx_selected
                # all_idx = [index for key, index in idx_selected_dict.items()]
                # inter_idx = list(set(all_idx[0]).intersection(*all_idx[1:]))
                # union_idx = list(set(all_idx[0]).union(*all_idx[1:]))
                # symm_idx = list(set(all_idx[0]).symmetric_difference(all_idx[1]))
                # for idx in all_idx[2:]:
                #     symm_idx = list(set(symm_idx).symmetric_difference(idx))
                # symm_idx.extend(inter_idx)   #a = list(set(symm_idx).union(inter_idx))
                # idx_selected = symm_idx
                # ## 并差集 ###

                # dis_dict_others = {}
                # dis_dict_others[tgt_type] = idx_selected
                # real_reduction_rate = sum(agent.num_class_dict.values())/node_type_nodes[tgt_type]
                # for key, value in features_list_dict_cp.items():
                #     reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                #     if key != tgt_type:
                #         if reduce_nodes == 0:
                #             dis_dict_others[key] = agent.select_other_types(features_list_dict_cp[key], int(0.1 * node_type_nodes[key]))
                #         else:
                #             dis_dict_others[key] = agent.select_other_types(features_list_dict_cp[key], reduce_nodes)

                # new_adjs = {}
                # for key, value in adjs.items():
                #     new_adjs[key] = adjs[key][dis_dict_others[key[0]]][:, dis_dict_others[key[1]]]

                    
                for key, adj in adjs.items():
                    if key[0]==tgt_type and key[-1]!=tgt_type:
                        adjs[key] = adj[idx_selected]
                    if key[-1]==tgt_type and key[0]!=tgt_type:
                        adjs[key]  = adj[:,idx_selected]
                    if key[0]==tgt_type and key[-1]==tgt_type:
                        adjs[key]  = adj[idx_selected][:,idx_selected]
                new_adjs = adjs
                features_list_dict_cp[tgt_type] = features_list_dict_cp[tgt_type][idx_selected]
            elif args.method == 'herding_class':
                key_counter = {}
                a = {}
                for key in features_list_dict.keys():
                    key_counter.setdefault(key[-1], []).append(key)
                for key, value in key_counter.items():
                    a[key] = 0
                for class_id, cnt in agent.num_class_dict.items():
                    class_selected = agent.select(class_id, cnt, features_list_dict, key_counter)
                    for key, value in key_counter.items():
                        for class_key, class_value in class_selected.items():
                            if class_key in value:
                                a[key] += class_value
            elif args.method == 'FreeHGC':
                import copy
                from sklearn.metrics import jaccard_score
                budget = int(args.reduction_rate * len(train_nid))  #wrong node_type_nodes[tgt_type]
                ###############################
                ###condense target node type###
                key_counter = {}
                ppr = {}
                ppr_sum = {}
                for key in adj_dict.keys():
                        key_counter.setdefault(key[-1], []).append(key)

                ### jaccord score ###
                jaccard_score_buffer_sum = {}
                for key_A, key_B in key_counter.items():
                    jaccard_score_buffer_sum[key_A] = {}
                    if len(key_B) == 1:
                        jaccard_score_buffer_sum[key_A][''.join(key_B)] = torch.zeros(adj_dict[''.join(key_B)].size(0))
                    if len(key_B)>1:
                        for i, key in enumerate(key_B):
                            print("1: ", key)
                            a = adj_dict[key].to_dense()  #((adj_dict[key].to_dense()!=0)+0).cpu().numpy()
                            jaccard_score_buffer_sum[key_A][key] = 0
                            for key_compare in key_B:
                                if key_compare != key:
                                    print("2: ",key_compare)
                                    b = adj_dict[key_compare].to_dense() #((adj_dict[key_compare].to_dense()!=0)+0).cpu().numpy()
                                    intersection = torch.logical_and(a, b)
                                    union = torch.logical_or(a, b)
                                    tmp = torch.sum(intersection, dim=1) / torch.sum(union, dim=1)  #jaccard_score(a, b, average='samples')
                                    jaccard_score_buffer_sum[key_A][key] += tmp
                            jaccard_score_buffer_sum[key_A][key] = jaccard_score_buffer_sum[key_A][key]/(len(key_B) - 1)
                ### jaccord score ###
                
                idx_train_nnd_metapth = defaultdict(list)
                score_train_nnd_metapth = defaultdict(list)
                labels_train = labels[train_nid]
                for key, adj in adj_dict.items():
                    idx_train_nnd_metapth[key] = {}
                    score_train_nnd_metapth[key] = {}
                    weighted_score_B = torch.ones(adj.size(1), dtype=torch.float32).to(device)
                    for class_id, cnt in num_class_dict.items():
                        idx = train_nid[labels_train==class_id]
                        idx_train_nnd = []
                        score_train_nnd = []
                        idx_avaliable_temp = copy.deepcopy(list(idx))
                        adj = adj.to_dense()
                        t = perf_counter()
                        for count in range(cnt):
                            max_receptive_node, max_total_score = get_max_nnd_node_dense(idx_train_nnd,idx_avaliable_temp, adj, jaccard_score_buffer_sum[key[-1]][key], weighted_score_B)
                            idx_train_nnd.append(max_receptive_node)
                            score_train_nnd.append(max_total_score.item())
                            idx_avaliable_temp.remove(max_receptive_node)
                        print("time budget:", perf_counter()-t)
                        idx_train_nnd_metapth[key][class_id] = idx_train_nnd
                        score_train_nnd_metapth[key][class_id] = score_train_nnd
                torch.save(idx_train_nnd_metapth, f'/home/public/lyx/FreeHGC/hgb/idx_train_nnd_{args.dataset}_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')                    
                torch.save(score_train_nnd_metapth, f'/home/public/lyx/FreeHGC/hgb/score_train_nnd_{args.dataset}_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
                idx_train_nnd_metapth = torch.load(f'/home/public/lyx/FreeHGC/hgb/idx_train_nnd_{args.dataset}_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
                score_train_nnd_metapth = torch.load(f'/home/public/lyx/FreeHGC/hgb/score_train_nnd_{args.dataset}_{args.num_hops}_{args.reduction_rate}_{tgt_type}.pt')
                idx_selected = []
                score_train_idx_sum = defaultdict(list)
                for class_id, class_budget in num_class_dict.items():
                    score_train_idx_sum[class_id] = None
                    for key,value in idx_train_nnd_metapth.items():        
                        score_train_idx = dict(zip(value[class_id], score_train_nnd_metapth[key][class_id]))
                        score_train_idx_sum[class_id] = dict(Counter(score_train_idx_sum[class_id]) + Counter(score_train_idx))
                    ###### 对每个class_id的所有meta的结果score_train_idx_sum求topk
                    # score_train_idx = torch.argsort(torch.tensor(list(score_train_idx_sum[class_id].values())), descending=True)
                    _, score_train_idx = torch.topk(torch.tensor(list(score_train_idx_sum[class_id].values())), class_budget)
                    score_train_idx = torch.tensor(list(score_train_idx_sum[class_id].keys()))[score_train_idx]
                    idx_selected.append(score_train_idx)
                idx_selected = torch.cat(idx_selected, dim = 0).numpy()
                assert set(idx_selected) < set(train_nid)
                assert Counter(labels[idx_selected].numpy()) == num_class_dict
                ###condense target node type###
                ###############################

                # ### PPR: condense other node types ###
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
                #         ppr[key_A][key]= calc_ppr(adj_dict[key], key, device)  #[score_train_idx]待验证
                #         ppr_sum[key_A] += ppr[key_A][key][idx_selected] ##不同metapath直接相加，这里可以考虑优化
                #         # ppr[key_A][key] = None
                #     ppr_sum[key_A] = torch.sum(ppr_sum[key_A], dim = 0)
                # torch.save(ppr_sum, f'/home/public/lyx/FreeHGC/hgb/ppr_sum_{args.dataset}_{args.reduction_rate}_{tgt_type}.pt')                    
                # ppr_sum = torch.load(f'/home/public/lyx/FreeHGC/hgb/ppr_sum_{args.dataset}_{args.reduction_rate}_{tgt_type}.pt')
                candidate = {}
                candidate[tgt_type] = idx_selected

                # real_reduction_rate = len(idx_selected)/node_type_nodes[tgt_type]
                # for key, value in ppr_sum.items():
                #     reduce_nodes = int(real_reduction_rate * node_type_nodes[key])  #args.reduction_rate
                #     if reduce_nodes == 0:
                #         _, candidate[key] = torch.topk(ppr_sum[key], k = int(0.1 * node_type_nodes[key]))
                #     else:
                #         _, candidate[key] = torch.topk(ppr_sum[key], k = reduce_nodes)
                # x = 1
                # new_adjs = {}
                # for key, value in adjs.items():
                #     new_adjs[key] = adjs[key][candidate[key[0]]][:, candidate[key[1]]]


                for key, adj in adjs.items():
                    if key[0]==tgt_type and key[-1]!=tgt_type:
                        adjs[key] = adj[idx_selected]
                    if key[-1]==tgt_type and key[0]!=tgt_type:
                        adjs[key]  = adj[:,idx_selected]
                    if key[0]==tgt_type and key[-1]==tgt_type:
                        adjs[key]  = adj[idx_selected][:,idx_selected]
                new_adjs = adjs
                features_list_dict_cp[tgt_type] = features_list_dict_cp[tgt_type][idx_selected]

                ### PPR: condense other node types ###
            elif args.method == 'random':
                idx_selected = agent.select(features_list_dict)
                for key, adj in adjs.items():
                    if key[0]==tgt_type and key[-1]!=tgt_type:
                        adjs[key] = adj[idx_selected]
                    if key[-1]==tgt_type and key[0]!=tgt_type:
                        adjs[key]  = adj[:,idx_selected]
                    if key[0]==tgt_type and key[-1]==tgt_type:
                        adjs[key]  = adj[idx_selected][:,idx_selected]
                new_adjs = adjs
                features_list_dict_cp[tgt_type] = features_list_dict_cp[tgt_type][idx_selected]
            # for key, value in features_list_dict_cp.items():
            #     features_list_dict_cp[key] = features_list_dict_cp[key][candidate[key]]
                        
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
            keys = list(features_list_dict.keys())
            print(f'For tgt {tgt_type}, feature keys {keys}')
            keys_extra = list(extra_features_buffer.keys())
            print(f'For tgt {tgt_type}, extra feature keys {keys_extra}')
            for k in keys:
                feats[k] = features_list_dict.pop(k)
                feats_core[k] = coreset_features_list_dict.pop(k)
            for k in keys_extra:
                feats_extra[k] = extra_features_buffer.pop(k)
            data_size = {k: v.size(-1) for k, v in feats.items()}
            data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}

        elif args.dataset == "Freebase":
            prop_device = 'cuda:{}'.format(args.gpu)
            threshold_metalen = args.threshold
            #features_list_dict = hg_propagate_sparse_pyg_A(adjs, features_list_dict, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device='cpu')
            
            save_name = f'feat_seed{args.seed}_hop{args.num_hops}'
            if args.seed > 0 and os.path.exists(f'{save_name}_00_int64.npy'):
                # meta_adjs = torch.load(save_name)
                meta_adjs = {}
                for srcname in tqdm(dl.nodes['count'].keys()):
                    tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                    for k in tmp.keys:
                        assert k not in meta_adjs
                    meta_adjs.update(tmp.load_adjs(expand=True))
                    del tmp
            else:
                features_list_dict, extra_features_buffer = hg_propagate_sparse_pyg_freebase(adjs, threshold_metalen, tgt_type, args.num_hops, max_length, extra_metapath, prop_device, args.enhance, prop_feats=True, echo=True)
            feats = {}
            feats_extra = {}
            keys = list(features_list_dict.keys())
            print(f'For tgt {tgt_type}, feature keys {keys}')
            keys_extra = list(extra_features_buffer.keys())
            for k in keys:
                feats[k] = features_list_dict.pop(k)
            for k in keys_extra:
                feats_extra[k] = extra_features_buffer.pop(k)
            #feats = {k: v.clone() for k, v in features_list_dict.items()}
            feats['0'] = SparseTensor.eye(dl.nodes['count'][0])
            print(f'For tgt {tgt_type}, extra feature keys {keys_extra}')
            ##feats[k], _ = v.sample_adj(init2sort, -1, False) # faster, 50% time acceleration
            data_size = dict(dl.nodes['count'])
            data_size_extra = {k: v.size(-1) for k, v in feats_extra.items()}

        feats_r_ensemble.append(feats)
        coreset_feats_r_ensemble.append(feats_core)
        extra_feats_r_ensemble.append(feats_extra)

        # =======
        # labels propagate alongside the metapath
        # =======
        num_nodes = dl.nodes['count'][0]
        label_feats = {}
        label_feats_r_ensemble.append(label_feats)
    ################
    g = None
    ################

    if args.dataset == 'IMDB':
        labels = labels.float().to(device)
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
    model = GlobalMetaAggregator(feats.keys(), feats_extra.keys(), label_feats.keys(), data_size, data_size_extra, in_feats,
                                  r_len, tgt_type, args.input_dropout, args.dropout, args.num_hidden, 
                                  num_classes, args.ff_layer_2, args.att_drop, args.sum_metapath, args.bns)
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
            train(model, coreset_feats_r_ensemble, extra_feats_r_ensemble, label_feats_r_ensemble, labels_train, train_nid, loss_fcn, optimizer, args.batch_size, args.dataset)
            end = time.time()
            #start = time.time()
            #if epoch % args.eval_every == 0:
            with torch.no_grad():
                if args.micro:
                    train_micro, val_micro, test_micro, train_macro, val_macro, test_macro, loss_train, loss_val, loss_test = test(
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
                args.patience = 20
                if epoch - best_epoch > args.patience: break
    if args.micro:
        print(f'Best Epoch {best_epoch} at {checkpt_file.split("/")[-1]}\n\t with val loss {best_val_loss:.4f} and test loss {best_test_loss:.4f}')
        logging.info("macro: Best Val {:.4f}, Best Test {:.4f}".format(best_val_macro, best_test_macro))
        logging.info("micro: Best Val {:.4f}, Best Test {:.4f}".format(best_val_micro, best_test_micro))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HGAMLP-hgb")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--num-hops", type=int, default=2,
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
    parser.add_argument("--residual", action='store_true', default=False,
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
    parser.add_argument("--ACM-keep-F", type=bool, default=False)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--r", nargs='+', type=float, default=[0.0],
                        help="the seed used in the training")
    parser.add_argument("--sum-metapath", action='store_true', default=False)
    parser.add_argument("--bns", action='store_true', default=False)
    parser.add_argument("--mean", action='store_true', default=False)
    parser.add_argument("--transformer", action='store_true', default=False)
    parser.add_argument('--method', type=str, default='random', choices=['kcenter', 'herding', 'herding_class','random', 'FreeHGC'])
    parser.add_argument("--reduction-rate", type=float, default=0.1)
    #parser.add_argument("--infeat", type=int, default=512)
    args = parser.parse_args()

    print(args)
    main(args)