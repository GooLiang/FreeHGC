# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

###############################################################################
# Training and testing for one epoch
###############################################################################

def train(model, feats, extra_feats, label_feats, labels, train_nid, loss_fcn, optimizer, batch_size, dataset, history=None, ):  #len(feats)=6, 6是hop数, 每个hop包含所有关系子类型的特征, for x in feats, shape = [736389, 8, 256]
    model.train()
    device = labels.device
    # import sys
    # sys.getsizeof(feats)
    #batch_size = 80
    dataloader = torch.utils.data.DataLoader(
        train_nid, batch_size=batch_size, shuffle=True, drop_last=False)

    for batch in dataloader:
        batch_feats_ensemble = []
        batch_label_feats_ensemble = []
        batch_extra_feats_ensemble = []
        for i in range(len(feats)):
            #batch_feats = [x[batch].to(device) for x in feats[i]]  #存每个hop的batch个节点的特征，batch中存储的并不是连续id，是离散的随机id(e.g., tensor([ 16195,  67215, 213209,  ..., 196225, 455400, 467634])).
            batch_feats = {k: x[batch].to(device) for k, x in feats[i].items()}
            batch_extra_feats = {k: x[batch].to(device) for k, x in extra_feats[i].items()}
            batch_laebl_feats = {k: x[batch].to(device) for k, x in label_feats[i].items()}
            batch_feats_ensemble.append(batch_feats)
            batch_extra_feats_ensemble.append(batch_extra_feats)
            batch_label_feats_ensemble.append(batch_laebl_feats)

        output = model(batch_feats_ensemble, batch_extra_feats_ensemble, batch_label_feats_ensemble)
        if dataset == 'IMDB':
            output = torch.sigmoid(output)
        loss = loss_fcn(output, labels[batch])
        optimizer.zero_grad()   #梯度清零
        loss.backward() #retain_graph=True
        optimizer.step()
        
        if dataset != 'IMDB':
            preds = output.detach().cpu().numpy().argmax(axis=1)
        else:
            preds = (output.cpu().numpy()>0.5).astype(int)   ###>0.5?when
        train_micro = f1_score(preds, labels[train_nid].cpu(), average='micro')
        train_macro = f1_score(preds, labels[train_nid].cpu(), average='macro')
    return train_micro, train_macro

def test(model, feats, extra_feats, label_feats, labels, train_nid, val_nid, test_nid, loss_fcn, evaluator, batch_size, micro, dataset):
    with torch.no_grad():
        model.eval()
        num_nodes = labels.shape[0]
        device = labels.device
        dataloader = torch.utils.data.DataLoader(
            torch.arange(num_nodes), batch_size=batch_size,
            shuffle=False, drop_last=False)
        scores = []
        pred_buffer = []
        batch_buffer = []
        for batch in dataloader:
            batch_feats_ensemble = []
            batch_label_feats_ensemble = []
            batch_extra_feats_ensemble = []
            for i in range(len(feats)):
                batch_feats = {k: x[batch].to(device) for k, x in feats[i].items()}
                batch_extra_feats = {k: x[batch].to(device) for k, x in extra_feats[i].items()}
                batch_label_feats = {k: x[batch].to(device) for k, x in label_feats[i].items()}
                batch_feats_ensemble.append(batch_feats)
                batch_extra_feats_ensemble.append(batch_extra_feats)
                batch_label_feats_ensemble.append(batch_label_feats)
            pred = model(batch_feats_ensemble, batch_extra_feats_ensemble, batch_label_feats_ensemble)  ##soft label for batch\
            pred_buffer.append(pred)
            #print(pred)
        preds = torch.cat(pred_buffer, dim=0)
        if dataset == 'IMDB':
            preds = torch.sigmoid(preds)
        loss_train = loss_fcn(preds[train_nid], labels[train_nid]).item()
        loss_val = loss_fcn(preds[val_nid], labels[val_nid]).item()
        loss_test = loss_fcn(preds[test_nid], labels[test_nid]).item()

        if micro:
            from sklearn.metrics import f1_score
            if dataset != 'IMDB':
                preds = preds.cpu().numpy().argmax(axis=1)
            else:
                preds = (preds.cpu().numpy()>0.5).astype(int)   ###>0.5?when
            val_micro = f1_score(preds[val_nid], labels[val_nid].cpu(), average='micro')
            test_micro = f1_score(preds[test_nid], labels[test_nid].cpu(), average='micro')
            val_macro = f1_score(preds[val_nid], labels[val_nid].cpu(), average='macro')
            test_macro = f1_score(preds[test_nid], labels[test_nid].cpu(), average='macro')
        return val_micro, test_micro, val_macro, test_macro, loss_train, loss_val, loss_test


def train_SeHGNN(model, feats, extra_feats, label_feats, labels, train_nid, loss_fcn, optimizer, batch_size, dataset, history=None, ):  #len(feats)=6, 6是hop数, 每个hop包含所有关系子类型的特征, for x in feats, shape = [736389, 8, 256]
    model.train()
    device = labels.device
    # import sys
    # sys.getsizeof(feats)
    #batch_size = 80
    dataloader = torch.utils.data.DataLoader(
        train_nid, batch_size=batch_size, shuffle=True, drop_last=False)

    for batch in dataloader:
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}

        output = model(None, batch_feats, batch_labels_feats, None)
        if dataset == 'IMDB':
            output = torch.sigmoid(output)
        loss = loss_fcn(output, labels[batch])
        optimizer.zero_grad()   #梯度清零
        loss.backward() #retain_graph=True
        optimizer.step()
        if dataset != 'IMDB':
            preds = output.detach().cpu().numpy().argmax(axis=1)
        else:
            preds = (output.detach().cpu().numpy()>0.5).astype(int)   ###>0.5?when
        train_micro = f1_score(preds, labels[batch].cpu(), average='micro')  ###train_nid
        train_macro = f1_score(preds, labels[batch].cpu(), average='macro')
    return train_micro, train_macro

def test_SeHGNN(model, feats, extra_feats, label_feats, labels, train_nid, val_nid, test_nid, loss_fcn, evaluator, batch_size, micro, dataset):
    with torch.no_grad():
        model.eval()
        num_nodes = labels.shape[0]
        device = labels.device
        dataloader = torch.utils.data.DataLoader(
            torch.arange(num_nodes), batch_size=batch_size,
            shuffle=False, drop_last=False)
        scores = []
        pred_buffer = []
        for batch in dataloader:
            if isinstance(feats, list):
                batch_feats = [x[batch].to(device) for x in feats]
            elif isinstance(feats, dict):
                batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}

            pred = model(None, batch_feats, batch_labels_feats, None)
            pred_buffer.append(pred)
            #print(pred)
        preds = torch.cat(pred_buffer, dim=0)
        if dataset == 'IMDB':
            preds = torch.sigmoid(preds)
        loss_train = 0 ### 这里的train_nid是压缩后的
        loss_val = loss_fcn(preds[val_nid], labels[val_nid]).item()
        loss_test = loss_fcn(preds[test_nid], labels[test_nid]).item()

        if micro:
            if dataset != 'IMDB':
                preds = preds.cpu().numpy().argmax(axis=1)
            else:
                preds = (preds.cpu().numpy()>0.5).astype(int)   ###>0.5?when
            val_micro = f1_score(preds[val_nid], labels[val_nid].cpu(), average='micro')
            test_micro = f1_score(preds[test_nid], labels[test_nid].cpu(), average='micro')
            val_macro = f1_score(preds[val_nid], labels[val_nid].cpu(), average='macro')
            test_macro = f1_score(preds[test_nid], labels[test_nid].cpu(), average='macro')
        return val_micro, test_micro, val_macro, test_macro, loss_train, loss_val, loss_test


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid, show_test=True, loss_type='ce'):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == 'bce':
        na, nb, nc = len(train_nid) * v.size(1), len(val_nid) * v.size(1), len(test_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                         F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                         F.nll_loss(vv[test_nid], init_labels[test_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1-1e-6)
            la, lb, lc = F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float()), \
                         F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float()), \
                         F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float())
        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        if show_test:
            print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)
        else:
            print(k, ra, rb, la, lb, (ra/rb-1)*100, (1-la/lb)*100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)
###############################################################################
# Evaluator for different datasets
###############################################################################

def batched_acc(pred, labels):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)

def batched_acc_multi(pred, labels):
    # testing accuracy for multi label multi-class prediction
    return ((pred > 0.).int() == labels,)

def get_evaluator(dataset):
    dataset = dataset.lower()
    if dataset == 'imdb':
        return batched_acc_multi
    else:
        return batched_acc


def compute_mean(metrics, nid):
    num_nodes = len(nid)
    return [m[nid].float().sum().item() / num_nodes for  m in metrics]