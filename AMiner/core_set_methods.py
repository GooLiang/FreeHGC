import torch
import numpy as np
from collections import Counter, defaultdict

class Base:
    def __init__(self, labels, train_nid, args, device='cuda', **kwargs):
        self.labels = labels
        self.train_nid = train_nid
        self.device = device
        counter = Counter(labels[train_nid].numpy()) ###每个类的标签数量并不一致
        num_class_dict = {}
        n = len(labels[train_nid])
        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(num * args.reduction_rate) #int(n * args.reduction_rate) - sum_
            else:
                num_class_dict[c] = max(int(num * args.reduction_rate), 1)
                sum_ += num_class_dict[c]
        self.num_class_dict = num_class_dict
    def select(self):
        return

class KCenter(Base):

    def __init__(self, labels, train_nid, args, device, **kwargs):
        super(KCenter, self).__init__(labels, train_nid, args, device, **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # kcenter # class by class
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.train_nid))
        else:
            idx_train = self.train_nid
        labels_train = self.labels[self.train_nid]
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            # dis = distance(feature, mean)[:,0]
            dis = torch.cdist(feature, mean)[:,0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for i in range(cnt-1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)

            idx_selected.append(idx[idx_centers])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)
    def select_top(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.train_nid))
        else:
            idx_train = self.train_nid
        labels_train = self.labels[self.train_nid]
        idx_selected = []

        # herding # class by class
        dis_dict = {}
        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            dis = torch.cdist(feature, mean)[:,0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            dis_dict[class_id] = {}
            for i in range(cnt):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max_index = torch.argmax(dis_min).item()
                id_max_value = torch.max(dis_min).item()
                dis_dict[class_id][idx[id_max_index]] = id_max_value
                idx_centers.append(id_max_index)
            # idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return dis_dict

    def select_other_types_top(self, feature, cnt):
        dis_dict = {}
        idx = np.arange(feature.shape[0]).tolist()
        mean = torch.mean(feature, dim=0, keepdim=True)
        dis = torch.cdist(feature, mean)[:,0]
        rank = torch.argsort(dis)
        idx_centers = rank[:1].tolist()
        for i in range(cnt):
            feature_centers = feature[idx_centers]
            dis_center = torch.cdist(feature, feature_centers)
            dis_min, _ = torch.min(dis_center, dim=-1)
            id_max_index = torch.argmax(dis_min).item()
            id_max_value = torch.max(dis_min).item()
            dis_dict[idx[id_max_index]] = id_max_value
            idx_centers.append(id_max_index)
            # idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return dis_dict

class Herding(Base):

    def __init__(self, labels, train_nid, args, device, **kwargs):
        super(Herding, self).__init__(labels, train_nid, args, device, **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.train_nid))
        else:
            idx_train = self.train_nid
        labels_train = self.labels[self.train_nid]
        idx_selected = []

        # herding # class by class
        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            features = embeds[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                det = mean*(i+1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                selected.append(idx_left[id_min])
                del idx_left[id_min]
            idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)
    
    def select_top(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.train_nid))
        else:
            idx_train = self.train_nid
        labels_train = self.labels[self.train_nid]
        idx_selected = []

        # herding # class by class
        dis_dict = {}
        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            features = embeds[idx]
            features = features.to_dense()
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()
            dis_dict[class_id] = {}
            for i in range(cnt):
                det = mean*(i+1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                id_min = torch.argmin(dis)
                id_min_value = torch.min(dis)
                selected.append(idx_left[id_min])
                dis_dict[class_id][idx_left[id_min]] = id_min_value
                del idx_left[id_min]
            # idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return dis_dict

    def select_other_types_top(self, features, cnt):
        dis_dict = {}
        features = features.to_dense()
        mean = torch.mean(features, dim=0, keepdim=True)
        selected = []
        idx_left = np.arange(features.shape[0]).tolist()
        for i in range(cnt):
            det = mean*(i+1) - torch.sum(features[selected], dim=0)
            dis = torch.cdist(det, features[idx_left])
            id_min = torch.argmin(dis)
            id_min_value = torch.min(dis)
            selected.append(idx_left[id_min])
            dis_dict[idx_left[id_min]] = id_min_value
            del idx_left[id_min]
            # idx_selected.append(idx[selected])
        # return np.array(idx_selected).reshape(-1)
        return dis_dict
    
    def select_other_types(self, features, cnt):
        # herding # class by class
        mean = torch.mean(features, dim=0, keepdim=True)
        selected = []
        idx_left = np.arange(features.shape[0]).tolist()
        for i in range(cnt):
            det = mean*(i+1) - torch.sum(features[selected], dim=0)
            dis = torch.cdist(det, features[idx_left])
            id_min = torch.argmin(dis)
            selected.append(idx_left[id_min])
            del idx_left[id_min]
        return selected

class Random(Base): ###对每种标签选取自定义个数的idx_train

    def __init__(self, labels, train_nid, args, device, **kwargs):
        super(Random, self).__init__(labels, train_nid, args, device, **kwargs)

    def select(self, inductive=False):  ###random不使用embeds,只对训练节点做随机选择
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.train_nid))
        else:
            idx_train = self.train_nid

        labels_train = self.labels[self.train_nid]
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = idx_train[labels_train==class_id]
            selected = np.random.permutation(idx) ###打乱idx
            idx_selected.append(selected[:cnt])

        # return np.array(idx_selected).reshape(-1)
        return np.hstack(idx_selected)

    def select_other_types(self, idx, cnt):  ###random不适用embeds,只对训练节点做随机选择

        selected = np.random.permutation(idx) ###打乱idx
        idx_selected = (selected[:cnt])

        # return np.array(idx_selected).reshape(-1)
        return idx_selected
class Herding_class(Base):

    def __init__(self, labels, train_nid, args, device, **kwargs):
        super(Herding_class, self).__init__(labels, train_nid, args, device, **kwargs)

    def select(self, class_id, cnt, embeds, key_counter, inductive=False):
        num_class_dict = self.num_class_dict
        if inductive:
            idx_train = np.arange(len(self.train_nid))
        else:
            idx_train = self.train_nid
        labels_train = self.labels[self.train_nid]
        idx_selected = []
        class_selected = {}
        idx = idx_train[labels_train==class_id]
        # herding # class by class
        for key in embeds:
            features = embeds[key][idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()
            class_selected[key] = {}
            for i in range(cnt):
                det = mean*(i+1) - torch.sum(features[selected], dim=0)
                dis = torch.cdist(det, features[idx_left])
                class_selected[key][i] = dis
            #     id_min = torch.argmin(dis)
            #     selected.append(idx_left[id_min])
            #     del idx_left[id_min]
            # idx_selected.append(idx[selected])
        meta_sum_selected = {key:{i:0 for i in range(cnt)} for key, value in key_counter.items()}
        for i in range(cnt):
            for key, value in key_counter.items():
                for class_key, class_value in class_selected.items():
                    if class_key in value:
                        meta_sum_selected[key][i] += class_selected[class_key][i]
        idx_selected = []
        for key in meta_sum_selected.keys():
            selected = []
            for i in range(cnt):
                id_min = torch.argmin(meta_sum_selected[key][i])
                selected.append(idx_left[id_min])
            idx_selected.append(idx[selected])
        return meta_sum_selected
        a = list(set(idx_selected[0]).union(*idx_selected[1:]))