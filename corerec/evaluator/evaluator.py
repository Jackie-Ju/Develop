import numpy as np
import torch
import inspect
import sys
from corerec.evaluator import metrics
from corerec.utils.utils import update_dict
from time import time


class AbstractEvaluator(object):
    pass


class Collector(object):
    def __init__(self, n_items, top_k):
        self.top_k = top_k
        self.n_items = n_items
        self.topK_mat = None
        self.topk_mat_with_pos_item_len = None

    def eval_batch_collector(self, positive_u, items, scores, unique_users):
        unique_len = len(unique_users)
        positive_mat = torch.zeros((unique_len, self.n_items), dtype=torch.int)
        positive_mat[positive_u, items] = 1
        _, topk_idx = torch.topk(scores, self.top_k, dim=1)

        if self.topK_mat is None:
            self.topK_mat = topk_idx
        else:
            self.topK_mat = torch.cat((self.topK_mat, topk_idx), dim=0)
        positive_len_list = positive_mat.sum(dim=1, keepdim=True)
        positive_idx_mat = torch.gather(positive_mat, dim=1, index=topk_idx)
        positive_idx_mat_with_len = torch.cat((positive_idx_mat, positive_len_list), dim=1)
        if self.topk_mat_with_pos_item_len is None:
            self.topk_mat_with_pos_item_len = positive_idx_mat_with_len
        else:
            self.topk_mat_with_pos_item_len = torch.cat((self.topk_mat_with_pos_item_len, positive_idx_mat_with_len), dim=0)

    def get_topk_mat_with_len(self):
        return self.topk_mat_with_pos_item_len

    def get_topK_mat(self):

        return self.topK_mat

    def get_topk(self):

        return self.top_k


class Evaluator(object):
    def __init__(self, topk_mat_with_pos_item_len, metrics, top_k, coreset_indices):
        self.topk_mat_with_pos_item_len = topk_mat_with_pos_item_len
        self.metrics = metrics
        self.metric_module_name = 'corerec.evaluator.metrics'
        self.top_k = top_k
        self.coreset_indices = coreset_indices

    def evaluate(self):
        result_dict = {}
        metric_dict = self._metrics()
        for metric_name in self.metrics:
            metric = metric_dict[metric_name.lower()](self.topk_mat_with_pos_item_len, self.top_k, self.coreset_indices)
            metric_val = metric.calculate_metric()
            result_dict = update_dict(result_dict, metric_val)
        return result_dict

    def _metrics(self):
        metric_dict = {}
        metric_class = inspect.getmembers(
            sys.modules[self.metric_module_name],
            lambda x: inspect.isclass(x) and x.__module__ == self.metric_module_name
        )
        for name, metric_cls in metric_class:
            name = name.lower()
            metric_dict[name] = metric_cls
        return metric_dict


class Recommender():
    def __init__(self, collector):
        self.topK_mat = collector.get_topK_mat()
        self.top_k = collector.get_topk()

    def recommend(self):

        return self.topK_mat
    '''
    def cheat_score_mat(self, item_id=None):
        if item_id is not None:
            self.score_mat[:,item_id] = -np.inf
    '''