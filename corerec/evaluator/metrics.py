import numpy as np
from .base_metrics import *


class Hit(TopkMetric):
    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        super().__init__(topk_mat_with_pos_item_len, top_k, coreset_indices)

    def calculate_metric(self):
        topk_mat, _ = self.used_info()
        result = self.metric_info(topk_mat)
        metric_dict = self.topk_result('hit', result)
        return metric_dict

    def metric_info(self, topk_mat):
        result = np.cumsum(topk_mat, axis=1)
        return (result > 0).astype(int)

class Recall(TopkMetric):
    r"""Recall_ is a measure for computing the fraction of relevant items out of all relevant items.

    .. _recall: https://en.wikipedia.org/wiki/Precision_and_recall#Recall

    .. math::
       \mathrm {Recall@K} = \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}

    :math:`|R(u)|` represents the item count of :math:`R(u)`.
    """

    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        super().__init__(topk_mat_with_pos_item_len, top_k, coreset_indices)

    def calculate_metric(self):
        topk_mat, pos_len = self.used_info()
        result = self.metric_info(topk_mat, pos_len)
        metric_dict = self.topk_result('recall', result)
        return metric_dict

    def metric_info(self, topk_mat, pos_len):
        return np.cumsum(topk_mat, axis=1) / pos_len.reshape(-1, 1)

class Precision(TopkMetric):
    r"""Precision_ (also called positive predictive value) is a measure for computing the fraction of relevant items
    out of all the recommended items. We average the metric for each user :math:`u` get the final result.

    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} =  \frac{1}{|U|}\sum_{u \in U} \frac{|\hat{R}(u) \cap R(u)|}{|\hat {R}(u)|}

    :math:`|\hat R(u)|` represents the item count of :math:`\hat R(u)`.
    """

    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        super().__init__(topk_mat_with_pos_item_len, top_k, coreset_indices)

    def calculate_metric(self):
        topk_mat, _ = self.used_info()
        result = self.metric_info(topk_mat)
        metric_dict = self.topk_result('precision', result)
        return metric_dict

    def metric_info(self, topk_mat):
        return topk_mat.cumsum(axis=1) / np.arange(1, topk_mat.shape[1] + 1)

class MRR(TopkMetric):
    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        super().__init__(topk_mat_with_pos_item_len, top_k, coreset_indices)

    def calculate_metric(self):
        topk_mat, _ = self.used_info()
        result = self.metric_info(topk_mat)
        metric_dict = self.topk_result('mrr', result)
        return metric_dict

    def metric_info(self, topk_mat):
        idx_of_first_hit = topk_mat.argmax(axis=1)
        result = np.zeros_like(topk_mat, dtype=np.float)
        for row, idx in enumerate(idx_of_first_hit):
            if topk_mat[row, idx] > 0:
                result[row, idx:] = 1 / (idx + 1)
            else:
                result[row, idx:] = 0
        return result


class MAP(TopkMetric):
    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        super().__init__(topk_mat_with_pos_item_len, top_k, coreset_indices)

    def calculate_metric(self):
        topk_mat, n_ground_truth_items = self.used_info()
        result = self.metric_info(topk_mat, n_ground_truth_items)
        metric_dict = self.topk_result('map', result)
        return metric_dict

    def metric_info(self, topk_mat, n_ground_truth_items):
        pre = topk_mat.cumsum(axis=1) / np.arange(1, topk_mat.shape[1] + 1)
        sum_pre = np.cumsum(pre * topk_mat.astype(np.float), axis=1)
        len_rank = np.full_like(n_ground_truth_items, topk_mat.shape[1])
        actual_len = np.where(n_ground_truth_items > len_rank, len_rank, n_ground_truth_items)
        result = np.zeros_like(topk_mat, dtype=np.float)
        for row, lens in enumerate(actual_len):
            ranges = np.arange(1, topk_mat.shape[1] + 1)
            ranges[lens:] = ranges[lens - 1]
            result[row] = sum_pre[row] / ranges
        return result


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})
    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        super().__init__(topk_mat_with_pos_item_len, top_k, coreset_indices)

    def calculate_metric(self):
        pos_index, pos_len = self.used_info()
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result('ndcg', result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result