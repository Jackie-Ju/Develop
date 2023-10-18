import torch
import numpy as np


class TopkMetric(object):
    r"""This class gathers the performance of an RS model.

    Args:
        topk_mat_with_pos_item_len: A matrix with shape[n_user, topk+1]. The values in the first top K columns
        are binary. 1 represents that a recommended item appears in the user's ground truth and its position represents
        its ranking. 0 represents false-positive recommended items. The k+1 column indicates how long the user's
        ground truth list is. For example, if user 0 has 4 items in his ground truth list, the value of the last column
        will be 4. topk_mat_with_pos_item_len[0, topk] = 4.

        top_k: Cut off.

        coreset_indices: The coreset user ids. The performance of both coreset users and non-coreset users will be
        gathered and returned if provided. Otherwise, only the performance for all the users will be returned.

    """
    def __init__(self, topk_mat_with_pos_item_len, top_k, coreset_indices):
        self.topk_mat_with_pos_item_len = topk_mat_with_pos_item_len
        self.top_k = top_k
        self.coreset_indices = coreset_indices

    def used_info(self):
        topk_mat, n_ground_truth_itmes = torch.split(self.topk_mat_with_pos_item_len, [self.top_k, 1], dim=1)
        return topk_mat.to(torch.bool).numpy(), n_ground_truth_itmes.squeeze(-1).numpy()

    def topk_result(self, metric, value_mat):
        r"""Gather and return the performance metrics. Metrics from 1 to top k will be gathered and returned.
        For example, if top k is 5 and metric is nDCG, nDCG@1, nDCG@2, nDCG@3, nDCG@4 and nDCG@5 will be returned.

        Args:
            metric: The desired metric.
            value_mat: A matrix with shape [n_user, topk]. Each row stores the model performance on a specific user from
            top 1 to top k. Every metric has its own corresponding value_mat.

        Returns:
            metric_dict: A dictionary that contains all the metrics with cut-off up to top k.
        """
        metric_dict = {"full_user": {}}
        avg_result = value_mat.mean(axis=0)
        for k in range(1, self.top_k+1):
            key = '{}@{}'.format(metric, k)
            metric_dict["full_user"][key] = round(avg_result[k - 1], 4)

        # Gather the performance for both coreser users and non-coreset users
        # if coreset_indices is provided.
        if self.coreset_indices is not None:
            full_sum_result = value_mat.sum(axis=0)
            full_sum_result[full_sum_result < 1] = 1
            metric_dict["coreset_performance"] = {}
            metric_dict["coreset_contribution"] = {}

            metric_dict["rest_user_performance"] = {}
            metric_dict["rest_user_contribution"] = {}

            coreuser_mask = np.full(value_mat.shape[0], True)
            coreuser_mask.put(self.coreset_indices, False)
            coreset_value_mat = value_mat[self.coreset_indices]
            rest_user_value_mat = value_mat[coreuser_mask, :]

            coreset_avg_result = coreset_value_mat.mean(axis=0)
            rest_user_avg_result = rest_user_value_mat.mean(axis=0)
            coreset_contribution_result = coreset_value_mat.sum(axis=0) / full_sum_result
            rest_user_contribution_result = rest_user_value_mat.sum(axis=0) / full_sum_result

            for k in range(1, self.top_k + 1):
                key = '{}@{}'.format(metric, k)
                metric_dict["coreset_performance"][key] = round(coreset_avg_result[k - 1], 4)
                metric_dict["coreset_contribution"][key] = round(coreset_contribution_result[k - 1], 4)
                metric_dict["rest_user_performance"][key] = round(rest_user_avg_result[k - 1], 4)
                metric_dict["rest_user_contribution"][key] = round(rest_user_contribution_result[k - 1], 4)

        return metric_dict
