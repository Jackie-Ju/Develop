import numpy as np
import scipy as sp
import time
import torch
from sklearn.cluster import KMeans
from .abstractstrategy import AbstractStrategy
import similaripy as sim



class KCluster(AbstractStrategy):

    def __init__(self, dataset, model, args):
        """
        Constructor method
        """
        super(KCluster, self).__init__(dataset, model, args)

    def k_center(self, source_embs, budget):
        cluster_source = source_embs.cpu().numpy()
        n_cluster = int(source_embs.shape[0] * 0.01)
        if n_cluster > budget:
            n_cluster = budget
        clusterer = KMeans(n_clusters=n_cluster, init="k-means++", random_state=0)
        cluster_labels = clusterer.fit_predict(cluster_source)

        centres = clusterer.cluster_centers_
        ele_per_cluster = round(budget / n_cluster)
        selected_elements_per_cluster = []
        weights = []
        for i in range(n_cluster):
            idx_per_cluster = np.where(cluster_labels == i)[0]
            centre_point = centres[i]

            members = source_embs[idx_per_cluster].cpu().numpy()
            dist = sp.linalg.norm((members - centre_point), axis=1, keepdims=True)
            dist = dist.reshape(-1)
            # sort dist in ascending order
            sorted_idx = np.argsort(dist, axis=0)
            # get real id
            sorted_idx = idx_per_cluster[sorted_idx]
            selected_elements_per_cluster.extend(sorted_idx[:ele_per_cluster])
            weights.extend(np.full(ele_per_cluster, len(sorted_idx)))

        return selected_elements_per_cluster, weights

    def after_train(self, model_params):
        self.update_model(model_params)

    def finish_run(self):
        pass

    def select(self):
        start_time = time.time()
        user_idx_list = []
        gammas = []
        budget = 0
        if not self.propensity:
            source_embs = self.construct_matrix()
            budget = round(source_embs.shape[0] * self.coreset_size)
            user_idx_list, gammas = self.k_center(source_embs.detach(), budget)

        diff = budget - len(user_idx_list)
        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(user_idx_list))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            user_idx_list.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])

        end_time = time.time()
        self.logger.info("KCluster strategy data selection time is: %.4f", end_time-start_time)
        return user_idx_list
        # return total_greedy_list, torch.ones(len(gammas))

