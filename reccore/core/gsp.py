import numpy as np
import scipy as sp
import time
import torch
from .abstractstrategy import AbstractStrategy
import similaripy as sim



class GSP(AbstractStrategy):

    def __init__(self, dataset, model, args):
        """
        Constructor method
        """
        super(GSP, self).__init__(dataset, model, args)
        self.initial_signal = 1
        self.decay_factor = 0.9

    def _calculate_user_similarity(self, embeddings):
        csr_representation = sp.sparse.csr_matrix(embeddings)
        sparse_sim_mat = sim.cosine(csr_representation, k=csr_representation.shape[0], verbose=False)
        self.user_sim_mat = sparse_sim_mat.toarray()

    def _influence_propagation(self):
        pass

    def _graph_fourier_transform(self):
        pass

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

