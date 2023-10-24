import numpy as np
import scipy as sp
import time
import torch
from tqdm import tqdm
from .abstractstrategy import AbstractStrategy
import similaripy as sim


class GSP(AbstractStrategy):

    def __init__(self, dataset, model, args):

        super(GSP, self).__init__(dataset, model, args)
        self.decay_factor = args.dss_args.decay_factor
        self.hops = args.dss_args.hops
        self.low_freq_k = args.dss_args.low_freq_k

    def _calculate_user_similarity(self, embeddings):
        self.logger.info("Calculating user similarity...")
        csr_representation = sp.sparse.csr_matrix(embeddings)
        sparse_sim_mat = sim.cosine(csr_representation, k=csr_representation.shape[0], verbose=False)
        self.user_sim_mat = sparse_sim_mat.toarray()
        np.fill_diagonal(self.user_sim_mat, 0)

    def _calculate_laplacian(self):
        self.logger.info("Calculating user-user graph laplacian...")
        D = np.diag(np.sum(self.user_sim_mat, axis=1))
        L_weighted = D - self.user_sim_mat
        return L_weighted

    def _influence_propagation(self):
        self.logger.info("Graph influence propagation...")
        initial_influence = np.eye(self.user_sim_mat.shape[0])
        all_influences = []
        propagated_influence_matrix = initial_influence
        for hop in tqdm(range(self.hops), desc="Hop"):
            propagated_influence_matrix = self.decay_factor * self.user_sim_mat @ propagated_influence_matrix
            np.fill_diagonal(propagated_influence_matrix, 0)
            if hop > 0:
                propagated_influence_matrix /= (self.user_sim_mat.shape[0]-2)
            all_influences.append(propagated_influence_matrix)
        self.propagated_influences = np.sum(all_influences, axis=0)

    def _graph_fourier_transform(self):
        self.logger.info("Conducting Graph Fourier Transform on the graph influence matrix...")
        L_weighted = self._calculate_laplacian()
        eigen_vals, eigen_vecs = np.linalg.eigh(L_weighted)
        GFT_frequency = eigen_vecs.T @ self.propagated_influences.T
        return GFT_frequency

    def after_train(self, model_params):
        self.update_model(model_params)

    def finish_run(self):
        pass

    def select(self):
        self.logger.info("GSP coreset selection begins...")
        start_time = time.time()
        user_idx_list = []
        if not self.propensity:
            source_embs = self.construct_matrix()
            budget = round(source_embs.shape[0] * self.coreset_size)
            self._calculate_user_similarity(source_embs)
            self._influence_propagation()
            GFT_frequency = self._graph_fourier_transform()
            low_frequency_content = GFT_frequency[:self.low_freq_k]
            low_sum = np.sum(low_frequency_content, axis=0)
            indices = np.argsort(low_sum)[::-1]
            user_idx_list.extend(indices[:budget])

        end_time = time.time()
        self.logger.info("GSP strategy data selection time is: %.4f", end_time-start_time)
        return user_idx_list

