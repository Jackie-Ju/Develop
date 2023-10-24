import numpy as np
# import cupy as cp
# from cuml.cluster import KMeans
import time
import torch
from sklearn.cluster import KMeans
from .abstractstrategy import AbstractStrategy
from corerec.core.methods_utils.omp_solvers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
import similaripy as sim



class OMPKMeans(AbstractStrategy):

    def __init__(self, dataset, model, args):
        """
        Constructor method
        """
        super(OMPKMeans, self).__init__(dataset, model, args)
        self.lam = args.dss_args.lam
        self.eps = args.dss_args.eps
        self.v1 = args.dss_args.v1
        self.positive = args.dss_args.positive

    def ompwrapper(self, X, Y, bud):
        if self.device == "cpu":
            reg = OrthogonalMP_REG(X.numpy(), Y.numpy(), nnz=bud, positive=self.positive, lam=self.lam, tol=self.eps)
            ind = np.nonzero(reg)[0]
        else:
            if self.v1:
                reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                                 positive=self.positive, lam=self.lam,
                                                 tol=self.eps, device=self.device)
            else:
                reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                                positive=self.positive, lam=self.lam,
                                                tol=self.eps, device=self.device)
            ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def k_center(self, source_embs, budget):
        cluster_source = source_embs.cpu().numpy()
        n_cluster = int(source_embs.shape[0] * 0.01)
        clusterer = KMeans(n_clusters=n_cluster, init="k-means++", random_state=0)
        cluster_labels = clusterer.fit_predict(cluster_source)

        ele_per_cluster = int(budget / n_cluster)
        _, member_counts = np.unique(cluster_labels, return_counts=True)
        qualified_clusters = np.where(member_counts >= ele_per_cluster)[0]
        total_members = member_counts[qualified_clusters].sum()

        # todo: different number of selected items for differernt clusters
        '''
        unqualified_clusters = np.setdiff1d(_, qualified_clusters)
        total_qualifide_members = member_counts[qualified_clusters].sum()
        total_unqualified_member = member_counts.sum() - total_qualifide_members
        number_users_to_be_selected = budget - total_unqualified_member
        '''

        selected_elements_per_cluster = []
        weights = []
        for i in range(n_cluster):
            if i not in qualified_clusters:
                continue
            idx_per_cluster = np.where(cluster_labels == i)[0]
            ele_per_cluster = round(budget * len(idx_per_cluster) / total_members)
            if ele_per_cluster < 1:
                # ele_per_cluster = 1
                continue

            # self.eps = self.eps * len(idx_per_cluster) / total_members
            members = source_embs[idx_per_cluster]
            target_signal = torch.sum(members, dim=0)
            idxs_temp, gammas_temp = self.ompwrapper(X=torch.transpose(members, 0, 1),
                                                     Y=target_signal,
                                                     bud=ele_per_cluster)
            # get real id
            selected_idx = idx_per_cluster[idxs_temp]
            selected_elements_per_cluster.extend(selected_idx)

            # generalise weights to the whole grad matrix
            # gammas_temp = np.array(gammas_temp) * len(idx_per_cluster) * (len(idx_per_cluster)/self.N_trn)

            weights.extend(gammas_temp)
            # weights.extend(np.full(ele_per_cluster, len(selected_idx)))
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
        self.logger.info("OMPKMeans strategy data selection time is: %.4f", end_time-start_time)
        return user_idx_list
        # return total_greedy_list, torch.ones(len(gammas))

