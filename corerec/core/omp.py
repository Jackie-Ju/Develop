import math
import time
import torch
import numpy as np
from .abstractstrategy import AbstractStrategy
from corerec.core.methods_utils.omp_solvers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, \
    OrthogonalMP_REG_Parallel_V1
from torch.utils.data import Subset, DataLoader


class OMP(AbstractStrategy):

    def __init__(self, dataset, model, args):
        """
        Constructor method
        """
        super(OMP, self).__init__(dataset, model, args)
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

    def after_train(self, model_params):
        self.update_model(model_params)

    def finish_run(self):
        pass

    def select(self):
        omp_start_time = time.time()
        idxs = []
        gammas = []
        budget = 0
        if not self.propensity:
            source_embs = self.construct_matrix()
            budget = round(source_embs.shape[0] * self.coreset_size)
            target_signal = torch.sum(source_embs, dim=0)
            idxs_temp, gammas_temp = self.ompwrapper(X=torch.transpose(source_embs, 0, 1), Y=target_signal, bud=budget)
            idxs.extend(idxs_temp)
            gammas.extend(gammas_temp)

        diff = budget - len(idxs)
        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])

        omp_end_time = time.time()
        self.logger.info("OMP algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs
