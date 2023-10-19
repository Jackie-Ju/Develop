import apricot
import numpy as np
import time
import torch
from .abstractstrategy import AbstractStrategy


class FacilityLocation(AbstractStrategy):

    def __init__(self, dataset, model, args):
        """
        Constructer method
        """
        super(FacilityLocation, self).__init__(dataset, model, args)
        # self.logger = logger
        self.optimizer = args.dss_args.optimizer

    def distance(self, x, y, exp=2):
        """
        Compute the distance.

        Parameters
        ----------
        x: Tensor
            First input tensor
        y: Tensor
            Second input tensor
        exp: float, optional
            The exponent value (default: 2)

        Returns
        ----------
        dist: Tensor
            Output tensor
        """

        # n = x.size(0)
        # m = y.size(0)
        # d = x.size(1)
        # x = x.unsqueeze(1).expand(n, m, d)
        # y = y.unsqueeze(0).expand(n, m, d)
        # dist = torch.pow(x - y, exp).sum(2)
        #
        dist = torch.cdist(x, y, 2)
        return dist

    def compute_score(self, source_embs):
        """
        Compute the score of the indices.

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
            :param source_embs:
        """
        self.dist_mat = torch.zeros([source_embs.shape[0], source_embs.shape[0]], dtype=torch.float32)

        self.dist_mat = self.distance(source_embs, source_embs)

        self.const = torch.max(self.dist_mat)
        self.dist_mat = self.const - self.dist_mat

    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.

        Parameters
        ----------
        idxs: list
            The indices

        Returns
        ----------
        gamma: list
            Gradient values of the input indices
        """

        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs].to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1
        return gamma
    
    def after_train(self, model_params):
        self.update_model(model_params)

    def finish_run(self):
        pass

    def select(self):

        start_time = time.time()
        user_idx_list = []
        # gammas = []
        if not self.propensity:
            source_embs = self.construct_matrix()
            budget = round(source_embs.shape[0] * self.coreset_size)
            self.compute_score(source_embs.detach())
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget,
                                                                              optimizer=self.optimizer)
            fl.fit_transform(self.dist_mat.cpu().numpy())
            user_idx_list = list(fl.ranking)
            # gammas = self.compute_gamma(user_idx_list)
        end_time = time.time()
        self.logger.info("FacilityLocation strategy data selection time is: %.4f", end_time-start_time)
        return user_idx_list
        # return total_greedy_list, torch.ones(len(gammas))

