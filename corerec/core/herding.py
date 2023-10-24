import time

from .abstractstrategy import AbstractStrategy
import torch
import numpy as np
from .methods_utils import euclidean_dist


class Herding(AbstractStrategy):
    def __init__(self, dataset, model, args):
        super(Herding, self).__init__(dataset, model, args)
        self.distance = args.dss_args.distance
        if self.distance == "euclidean":
            self.metric = euclidean_dist

    def herding(self, matrix, coreset_size, index=None):

        sample_num = matrix.shape[0]
        budget = round(sample_num * coreset_size)
        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            mu = torch.mean(matrix, dim=0)
            select_result = np.zeros(sample_num, dtype=bool)

            if not self.propensity:
                for i in range(budget):
                    # if i % self.print_freq == 0:
                    #     print("| Selecting [%3d/%3d]" % (i + 1, budget))
                    dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                       matrix[~select_result])
                    p = torch.argmax(dist).item()
                    p = indices[~select_result][p]
                    select_result[p] = True
            else:
                if self.select_target == "user":
                    propensity = self.calculate_propensity(target="user")
                    for i in range(budget):
                        # if i % self.print_freq == 0:
                        #     print("| Selecting [%3d/%3d]" % (i + 1, budget))
                        dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                           matrix[~select_result])
                        # propensity = propensity[~select_result]
                        dist = dist * propensity[~select_result]
                        p = torch.argmax(dist).item()
                        p = indices[~select_result][p]
                        select_result[p] = True
                else:
                    propensity = np.zeros(matrix.shape[0])
                    user_propensity = self.calculate_propensity(target="user")
                    item_propensity = self.calculate_propensity(target="item")

                    # put the propensity score to the corresponding position in the propensity vector
                    # for instance, dist[0] has propensity score at propensity[0]
                    for position in range(matrix.shape[0]):
                        (uid, iid) = self.dataset.ui_decoder[position]
                        u_p = user_propensity[uid]
                        i_p = item_propensity[iid]
                        max_p = max(u_p, i_p)
                        propensity[position] = max_p

                    for i in range(budget):
                        # if i % self.print_freq == 0:
                        #     print("| Selecting [%3d/%3d]" % (i + 1, budget))
                        dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                           matrix[~select_result])
                        # propensity = propensity[~select_result]
                        dist = dist * propensity[~select_result]
                        p = torch.argmax(dist).item()
                        p = indices[~select_result][p]
                        select_result[p] = True

        if index is None:
            index = indices
        return index[select_result]

    def after_train(self, model_params):
        self.update_model(model_params)

    def finish_run(self):
        pass

    def select(self):
        start = time.time()
        selection_result = self.herding(self.construct_matrix(), coreset_size=self.coreset_size)
        end = time.time()
        self.logger.info("Herding strategy data selection time is: %.4f", end - start)
        return selection_result

