from ..coresetdataloader import CoresetDataLoader
import torch
import time


class FixedUserDataLoader(CoresetDataLoader):
    """
    Implements of RandomDataLoader that serves as the dataloader for the non-adaptive Random subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Random subset selection strategy
    logger: class
        Logger for logging the information
    """
    def __init__(self, train_loader, dss_args, fixed_users, logger, *args, **kwargs):
        """
        Constructor function
        """
        super(FixedUserDataLoader, self).__init__(self.__class__.__name__, train_loader, dss_args,
                                                  logger, *args, **kwargs)
        self.fixed_users = fixed_users
        self.budget = len(fixed_users)
        self._init_subset_loader()

    def _init_subset_indices(self):
        """
        Function that initializes the subset indices with fixed users
        """
        return self.train_loader.dataset.user_mapping(self.fixed_users)

    def _resample_subset_indices(self):
        """
        Function that calls the Random subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        self.logger.debug("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        self.logger.debug("Fixed User budget: %d", self.budget)
        subset_indices, subset_weights = self.train_loader.dataset.user_mapping(self.fixed_users), torch.ones(self.budget)
        end = time.time()
        self.logger.info("Epoch: {0:d}, Fixed subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights
