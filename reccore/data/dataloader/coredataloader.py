import copy
from abc import abstractmethod
from reccore.utils.data.data_utils import WeightedSubset
from reccore.utils import constants
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np


# Base objects
class CoreDataLoader:
    """
    Implementation of DSSDataLoader class which serves as base class for dataloaders of other
    selection strategies for supervised learning framework.

    Parameters
    -----------
    full_data: torch.utils.data.Dataset Class
        Full dataset from which data subset needs to be selected.
    dss_args: dict
        Data subset selection arguments dictionary
    logger: class
        Logger class for logging the information
    """

    def __init__(self, strategy, train_loader, dss_args, logger, *args, **kwargs):
        """
        Constructor Method
        """
        super(CoreDataLoader, self).__init__()
        # TODO: Integrate verbose in logging
        # Arguments assertion check
        assert "fraction" in dss_args.keys(), "'fraction' is a compulsory argument. Include it as a key in dss_args"
        assert "select_every" in dss_args.keys(), "'select_every' is a compulsory argument. Include it as a key in dss_args"
        assert "device" in dss_args.keys(), "'device' is a compulsory argument. Include it as a key in dss_args"
        self.train_loader = train_loader
        self.dataset = train_loader.dataset
        self.len_full = len(self.dataset)
        self.dss_args = dss_args
        self.select_every = dss_args.select_every
        self.device = dss_args.device
        if dss_args.warmup_epochs > 0:
            self.warmup_epochs = dss_args.warmup_epochs
        else:
            self.select_after = 0
            self.warmup_epochs = 0
        self.initialized = False
        if (dss_args.fraction > 1) or (dss_args.fraction < 0):
            raise ValueError("'fraction' should lie between 0 and 1")

        self.fraction = dss_args.fraction
        self.budget = int(self.len_full * self.fraction)
        self.logger = logger
        self.dataset.set_flag("train")
        self.loader_args = args
        self.loader_kwargs = kwargs
        self.subset_indices = None
        self.subset_weights = None
        self.subset_loader = None
        self.batch_wise_indices = None
        self.strategy_name = strategy[:-10].lower()
        self.cur_epoch = 1
        wt_trainset = WeightedSubset(self.dataset, list(range(self.len_full)), [1] * self.len_full)
        self.wtdataloader = torch.utils.data.DataLoader(wt_trainset, *self.loader_args, **self.loader_kwargs)
        # self._init_subset_loader()

    def __getattr__(self, item):
        return object.__getattribute__(self, "subset_loader").__getattribute__(item)

    def __iter__(self):
        """
        Iter function that returns the iterator of full data loader or data subset loader or empty loader based on the
        warmstart kappa value.
        """
        self.initialized = True
        self.cur_epoch += 1
        return self.subset_loader.__iter__()

    def __len__(self) -> int:
        """
        Returns the length of the current data loader
        """
        return len(self.subset_loader)

    def _init_subset_loader(self):
        """
        Function that initializes the random data subset loader
        """
        # If warm-up is not set, all strategies start with random selection
        self.subset_indices = self._init_subset_indices()
        self.subset_weights = torch.ones(self.budget)
        self._refresh_subset_loader()

    # Default subset indices comes from random selection
    def _init_subset_indices(self):
        """
        Function that initializes the subset indices randomly
        """
        return list(np.random.choice(self.len_full, size=self.budget, replace=False))

    def _refresh_subset_loader(self):
        """
        Function that regenerates the data subset loader using new subset indices and subset weights
        """
        self.subset_loader = DataLoader(WeightedSubset(self.dataset, self.subset_indices, self.subset_weights),
                                        *self.loader_args, **self.loader_kwargs)
        self.batch_wise_indices = list(self.subset_loader.batch_sampler)

    def resample(self):
        """
        Function that resamples the subset indices and recalculates the subset weights
        """
        #self.best_model_dict = torch.load(self.dss_args.model_dict_path)
        self.subset_indices, self.subset_weights = self._resample_subset_indices()
        self.logger.debug("Subset indices length: %d", len(self.subset_indices))
        self._refresh_subset_loader()
        self.logger.debug("Subset loader initiated, args: %s, kwargs: %s", self.loader_args, self.loader_kwargs)
        self.logger.debug('Subset selection finished, Training data size: %d, Subset size: %d',
                          self.len_full, len(self.subset_loader.dataset))

    @abstractmethod
    def _resample_subset_indices(self):
        """
        Abstract function that needs to be implemented in the child classes.
        Needs implementation of subset selection implemented in child classes.
        """
        raise Exception('Not implemented.')

    @abstractmethod
    def _record_intermediate_gradient(self):
        """
        Abstract function that needs to be implemented in the child classes.
        Needs implementation of recording gradient matrix in child classes.
        """
        raise Exception('Not implemented.')
