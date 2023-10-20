import time

import torch
import torch.nn as nn
from reccore.model.init import xavier_normal_initialization

import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
from time import time
import similaripy as sim

class ItemKNN(nn.Module):

    def __init__(self, dataset, config):
        super(ItemKNN, self).__init__()

        self.device = config.train_args.device
        self.n_real_users = dataset.user_num
        self.n_train_users = dataset.train_df.uid.nunique()
        self.n_items = dataset.item_num
        self.core_train = True if dataset.core_users is not None else False
        self.full_interaction_matrix = dataset.full_interaction_matrix

        if self.core_train:
            self.train_interaction_matrix = dataset.train_interaction_matrix
        else:
            self.train_interaction_matrix = self.full_interaction_matrix

        self.w = sim.cosine(self.train_interaction_matrix.T, k=100, verbose=False)
        if self.full_interaction_matrix.shape[0] < self.w.shape[0]:
            self.w = self.w.tocsc()
            self.full_interaction_matrix = self.full_interaction_matrix.tocsc()
        self.pred_mat = self.full_interaction_matrix.dot(self.w.T)
        self.fake_loss = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return self.fake_loss

    """
    def predict(self, batch):
        user_idx = batch[0]
        item_idx = batch[1]

        user = user_idx.cpu().numpy().astype(int)
        item = item_idx.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.pred_mat[uid, iid]
            result.append(score)
        result = torch.from_numpy(np.array(result))

        return result
    """

    def full_predict(self, batch):
        user = batch["userId"].cpu().numpy()
        score = torch.from_numpy(self.pred_mat[user, :].toarray())

        return score

    def build_user_embeddings(self):
        pass



