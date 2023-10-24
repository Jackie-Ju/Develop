import time

import torch
import torch.nn as nn
from corerec.model.init import xavier_normal_initialization

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()

class BPR(nn.Module):

    def __init__(self, dataset, config):
        super(BPR, self).__init__()
        self.device = config.train_args.device
        self.n_real_users = dataset.user_num
        self.n_train_users = dataset.train_df.uid.nunique()
        self.n_items = dataset.item_num
        self.core_train = True if dataset.core_users is not None else False
        if self.core_train:
            self.core_uid = dataset.core_user_uid
            self.sim_mat = dataset.torch_user_similarity_matrix.to(self.device)
            self.core_user_simuid = dataset.core_user_simuid
            self.noncore_user_simuid = dataset.noncore_user_simuid
            self.user_id_for_embed_construction = dataset.get_remapped_users.to(self.device)
            self.full_user_embeddings = None
        # load parameters info
        self.embedding_size = config.model.embedding_size


        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_train_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, batch):
        user = batch["userId"]
        item = batch["itemId"]
        neg_item = batch["neg_itemId"]
        user_e = self._get_user_embedding(user)
        pos_e = self._get_item_embedding(item)
        neg_e = self._get_item_embedding(neg_item)

        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1),\
            torch.mul(user_e, neg_e).sum(dim=1)
        return pos_item_score, neg_item_score

    # def predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #     user_e, item_e = self.forward(user, item)
    #     return torch.mul(user_e, item_e).sum(dim=1)

    def full_predict(self, batch):
        user = batch["userId"]
        if self.core_train:
            user_e = self.full_user_embeddings[user]
        else:
            user_e = self._get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def build_user_embeddings(self):
        if self.core_train:
            built_embeds = torch.matmul(self.sim_mat, self.get_all_user_embeddings())
            all_u_embeds = torch.vstack((self.get_all_user_embeddings(), built_embeds))
            self.full_user_embeddings = all_u_embeds[self.user_id_for_embed_construction]
        else:
            pass

    def _get_user_embedding(self, user):
        return self.user_embedding(user)

    def _get_item_embedding(self, item):
        return self.item_embedding(item)

    def get_all_user_embeddings(self):
        return self.user_embedding.weight

    def get_all_item_embeddings(self):
        return self.item_embedding.weight
