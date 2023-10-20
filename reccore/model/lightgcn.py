import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from reccore.model.init import xavier_uniform_initialization

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()

class LightGCN(nn.Module):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, dataset, config):
        super(LightGCN, self).__init__()

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

        # load dataset info
        self.interaction_matrix = dataset.build_sparse_inter_matrix(form='coo', core=True if self.core_train else False,
                                                                    shape=(self.n_train_users, self.n_items)
                                                                    ).astype(np.float32)

        # load parameters info
        self.latent_dim = config.model.embedding_size  # int type:the embedding size of lightGCN
        self.n_layers = config.model.n_layers  # int type:the layer num of lightGCN

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_train_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        # self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_train_users + self.n_items, self.n_train_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_train_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_train_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, batch):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = batch["userId"]
        pos_item = batch["itemId"]
        neg_item = batch["neg_itemId"]

        user_all_embeddings, item_all_embeddings = self._aggregate()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        # calculate Reg Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        return pos_scores, neg_scores, (u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

    def _aggregate(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_train_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def restore_embeddings(self):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self._aggregate()

    def full_predict(self, batch):
        user = batch["userId"]

        if self.core_train:
            user_e = self.full_user_embeddings[user]
        else:
            user_e = self._get_user_embedding(user)
        all_item_e = self.get_all_item_embeddings()
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def build_user_embeddings(self):
        self.restore_embeddings()
        if self.core_train:
            built_embeds = torch.matmul(self.sim_mat, self.get_all_user_embeddings())
            all_u_embeds = torch.vstack((self.get_all_user_embeddings(), built_embeds))
            self.full_user_embeddings = all_u_embeds[self.user_id_for_embed_construction]
        else:
            pass

    def _get_user_embedding(self, user):
        self.restore_embeddings()
        return self.restore_user_e[user]

    def _get_item_embedding(self, item):
        self.restore_embeddings()
        return self.restore_item_e[item]

    def get_all_user_embeddings(self):
        self.restore_embeddings()
        return self.restore_user_e

    def get_all_item_embeddings(self):
        self.restore_embeddings()
        return self.restore_item_e