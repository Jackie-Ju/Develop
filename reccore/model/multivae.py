import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class MultiVAE(nn.Module):
    def __init__(self, dataset, config):
        super(MultiVAE, self).__init__()
        self.layers = config.model.mlp_hidden_size
        self.lat_dim = config.model.latent_dimension
        self.drop_out = config.model.dropout_prob
        self.device = config.train_args.device
        self.core_train = True if dataset.core_users is not None else False

        self.item_num = dataset.item_num
        self.user_history = dataset.train_history_dict
        self.user_value_history = dataset.train_history_value_dict

        self.encode_layer_dims = [self.item_num] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

                Args:
                    user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

                Returns:
                    torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
                """

        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = []
        history_num = []
        values = []
        for u in user:
            u_history = self.user_history[u.item()]
            history_num.append(len(u_history))
            col_indices.extend(u_history)
            values.extend(self.user_value_history[u.item()])

        history_num = torch.LongTensor(history_num)
        # row_indices = torch.arange(user.shape[0]).to(self.device).repeat_interleave(history_num, dim=0)
        # rating_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        row_indices = torch.arange(len(user)).repeat_interleave(history_num, dim=0)
        rating_matrix = torch.zeros(1).repeat(len(user), self.item_num)
        # rating_matrix.index_put_((row_indices, torch.tensor(col_indices)), torch.ones(len(col_indices)))
        rating_matrix.index_put_((row_indices, torch.tensor(col_indices)), torch.tensor(values, dtype=torch.float32))
        return rating_matrix.to(self.device)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, batch):

        rating_matrix = self.get_rating_matrix(batch["userId"])

        h = F.normalize(rating_matrix)
        h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)

        mu = h[:, :int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2):]

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar, rating_matrix

    def user_latent_representation(self, batch):
        rating_matrix = self.get_rating_matrix(batch["userId"])
        h = F.normalize(rating_matrix)
        h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)
        return h

    def build_user_embeddings(self):
        pass

    '''
    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        scores, _, _ = self.forward(rating_matrix)

        return scores[[torch.arange(len(item)).to(self.device), item]]
    '''

    def full_predict(self, batch):
        scores, _, _, rating_matrix = self.forward(batch)
        return scores, rating_matrix


