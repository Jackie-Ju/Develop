import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import ceil
import similaripy as sim
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
from logging import getLogger
from reccore.utils.utils import set_color
from copy import deepcopy
tqdm.pandas()

class BasicDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.logger = getLogger()
        self.flag = None
        self.core_users = config.core_user
        self.history_dict = {}
        self.history_value_dict = {}
        self.neg_pool = {}
        self.valid_items = {}
        self.test_items = {}

    def _load_data(self):
        path = os.path.join(self.config.datadir, self.config.name)
        file = os.path.join(path, self.config.name + '.csv')
        self.logger.info(set_color("Loading dataset", "blue") + f" {self.config.name}")
        self.dataset = pd.read_csv(file)
        self.user_uid_map, self.item_iid_map = self._build_user_item_map()

        self.dataset['uid'] = self.dataset['userId'].map(self.user_uid_map)
        self.dataset['iid'] = self.dataset['itemId'].map(self.item_iid_map)

    def _build_user_item_map(self):
        r"""Mapping original user/item ids to internal ids

        Returns:
            user_to_uid: A dictionary maps original user id to internal user id.
            item_to_uid: A dictionary maps original item id to internal item id.

        """
        self.logger.info(set_color("Building internal mapping...", "blue"))
        unique_users = self.dataset['userId'].unique()
        user_to_uid = {old: new for new, old in enumerate(unique_users)}

        unique_items = self.dataset['itemId'].unique()
        item_to_iid = {old: new for new, old in enumerate(unique_items)}

        self.uid_user_map = {}
        self.iid_item_map = {}
        for user, uid in user_to_uid.items():
            self.uid_user_map[uid] = user
        for item, iid in item_to_iid.items():
            self.iid_item_map[iid] = item

        return user_to_uid, item_to_iid

    def _build_core_user_item_map(self):
        r"""
        mapping the internal uid of untrained/trained (core) users to a new simuid [0, n_untrained_user], [0, n_core_user]
        mapping the internal iid of untrained/trained (core) items to a new simiid [0, n_untrained_item], [0, n_core_item]

        The dataset will now contain fields:
        user_Id, item_Id, rating, timestamp, uid, iid, simuid, simiid

        coreset user simuid: 0 - n_core
        non-coreset user simuid: 0 - n_noncore

        """
        self.logger.info(set_color("Building core user/item mapping...", "blue"))

        self.core_uid = [self.user_uid_map[raw_id] for raw_id in self.core_users]
        noncore_uid = np.setdiff1d(self.dataset['uid'].unique(), self.core_uid)
        self.core_uid_to_simuid = {uid: simuid for simuid, uid in enumerate(self.core_uid)}
        self.noncore_uid_to_simuid = {uid: simuid for simuid, uid in enumerate(noncore_uid)}

        simuid_list = []
        for uid in self.dataset['uid'].values:
            if uid in self.core_uid_to_simuid.keys():
                simuid_list.append(self.core_uid_to_simuid[uid])
            else:
                simuid_list.append(self.noncore_uid_to_simuid[uid])
        self.dataset['simuid'] = simuid_list

        self.core_items = self.dataset.loc[self.dataset.userId.isin(self.core_users)].itemId.unique()
        self.core_iid = [self.item_iid_map[raw_id] for raw_id in self.core_items]
        noncore_iid = np.setdiff1d(self.dataset['iid'].unique(), self.core_iid)
        self.core_iid_to_simiid = {iid: simiid for simiid, iid in enumerate(self.core_iid)}
        self.noncore_iid_to_simiid = {iid: simiid for simiid, iid in enumerate(noncore_iid)}

        simiid_list = []
        for iid in self.dataset['iid'].values:
            if iid in self.core_iid_to_simiid.keys():
                simiid_list.append(self.core_iid_to_simiid[iid])
            else:
                simiid_list.append(self.noncore_iid_to_simiid[iid])
        self.dataset['simiid'] = simiid_list

    def _build_sparse_matrix(self, dataset=None, form='coo', explicit=False, shape=None, core=True):
        if not core:
            index_u = dataset['uid'].values
        else:
            index_u = dataset['simuid'].values
        index_i = dataset['iid'].values
        if explicit:
            data = dataset['rating'].values
        else:
            data = np.ones_like(dataset['rating'].values, dtype=float)
        mat = coo_matrix((data, (index_u, index_i)), shape=shape)

        if form == 'csr':
            return mat.tocsr()
        if form == 'dok':
            return mat.todok()
        if form == 'csc':
            return mat.tocsc()
        return mat

    def build_sparse_inter_matrix(self, form='coo', core=False, shape=None):
        return self._build_sparse_matrix(self.train_df, form=form, core=core, shape=shape)

    def _core_similarity_mat(self):
        r"""Calculating the cosine similarity between non-coreset users and coreset users based on their history.

        Key Variables:
            core_data: An interaction matrix of coreset users with shape [n_core, n_items]

            noncore_data: An ineraction matrix of non-coreset users with shape [n_noncore, n_items]

            untrained_user_sim_mat: The consine similarity matrix between non-coreset users and coreset users,
            with shape [n_noncore, n_core]

        """
        core_data = self.dataset.loc[self.dataset.userId.isin(self.core_users)].copy()
        noncore_data = self.dataset.loc[~self.dataset.userId.isin(self.core_users)].copy()

        self.logger.info(set_color("Building untrained/trained user similarity matrix...", "blue"))
        core_data.sort_values(by='simuid', ascending=True, ignore_index=True, inplace=True)
        noncore_data.sort_values(by='simuid', ascending=True, ignore_index=True, inplace=True)
        core_user_sparse_mat = self._build_sparse_matrix(dataset=core_data, explicit=False,
                                                    shape=(len(self.core_users),self.item_num))
        noncore_user_sparse_mat = self._build_sparse_matrix(dataset=noncore_data, explicit=False,
                                                       shape=(noncore_data['simuid'].nunique(),self.item_num))
        self.untrained_user_sim_mat = sim.cosine(noncore_user_sparse_mat, core_user_sparse_mat.T,
                                                 k=core_user_sparse_mat.shape[0], verbose=True)
        # with open(
        #         f"/mnt/recsys/zhengju/projects/CoreRec/short_paper/scores_embeddings/{self.args.model.architecture}/similarity.npy",
        #         'wb') as f:
        #     np.save(f, self.untrained_user_sim_mat.dense())

        # self.logger.info(set_color("Building untrained/trained item similarity matrix...", "blue"))
        # core_data.sort_values(by='simiid', ascending=True, ignore_index=True, inplace=True)
        # noncore_data.sort_values(by='simiid', ascending=True, ignore_index=True, inplace=True)
        # core_item_sparse_mat = self._build_sparse_matrix(dataset=core_data, explicit=False,
        #                                                  shape=(len(self.core_items), self.user_num))
        # noncore_item_sparse_mat = self._build_sparse_matrix(dataset=noncore_data, explicit=False,
        #                                                     shape=(noncore_data['simiid'].nunique(), self.user_num))
        # self.untrained_item_sim_mat = sim.cosine(noncore_item_sparse_mat, core_item_sparse_mat.T,
        #                                          k=core_item_sparse_mat.shape[0], verbose=True)

    def _user_based_split(self):
        self.logger.info(set_color("Splitting dataset...", "blue"))
        test_idx = []
        valid_idx = []
        train_idx = []
        train_ratio = self.config.ratio['train']
        valid_ratio = self.config.ratio['valid']
        test_ratio = self.config.ratio['test']
        for _, def_u in tqdm(self.dataset.groupby('userId'),
                             total=self.user_num,
                             desc=set_color(f"Split data", "pink")):
            indices = def_u.index.values
            split_size = ceil(test_ratio * len(indices))
            selected_test_idx = np.random.choice(indices, split_size, replace=False)
            # selected_train_valid_idx = np.random.choice(indices, split_size, replace=False)

            selected_train_valid_idx = indices[~np.isin(indices, selected_test_idx)]

            valid_size = ceil(valid_ratio * len(indices))
            selected_valid_idx = np.random.choice(selected_train_valid_idx, valid_size, replace=False)

            selected_train_idx = selected_train_valid_idx[~np.isin(selected_train_valid_idx, selected_valid_idx)]

            test_idx.extend(selected_test_idx)
            valid_idx.extend(selected_valid_idx)
            train_idx.extend(selected_train_idx)
        self.test_df = self.dataset.loc[test_idx].reset_index(drop=True)
        self.valid_df = self.dataset.loc[valid_idx].reset_index(drop=True)
        self.train_df = self.dataset.loc[train_idx].reset_index(drop=True)

    def refresh_negatives(self):
        r"""
        Construct negative samples.
        """
        self.train_triplet = []
        for user in self.train_df.uid.unique():
            user_neg_pool = self.neg_pool[user]
            user_pos_items = self.history_dict[user]
            user_neg_items = np.random.choice(user_neg_pool, len(user_pos_items), replace=True)
            if self.core_users is not None:
                user = self.core_uid_to_simuid[user]
            self.train_triplet.extend(zip([user]*len(user_pos_items), user_pos_items, user_neg_items))
        self._shuffle()

    def _shuffle(self):
        random.shuffle(self.train_triplet)

    def _build_history(self):
        r"""Store the interaction history of each user.

        """
        all_items = np.arange(0, self.item_num, 1)
        self.logger.info(set_color("Building user interaction history...", "blue"))
        for uid, pdf in tqdm(self.train_df.groupby("uid"), total=self.user_num, desc=set_color(f"Train items", "pink")):
            key = int(uid)
            # self.len_profile[uid] = len(pdf.iid.values)
            if key not in self.history_dict.keys():
                self.history_dict[key] = []
                self.history_value_dict[key] = []
                self.neg_pool[key] = []
            self.history_dict[key].extend(pdf.iid.values)
            self.history_value_dict[key].extend(pdf.rating)
            self.neg_pool[key].extend(np.setdiff1d(all_items, pdf.iid.values))

        for uid, pdf in tqdm(self.valid_df.groupby("uid"), total=self.user_num, desc=set_color(f"Valid items", "pink")):
            key = int(uid)
            self.valid_profile[uid] = len(pdf)
            if key not in self.valid_items.keys():
                self.valid_items[key] = []
            self.valid_items[key].extend(pdf.iid.values)

        for uid, pdf in tqdm(self.test_df.groupby("uid"), total=self.user_num, desc=set_color(f"Test items", "pink")):
            key = int(uid)
            self.test_profile[uid] = len(pdf)
            if key not in self.test_items.keys():
                self.test_items[key] = []
            self.test_items[key].extend(pdf.iid.values)

    def _remapping_users_when_reconstruct_embeds(self):
        r"""When coreset training is enabled, transductive models are trained using the simuid of the users.
        In the validation or evaluation phase where uid is used, transductive modes use the remapped user ids to
        reconstruct full user embeddings.

        The full user embeddings are constructed by stacking the coreset user embeddings and non-coreset user embeddings.
        The shape of the reconstructed embeddings is [core+noncore, dim].
        The remapped_users are mapped user ids that is used to re-order the embeddings so that the order of the
        embedding as the same to uid.

        Given uid 0, its embedding should be the first one. The corresponding embedding in the transductive model should
        be extracted and placed as the first one.
        If uid 0 is in the coreset, the correct embedding index in the reconstructed embeddings is uid 0 ~ core_simuid X.
        The Xth embedding is extracted and placed position 0.
        If uid 0 is not in the coreset, the mapped id will be uid 0 ~ noncore_simuid X+n_core. The X+n_core-th embedding
        is extracted and placed position 0.

        """
        self.logger.info(set_color("Building mapped user id for embedding construction...", "blue"))
        n_cores = len(self.core_users)
        mapped_users = []
        for i in tqdm(range(self.user_num)):
            if i in self.core_uid:
                mapped_users.append(self.core_uid_to_simuid[i])
            else:
                mapped_users.append(self.noncore_uid_to_simuid[i]+n_cores)
        self.remapped_users = torch.LongTensor(mapped_users)

    def _sorted_user(self):
        if "user_type" in self.config.keys():
            if self.config["user_type"] == "head":
                self.sorted_users = self.train_df.uid.value_counts().index.values
            else:
                self.sorted_users = self.train_df.uid.value_counts(ascending=True).index.values

    def set_flag(self, flag):
        self.flag = flag

    @property
    def item_num(self):
        return self.dataset.itemId.nunique()

    @property
    def user_num(self):
        return self.dataset.userId.nunique()

    @property
    def interaction_num(self):
        return len(self.dataset)

    @property
    def train_history_dict(self):
        return self.history_dict

    @property
    def train_history_value_dict(self):
        return self.history_value_dict

    @property
    def core_user_uid(self):
        return self.core_uid

    @property
    def core_user_simuid(self):
        return self.core_uid_to_simuid

    @property
    def noncore_user_simuid(self):
        return self.noncore_uid_to_simuid

    @property
    def coo_user_similarity_matrix(self):
        return self.untrained_user_sim_mat

    @property
    def dense_user_similarity_matrix(self):
        return self.untrained_user_sim_mat.toarray()

    @property
    def torch_user_similarity_matrix(self):
        return torch.from_numpy(self.dense_user_similarity_matrix)

    @property
    def get_remapped_users(self):
        return self.remapped_users

    def user_mapping(self, original_idx):
        return [self.user_uid_map[i] for i in original_idx]

    def item_mapping(self, original_idx):
        return [self.item_iid_map[i] for i in original_idx]




