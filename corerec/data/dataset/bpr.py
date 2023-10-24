from tqdm import tqdm
import numpy as np
from corerec.data.dataset.dataset import BasicDataset
tqdm.pandas()

class DataBPR(BasicDataset):
    def __init__(self, config=None):
        super(DataBPR, self).__init__(config)
        self._load_data()
        if self.core_users is not None:
            self._build_core_user_item_map()
            self._core_similarity_mat()
            self._remapping_users_when_reconstruct_embeds()
        self._user_based_split()
        self.users = sorted(self.dataset.uid.unique())
        self.valid_profile = np.zeros(len(self.users), dtype=np.int)
        self.test_profile = np.zeros(len(self.users), dtype=np.int)
        self._build_history()
        if self.core_users is not None:
            self.train_df = self.train_df.loc[self.train_df.userId.isin(self.core_users)]
        self.refresh_negatives()

    def __getitem__(self, index):
        if self.flag == 'train':
            # the negatives in the train triplet will be refreshed at the beginning of each epoch
            return {"userId": self.train_triplet[index][0],
                    "itemId": self.train_triplet[index][1],
                    "neg_itemId": self.train_triplet[index][2]}

            # sample a negative for each pos item when fetching
            # user_id = self.train_tuple[index][0]
            # pos_item = self.train_tuple[index][1]
            # neg_item = np.random.choice(self.neg_pool[user_id])
            # return {"userId": user_id,
            #         "itemId": pos_item,
            #         "neg_itemId": neg_item}

            # return index
        elif self.flag == 'valid':
            batch_data = self.users[index]
            return {"userId": batch_data,
                    "indices": ([batch_data]*self.valid_profile[batch_data], self.valid_items[batch_data])}
        else:
            batch_data = self.users[index]
            return {"userId": batch_data,
                    "indices": ([batch_data]*self.test_profile[batch_data], self.test_items[batch_data])}

    def __len__(self):
        if self.flag == 'train':
            return len(self.train_df)
        elif self.flag == 'valid':
            return self.valid_df.uid.nunique()
        else:
            return self.test_df.uid.nunique()

