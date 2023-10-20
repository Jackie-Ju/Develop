import numpy as np
from tqdm import tqdm
from reccore.data.dataset.dataset import BasicDataset
from reccore.utils.utils import set_color
tqdm.pandas()

class DataItemKNN(BasicDataset):
    def __init__(self, config=None):
        super(DataItemKNN, self).__init__(config)
        self._load_data()
        if self.core_users is not None:
            self._build_core_user_item_map()
        self._user_based_split()
        self.full_interaction_matrix = self._build_sparse_matrix(self.train_df, 'csr',
                                                                 shape=(self.user_num, self.item_num),
                                                                 core=False
                                                                 ).astype(np.float32)
        self.users = sorted(self.dataset.uid.unique())
        self.valid_profile = np.zeros(len(self.users), dtype=np.int)
        self.test_profile = np.zeros(len(self.users), dtype=np.int)
        self._build_history()
        if self.core_users is not None:
            self.train_df = self.train_df.loc[self.train_df.userId.isin(self.core_users)]
            self.train_interaction_matrix = self._build_sparse_matrix(self.train_df, 'csr',
                                                                      shape=(self.train_df.uid.nunique(), self.item_num)
                                                                      ).astype(np.float32)

    def _build_history(self):
        self.logger.info(set_color("Building user interaction history...", "blue"))
        for uid, pdf in tqdm(self.train_df.groupby("uid"), total=self.user_num, desc=set_color(f"Train items", "pink")):
            key = int(uid)
            if key not in self.history_dict.keys():
                self.history_dict[key] = []
                self.history_value_dict[key] = []
            self.history_dict[key].extend(pdf.iid.values)
            self.history_value_dict[key].extend(pdf.rating)

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

    def __getitem__(self, index):
        if self.flag == 'train':
            return {"userId": self.train_df.uid.values[index]}
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
