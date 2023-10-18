import numpy as np
from tqdm import tqdm
from corerec.data.dataset.dataset import BasicDataset
from reccore.utils.utils import set_color
from copy import deepcopy
tqdm.pandas()


class DataMultiVAE(BasicDataset):
    def __init__(self, config=None):
        super(DataMultiVAE, self).__init__(config)

        self._load_data()
        self._user_based_split()
        self.all_users = sorted(self.dataset.uid.unique())
        self.valid_profile = np.zeros(len(self.all_users), dtype=np.int)
        self.test_profile = np.zeros(len(self.all_users), dtype=np.int)
        self._build_history()
        if self.core_users is not None:
            self.train_df = self.train_df.loc[self.train_df.userId.isin(self.core_users)]
        self.train_users = sorted(self.train_df.uid.unique())


    def _build_history(self):
        self.logger.info(set_color("Building user interaction history...", "blue"))
        for uid, pdf in tqdm(self.train_df.groupby("uid"), total=self.user_num, desc=set_color(f"Train items", "pink")):
            key = int(uid)
            # self.len_profile[uid] = len(pdf.iid.values)
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
            batch_data = self.train_users[index]
            return {"userId": batch_data}
        elif self.flag == 'valid':
            batch_data = self.all_users[index]
            return {"userId": batch_data,
                    "indices": ([batch_data]*self.valid_profile[batch_data], self.valid_items[batch_data])}
        else:
            batch_data = self.all_users[index]
            return {"userId": batch_data,
                    "indices": ([batch_data]*self.test_profile[batch_data], self.test_items[batch_data])}

    def __len__(self):
        if self.flag == 'train':
            return self.train_df.uid.nunique()
        elif self.flag == 'valid':
            return self.valid_df.uid.nunique()
        else:
            return self.test_df.uid.nunique()


