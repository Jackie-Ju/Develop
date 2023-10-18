import numpy as np
import torch
from torch.utils.data import Sampler


class BPRSampler(Sampler):
    def __init__(self, data_source, num_negatives=1):
        super().__init__(data_source)
        self.data = data_source
        self.num_negatives = num_negatives

    def __iter__(self):
        shuffled_indices = np.random.permutation(len(self.data))  # Shuffle the indices
        for idx in shuffled_indices:
            user = self.data.train_tuple[idx][0]
            item = self.data.train_tuple[idx][1]
            # user = data["userId"]
            # item = data["itemId"]
            for _ in range(self.num_negatives):
                negative_item = torch.randint(0, self.data.item_num, (1,)).item()
                while negative_item in self.data.history_dict[user.item()]:
                    negative_item = torch.randint(0, self.data.item_num, (1,)).item()
                yield {"userId": user,
                       "itemId": item,
                       "neg_itemId": negative_item}

    def __len__(self):
        return len(self.data) * self.num_negatives
