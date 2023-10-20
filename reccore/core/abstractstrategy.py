import torch
import numpy as np
from logging import getLogger
class AbstractStrategy(object):

    def __init__(self, dataset, model, args):
        self.dataset = dataset
        self.model=model
        self.args = args
        self.logger = getLogger()
        self.device = args.train_args.device
        self.propensity = args.dss_args.propensity
        self.coreset_size = args.dss_args.coreset_size
        self.select_target = args.dss_args.selection_type
        # self.print_freq = args.dss_args.print_freq
        if self.select_target == "user":
            self.N_trn = dataset.user_num
        else:
            self.N_trn = len(dataset.train_df)

    def update_model(self, model_params):
        self.model.load_state_dict(model_params)

    def after_batch(self, *args):
        pass

    def after_epoch(self, *args):
        pass

    def after_train(self, *args):
        pass

    def select(self):
        pass

    def construct_matrix(self):
        self.model.eval()
        with torch.no_grad():
            if self.select_target == "user":
                matrix = self.model.get_all_user_embeddings()
            else:
                matrix_list = []
                u_embd = self.model.get_all_user_embeddings()
                i_embd = self.model.get_all_item_embeddings()
                for i in range(u_embd.shape[0]):
                    for item in self.dataset.train_history_dict[i]:
                        matrix_list.append(torch.cat((u_embd[i], i_embd[item]), dim=0).view(1, -1))
                matrix = torch.cat(matrix_list, dim=0)
        return matrix.detach()

    def calculate_propensity(self, target='user'):

        if target == 'user':
            # frequency
            user_freq = torch.zeros(self.dataset.user_num(), requires_grad=False)  # .to(self.args.device)
            user_counter = self.dataset.train_df.uid.value_counts()
            uids = user_counter.index
            counts = user_counter.values
            for idx, uid in enumerate(uids):
                user_freq[uid] = counts[idx]

            # propensity
            num_instances = len(self.dataset.train_df)
            A=0.55
            B=1.5
            C = (np.log(num_instances) - 1) * np.power(B + 1, A)
            wts = 1.0 + C * np.power(np.array(user_freq) + B, -A)

            return np.ravel(wts)
        else:
            # frequency
            item_freq = torch.zeros(self.dataset.item_num(), requires_grad=False)  # .to(self.args.device)
            item_counter = self.dataset.train_df.iid.value_counts()
            iids = item_counter.index
            counts = item_counter.values
            for idx, iid in enumerate(iids):
                item_freq[iid] = counts[idx]

            # propensity
            num_instances = len(self.dataset.train_df)
            A = 0.55
            B = 1.5
            C = (np.log(num_instances) - 1) * np.power(B + 1, A)
            wts = 1.0 + C * np.power(np.array(item_freq) + B, -A)

            return np.ravel(wts)