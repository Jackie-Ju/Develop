import numpy as np
import torch
import os
from logging import getLogger
from corerec.utils.utils import ensure_dir, set_color


class EarlyStopping:

    def __init__(self, early_stop=False, patience=10, delta=0, ranking=False, args=None):

        self.args = args
        self.logger = getLogger()
        self.save_root = self.args.early_stopping.root
        # self.save_name = os.path.join(self.save_root,
        #                               f"{self.args.method_type}/{self.args.method_type}-{self.args.model.architecture}-" \
        #                               f"{self.args.dataset.name}-{self.args.core_args.fraction}.pth")
        self.save_name = args.eval_args.model_dict_path

        self.patience = patience
        self.verbose = args.verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = early_stop
        self.stop = False
        self.val_score = np.Inf
        self.delta = delta
        self.ranking = ranking

    def __call__(self, val_score, model, epoch):
        update_flag = False
        score = val_score if self.ranking else -val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, epoch)
            update_flag = True
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_score, model, epoch)
            self.counter = 0
            update_flag = True
        else:
            if self.early_stop:
                self.counter += 1
                stopping_message = set_color(f"Early Stopping Counter: {self.counter} out of {self.patience}", "yellow")
                self.logger.info(stopping_message)
                # print(f"Early Stopping Counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    stop_msg = set_color(f"Max patience reached, early stopping.", "yellow")
                    self.logger.info(stop_msg)
                    self.stop = True
        return update_flag

    def save_checkpoint(self, val_score, model, epoch):
        ensure_dir(self.save_root)
        ensure_dir(os.path.join(self.save_root, self.args.method_type))

        if epoch < 2:
            save_msg = set_color(f"Model saved after first epoch, validation score: {val_score}", "yellow")
            self.logger.info(save_msg)
        else:
            save_msg = set_color(f"Better parameters found, validation score: {val_score}", "yellow")
            self.logger.info(save_msg)
        torch.save(model.state_dict(), self.save_name)

    @property
    def save_path(self):
        return self.save_name

