import numpy as np
import torch
from reccore.trainer.trainer import GeneralTrainer

class MultiVAETrainer(GeneralTrainer):

    def __init__(self, model, dataset,
                 train_loader, valid_loader, test_loader,
                 loss_fn, optimizer, scheduler, args):
        super(MultiVAETrainer, self).__init__(model, dataset,
                                              train_loader, valid_loader, test_loader, args)

        self.loss = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_idx_dict = self.dataset.history_dict

    def _batch_train(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.args.train_args.device)
        self.optimizer.zero_grad()
        outputs, mu, logvar, targets = self.model(batch)
        loss = self.loss(outputs, mu, logvar, targets)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss

    def _full_batch_eval(self, bid, batch_data, isTest):
        offset = bid * self.args.dataloader.batch_size
        sparse_batch = batch_data["indices"]
        users = sparse_batch[0]
        items = sparse_batch[1]
        batch_data["userId"] = batch_data["userId"].to(self.args.train_args.device)
        scores, hist_matrix = self.model.full_predict(batch_data)

        unique_users, counts = torch.unique(users, sorted=True, return_counts=True)
        positive_u = sparse_batch[0] - offset
        scores[hist_matrix > 0] = -np.inf
        scores = scores.cpu()
        if isTest:
            self.score_mat.append(scores)
        return positive_u, items, scores, unique_users
