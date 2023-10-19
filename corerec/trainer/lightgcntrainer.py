import numpy as np
import torch
from corerec.trainer.trainer import GeneralTrainer, GeneralSelectionTrainer

class LightGCNTrainer(GeneralTrainer):

    def __init__(self, model, dataset,
                 train_loader, valid_loader, test_loader,
                 loss_fn, optimizer, scheduler, args):
        super(LightGCNTrainer, self).__init__(model, dataset,
                                              train_loader, valid_loader, test_loader, args)
        self.loss = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_idx_dict = self.dataset.history_dict

    def _batch_train(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.args.train_args.device)
        self.optimizer.zero_grad()
        pos_score, neg_score, ego_embeddings = self.model.forward(batch)
        loss = self.loss(pos_score, neg_score, ego_embeddings)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss

    def _full_batch_eval(self, bid, batch_data, isTest):
        offset = bid * self.args.dataloader.batch_size
        u_i = batch_data["indices"]
        users = u_i[0]
        items = u_i[1]
        batch_data["userId"] = batch_data["userId"].to(self.args.train_args.device)
        scores = self.model.full_predict(batch_data)

        unique_users, counts = torch.unique(users, sorted=True, return_counts=True)
        positive_u = u_i[0] - offset
        scores = scores.cpu()
        for row in range(len(unique_users)):
            u_id = unique_users[row].item()
            scores[row][self.dataset.history_dict[u_id]] = -np.inf
        if isTest:
            self.score_mat.append(scores)

        return positive_u, items, scores, unique_users

class LightGCNSelectionTrainer(GeneralSelectionTrainer):

    def __init__(self, model, dataset,
                 train_loader, valid_loader, test_loader,
                 loss_fn, optimizer, scheduler, args, strategy):
        super(LightGCNSelectionTrainer, self).__init__(model, dataset,
                                                       train_loader, valid_loader, test_loader,
                                                       args, strategy)
        self.loss = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_idx_dict = self.dataset.history_dict

    def _batch_train(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.args.train_args.device)
        self.optimizer.zero_grad()
        pos_score, neg_score, ego_embeddings = self.model.forward(batch)
        loss = self.loss(pos_score, neg_score, ego_embeddings)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss

    def _full_batch_eval(self, bid, batch_data, isTest):
        offset = bid * self.args.dataloader.batch_size
        u_i = batch_data["indices"]
        users = u_i[0]
        items = u_i[1]
        batch_data["userId"] = batch_data["userId"].to(self.args.train_args.device)
        scores = self.model.full_predict(batch_data)

        unique_users, counts = torch.unique(users, sorted=True, return_counts=True)
        positive_u = u_i[0] - offset
        scores = scores.cpu()
        for row in range(len(unique_users)):
            u_id = unique_users[row].item()
            scores[row][self.dataset.history_dict[u_id]] = -np.inf
        if isTest:
            self.score_mat.append(scores)

        return positive_u, items, scores, unique_users

