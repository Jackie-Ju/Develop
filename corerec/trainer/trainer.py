import time
import numpy as np
from tqdm import tqdm
import os
import torch
import copy
from logging import getLogger
from corerec.utils.early_stopping import EarlyStopping
from corerec.evaluator.evaluator import Collector, Evaluator
from corerec.utils.utils import set_color, generate_train_loss_output, get_gpu_usage

class GeneralTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    GeneralTrainer is a general class in which the training (fit) and evaluation (evaluate) logic have been implemented.
    The functioning methods _batch_train() and _full_batch_eval() should be implemented according to different training
    and evaluation strategies.
    """

    def __init__(self, model, dataset,
                 train_loader, valid_loader, test_loader, args):
        self.model = model
        self.dataset = dataset
        self.args = args
        self.logger = getLogger()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.collector = None
        self.evaluator = None
        self.score_mat = []

    def _batch_train(self, iter_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def _full_batch_eval(self, bid, batch_data, isTest):
        raise NotImplementedError('Method [next] should be implemented.')

    def fit(self):
        analysis_dict = {"train_epoch": [], "train_loss": [], "train_time": [],
                         "valid": [],
                         "coreset_performance": [],
                         "rest_user_performance": [],
                         "coreset_contribution": [],
                         "rest_user_contribution": [],
                         }

        self.early_stopping = EarlyStopping(early_stop=False, ranking=True, args=self.args)
        best_valid_result = None
        for epoch in range(1, self.args.train_args.num_epochs + 1):
            # self.dataset.refresh_negatives()
            train_start = time.time()
            self.model.train()
            self.dataset.set_flag('train')
            iter_data = (
                tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    desc=set_color(
                        f"Coreset Train {epoch:>5}" if self.args.core_args.type != "Full" else f"Full Train {epoch:>5}",
                        "pink")
                ) if self.args.train_args.verbose else self.train_loader
            )

            train_loss = 0
            for b_id, batch in enumerate(iter_data):
                batch_loss = self._batch_train(batch)
                train_loss += batch_loss.item()
                # show used gpu ram
                # if self.args.train_args.verbose and self.args.train_args.device == "cuda":
                #     iter_data.set_postfix_str(
                #         set_color("GPU RAM: " + get_gpu_usage(self.args.train_args.device), "yellow")
                #     )
            train_end = time.time()
            time_consumption = train_end - train_start
            train_loss_output = generate_train_loss_output(
                epoch, "Coreset" if self.args.core_args.type != "Full" else "Full", train_start, train_end, train_loss
            )
            if self.args.train_args.verbose:
                self.logger.info(train_loss_output)

            if (epoch - 1) % self.args.eval_args.eval_every == 0:
                analysis_dict["train_loss"].append(train_loss)
                analysis_dict["train_time"].append(time_consumption)
                analysis_dict["train_epoch"].append(epoch)

                if self.args.train_args.valid:
                    valid_start = time.time()
                    if self.args.eval_args.coreset_affect:
                        valid_result = self.evaluate(flag='valid', eval_loader=self.valid_loader, tqdm_dec="Validation",
                                                     coreset_indices=self.dataset.core_uid)
                    else:
                        valid_result = self.evaluate(flag='valid', eval_loader=self.valid_loader, tqdm_dec="Validation")
                    valid_end = time.time()
                    analysis_dict["valid"].append(valid_result["full_user"])

                    if self.args.eval_args.coreset_affect:
                        analysis_dict["coreset_performance"].append(valid_result["coreset_performance"])
                        analysis_dict["rest_user_performance"].append(valid_result["rest_user_performance"])
                        analysis_dict["coreset_contribution"].append(valid_result["coreset_contribution"])
                        analysis_dict["rest_user_contribution"].append(valid_result["rest_user_contribution"])


                    valid_score_output = (
                                                 set_color("epoch %d evaluating", "green")
                                                 + " ["
                                                 + set_color("time", "blue")
                                                 + ": %.6fs, "
                                                 + set_color("valid_score", "blue")
                                                 + ": %.4f]"
                                         ) % (epoch, valid_end - valid_start, valid_result["full_user"]['ndcg@20'])
                    # valid_result_output = (
                    #         set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                    # )
                    # if verbose:
                    self.logger.info(valid_score_output)
                    # self.logger.info(valid_result_output)

                    update_flag = self.early_stopping(valid_result["full_user"]['ndcg@20'], self.model, epoch)

                    if update_flag:
                        # best_valid_result = {}
                        # for key in valid_result["full_user"].keys():
                        #     best_valid_result[key] = valid_result["full_user"][key]
                        best_valid_result = {'hit@20': valid_result["full_user"]['hit@20'],
                                             'mrr@20': valid_result["full_user"]['mrr@20'],
                                             'ndcg@20': valid_result["full_user"]['ndcg@20'],
                                             'map@20': valid_result["full_user"]['map@20'],
                                             'recall@20': valid_result["full_user"]['recall@20'],
                                             'precision@20': valid_result["full_user"]['precision@20']}
                    if self.early_stopping.stop:
                        break

        return analysis_dict, best_valid_result

    @torch.no_grad()
    def evaluate(self, flag, eval_loader, load_best_model=False, tqdm_dec=None, coreset_indices=None):
        isTest = False #True if tqdm_dec == "Test" else False
        self.dataset.set_flag(flag)

        if load_best_model:
            # path = os.path.join(self.args.model_save_path, self.args.model_saves_name)
            loaded_dict = torch.load(self.early_stopping.save_path)
            self.model.load_state_dict(loaded_dict)
            load_msg = set_color("Load best parameters.", "yellow")
            self.logger.info(load_msg)
            if os.path.exists(self.early_stopping.save_path):
                os.remove(self.early_stopping.save_path)

        self.model.eval()
        # if self.model.core_train:
        self.model.build_user_embeddings()
        eval_loader = eval_loader
        self.collector = Collector(self.dataset.item_num, self.args.eval_args.top_k)
        iter_data = (
            tqdm(
                eval_loader,
                total=len(eval_loader),
                desc=set_color(f"{tqdm_dec  }", "pink")
            ) if self.args.eval_args.verbose else eval_loader
        )

        for bid, batch in enumerate(iter_data):
            positive_u, items, scores, unique_users = self._full_batch_eval(bid, batch, isTest)
            self.collector.eval_batch_collector(positive_u, items, scores, unique_users)
        self.evaluator = Evaluator(self.collector.get_topk_mat_with_len(), ['hit', 'mrr', 'ndcg', 'map', 'recall', 'precision'],
                               self.args.eval_args.top_k, coreset_indices)

        if isTest:
            self.logger.info(set_color("Saving score matrix...", "blue"))
            score_mat = torch.vstack(self.score_mat)
            score_mat = score_mat.numpy()
            file_name = self.args.core_args.type
            with open(f"/home/zhengju/projects/CoreRec/scores_embeddings/{self.args.model.architecture}/{file_name}_score.npy",'wb') as f:
                np.save(f, score_mat)
            embeddings = self.model.full_user_embeddings.cpu().numpy() if self.args.core_args.type != "Full" else self.model.user_embedding
            with open(f"/home/zhengju/projects/CoreRec/scores_embeddings/{self.args.model.architecture}/{file_name}_embd.npy",'wb') as f:
                np.save(f, embeddings)

        return self.evaluator.evaluate()

class GeneralSelectionTrainer(object):
    r"""SelectionTrainer Class is used to manage the training and evaluation processes of recommender system models
    and the coreset selection process of a specific coreset selection strategy.
    SelectionTrainer is a general class in which the training (fit) and evaluation (evaluate) logic have been implemented.
    The functioning methods _batch_train() and _full_batch_eval() should be implemented according to different training
    and evaluation strategies.
    """
    def __init__(self, model, dataset,
                 train_loader, valid_loader, test_loader, args, strategy):
        self.model = model
        self.dataset = dataset
        self.args = args
        self.logger = getLogger()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.collector = None
        self.evaluator = None
        self.strategy = strategy
        self.score_mat = []

    def _batch_train(self, iter_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def _full_batch_eval(self, bid, batch_data, isTest):
        raise NotImplementedError('Method [next] should be implemented.')

    def fit(self):

        self.early_stopping = EarlyStopping(early_stop=False, ranking=True, args=self.args)
        best_valid_result = None
        for epoch in range(1, self.args.train_args.num_epochs + 1):
            # self.dataset.refresh_negatives()
            train_start = time.time()
            self.model.train()
            self.dataset.set_flag('train')
            iter_data = (
                tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    desc=set_color(
                        f"Coreset Train {epoch:>5}" if self.args.core_args.type != "Full" else f"Full Train {epoch:>5}",
                        "pink")
                ) if self.args.train_args.verbose else self.train_loader
            )

            train_loss = 0
            for b_id, batch in enumerate(iter_data):
                batch_loss = self._batch_train(batch)
                train_loss += batch_loss.item()
                # show used gpu ram
                # if self.args.train_args.verbose and self.args.train_args.device == "cuda":
                #     iter_data.set_postfix_str(
                #         set_color("GPU RAM: " + get_gpu_usage(self.args.train_args.device), "yellow")
                #     )
                if self.strategy is not None:
                    self.strategy.after_batch()
            train_end = time.time()
            train_loss_output = generate_train_loss_output(
                epoch, "Coreset" if self.args.core_args.type != "Full" else "Full", train_start, train_end, train_loss
            )
            if self.args.train_args.verbose:
                self.logger.info(train_loss_output)
            if self.strategy is not None:
                clone_state_dict = copy.deepcopy(self.model.state_dict())
                self.strategy.after_epoch(clone_state_dict)

            if (epoch - 1) % self.args.eval_args.eval_every == 0:

                if self.args.train_args.valid:
                    valid_start = time.time()
                    valid_result = self.evaluate(flag='valid', eval_loader=self.valid_loader, tqdm_dec="Validation")
                    valid_end = time.time()
                    valid_score_output = (set_color("epoch %d evaluating", "green")
                                          + " ["
                                          + set_color("time", "blue")
                                          + ": %.6fs, "
                                          + set_color("valid_score", "blue")
                                          + ": %.4f]"
                                         ) % (epoch, valid_end - valid_start, valid_result["full_user"]['ndcg@20'])
                    self.logger.info(valid_score_output)

                    update_flag = self.early_stopping(valid_result["full_user"]['ndcg@20'], self.model, epoch)

                    if update_flag:
                        best_valid_result = {'hit@20': valid_result["full_user"]['hit@20'],
                                             'mrr@20': valid_result["full_user"]['mrr@20'],
                                             'ndcg@20': valid_result["full_user"]['ndcg@20'],
                                             'map@20': valid_result["full_user"]['map@20'],
                                             'recall@20': valid_result["full_user"]['recall@20'],
                                             'precision@20': valid_result["full_user"]['precision@20']}
                    if self.early_stopping.stop:
                        break

        selected_idx = []
        if self.strategy is not None:
            loaded_dict = torch.load(self.early_stopping.save_path)
            self.strategy.after_train(loaded_dict)
            selected_idx = self.strategy.select()

        return best_valid_result, selected_idx

    @torch.no_grad()
    def evaluate(self, flag, eval_loader, load_best_model=False, tqdm_dec=None):
        isTest = False #True if tqdm_dec == "Test" else False
        self.dataset.set_flag(flag)

        if load_best_model:
            # path = os.path.join(self.args.model_save_path, self.args.model_saves_name)
            loaded_dict = torch.load(self.early_stopping.save_path)
            self.model.load_state_dict(loaded_dict)
            load_msg = set_color("Load best parameters.", "yellow")
            self.logger.info(load_msg)
            if os.path.exists(self.early_stopping.save_path):
                os.remove(self.early_stopping.save_path)

        self.model.eval()
        eval_loader = eval_loader
        self.collector = Collector(self.dataset.item_num, self.args.eval_args.top_k)
        iter_data = (
            tqdm(
                eval_loader,
                total=len(eval_loader),
                desc=set_color(f"{tqdm_dec  }", "pink")
            ) if self.args.eval_args.verbose else eval_loader
        )

        for bid, batch in enumerate(iter_data):
            positive_u, items, scores, unique_users = self._full_batch_eval(bid, batch, isTest)
            self.collector.eval_batch_collector(positive_u, items, scores, unique_users)
        self.evaluator = Evaluator(self.collector.get_topk_mat_with_len(), ['hit', 'mrr', 'ndcg', 'map', 'recall', 'precision'],
                               self.args.eval_args.top_k, coreset_indices=None)

        if isTest:
            self.logger.info(set_color("Saving score matrix...", "blue"))
            score_mat = torch.vstack(self.score_mat)
            score_mat = score_mat.numpy()
            file_name = self.args.dss_args.method
            with open(f"/home/zhengju/projects/CoreRec/scores_embeddings/{self.args.model.architecture}/{file_name}_score.npy",'wb') as f:
                np.save(f, score_mat)
            embeddings = self.model.user_embedding
            with open(f"/home/zhengju/projects/CoreRec/scores_embeddings/{self.args.model.architecture}/{file_name}_embd.npy",'wb') as f:
                np.save(f, embeddings)

        return self.evaluator.evaluate()


