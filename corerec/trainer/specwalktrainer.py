from corerec.trainer.trainer import GeneralTrainer, GeneralSelectionTrainer

class SpecWalkSelectionTrainer(GeneralSelectionTrainer):
    def __init__(self, model, dataset,
                 train_loader, valid_loader, test_loader,
                 loss_fn, optimizer, scheduler, args, strategy):
        super(SpecWalkSelectionTrainer, self).__init__(model, dataset,
                                                  train_loader, valid_loader, test_loader,
                                                  args, strategy)
        self.model = model

    def fit(self):
        self.model.fit(epochs=self.args.dss_args.num_epoch)
        selected_idx = []
        if self.strategy is not None:
            self.strategy.after_train(self.model.state_dict())
            selected_idx = self.strategy.select()
        return None, selected_idx
