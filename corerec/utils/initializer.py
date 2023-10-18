import inspect
import sys
import torch
import torch.optim as optim
from corerec.utils.utils import list_batch_collate
import corerec.model
import corerec.utils.loss
import corerec.trainer

def create_dataloader(dataset, config):
    trn_batch_size = config.dataloader.batch_size
    val_batch_size = config.dataloader.batch_size
    tst_batch_size = config.dataloader.batch_size
    dataset.set_flag('train')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=trn_batch_size, shuffle=True)

    dataset.set_flag('valid')
    valloader = torch.utils.data.DataLoader(dataset, batch_size=val_batch_size,
                                            shuffle=False, collate_fn=list_batch_collate)

    dataset.set_flag('test')
    testloader = torch.utils.data.DataLoader(dataset, batch_size=tst_batch_size,
                                             shuffle=False, collate_fn=list_batch_collate)
    return trainloader, valloader, testloader


def create_model(dataset, config):
    r"""Loop through all the models implemented in the model package.

    Args:
        dataset: The preprocessed dataset
        config: Configration settings

    Returns:
        model: The desired model specified in config. The model is transported to the corresponding device.

    """

    model_dict = {}
    model_class = inspect.getmembers(
        sys.modules['corerec.model'], lambda x: inspect.isclass(x)
    )
    for model_name, model in model_class:
        model_dict[model_name] = model

    if config.model.architecture not in model_dict.keys():
        raise NotImplementedError("The specified model is not implemented.")

    model = model_dict[config.model.architecture](dataset=dataset, config=config)
    model = model.to(config.train_args.device)
    return model


def loss_function(config):
    r"""Loop through all the loss functions implemented in the loss.py file.

    Args:
        config: Configration settings.

    Returns:
        criterion: The desired loss function specified in config.

    """
    loss_dict = {}
    loss_class = inspect.getmembers(
        sys.modules['corerec.utils.loss'], lambda x: inspect.isclass(x) and x.__module__ == 'corerec.utils.loss'
    )
    for loss_name, loss in loss_class:
        loss_dict[loss_name] = loss

    if config.loss.type not in loss_dict.keys():
        raise NotImplementedError("The specified loss function is not implemented.")

    criterion = loss_dict[config.loss.type](config=config)

    return criterion


def optimizer_with_scheduler(model, config):
    optimizer, scheduler = None, None
    if config.optimizer.type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.optimizer.lr,
                              momentum=config.optimizer.momentum,
                              weight_decay=config.optimizer.weight_decay,
                              nesterov=config.optimizer.nesterov)
    elif config.optimizer.type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.optimizer.lr)
    elif config.optimizer.type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config.optimizer.lr)
    else:
        raise NotImplementedError("Optimizer is not specified")

    if config.scheduler.type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config.scheduler.T_max)
    elif config.scheduler.type == 'linear_decay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler.stepsize, gamma=config.scheduler.gamma)

    return optimizer, scheduler


def create_trainer(model, dataset,
                   trainloader,valloader, testloader,
                   criterion, optimizer, scheduler, config
                   ):
    r"""Loop through all the trainers implemented in the trainer package.

        Args:
            model: The RS model.

        Returns:
            trainer: The trainer corresponding to the RS model.

        """

    trainer_dict = {}
    trainer_class = inspect.getmembers(
        sys.modules['corerec.trainer'], lambda x: inspect.isclass(x)
    )
    for trainer_name, trainer in trainer_class:
        trainer_dict[trainer_name] = trainer

    desired_trainer = f"{model.__class__.__name__}Trainer"
    if desired_trainer not in trainer_dict.keys():
        raise NotImplementedError("The specified trainer is not implemented.")

    trainer = trainer_dict[desired_trainer](
        model=model,
        dataset=dataset,
        train_loader=trainloader,
        valid_loader=valloader,
        test_loader=testloader,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        args=config
    )

    return trainer
