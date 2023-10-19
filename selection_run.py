from logging import getLogger
from corerec.utils import selection_parser
from corerec.utils.logger import init_logger
from corerec.data.data_utils import gen_dataset
from corerec.utils.utils import set_color, setup_seed
from corerec.utils.initializer import create_dataloader, create_model, create_strategy,\
    create_selection_trainer, loss_function, optimizer_with_scheduler

if __name__ == '__main__':
    config = selection_parser.cfg
    setup_seed(config.dss_args.seed)
    init_logger(config)
    logger = getLogger()

    dataset = gen_dataset(config)
    logger.info(
        set_color("Preprocessed dataset", "blue") + f": {config.dataset.name}" +
        set_color(" Number of Users", "blue") + f": {dataset.user_num}" +
        set_color(" Number of Items", "blue") + f": {dataset.item_num}"
    )

    train_loader, val_loader, test_loader = create_dataloader(dataset, config)
    logger.info(
        set_color("Generate general dataloaders", "blue") + ": Train, Valid and Test"
    )

    model = create_model(dataset, config)
    logger.info(
        set_color("Initialize model", "blue") + f": {config.model.architecture}"
    )

    loss_func = loss_function(config)
    optimizer, scheduler = optimizer_with_scheduler(model, config)
    strategy = create_strategy(dataset, model, config)
    trainer = create_selection_trainer(model, dataset,
                             train_loader, val_loader, test_loader,
                             loss_func, optimizer, scheduler, config, strategy)
    logger.info(set_color("method: ", "red") + f"{config.dss_args.method}"+"\t"+
                 set_color("model: ", "red") + f"{config.model.architecture}" + "\t"+
                 set_color("coreset size: ", "red") + f"{config.dss_args.coreset_size}" + "\t"+
                 set_color("dataset: ", "red") + f"{config.dataset.name}" + "\t"+
                 set_color("seed: ", "red") + f"{config.dss_args.seed}")
    best_valid_result, selected_idx = trainer.fit()
    test_result_all = trainer.evaluate(flag='test', eval_loader=test_loader, load_best_model=True, tqdm_dec="Test")
    test_result = {'hit@20': test_result_all["full_user"]['hit@20'],
                   'mrr@20': test_result_all["full_user"]['mrr@20'],
                   'ndcg@20': test_result_all["full_user"]['ndcg@20'],
                   'map@20': test_result_all["full_user"]['map@20'],
                   'recall@20': test_result_all["full_user"]['recall@20'],
                   'precision@20': test_result_all["full_user"]['precision@20']}
    logger.info(set_color("Best valid result", "yellow") + f": {best_valid_result}")
    logger.info(set_color("Test result", "yellow") + f": {test_result}")

    print(len(selected_idx))
    print(selected_idx)


