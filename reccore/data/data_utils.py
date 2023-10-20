from reccore.data.dataset import *

# from datasets import load_dataset

def gen_dataset(config):
    """
    Generate train, val, and test datasets for supervised learning setting.

    Parameters
    """
    model_name = config.model.architecture

    if model_name == 'MultiVAE':
        dataset = DataMultiVAE(config.dataset)
        return dataset
    if model_name == 'BPR':
        dataset = DataBPR(config.dataset)
        return dataset
    if model_name == 'LightGCN':
        dataset = DataLightGCN(config.dataset)
        return dataset
    if model_name == 'ItemKNN' or 'EASE':
        dataset = DataItemKNN(config.dataset)
        return dataset

    else:
        raise NotImplementedError