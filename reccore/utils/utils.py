import os
import random
import datetime
import torch
import numpy as np
from torch.utils.data import Sampler
from argparse import ArgumentTypeError
from typing import Iterator, Sequence
import collections.abc



def update_dict(d, u):
    r"""Update dict. Dict with nested dicts can also be updated.

    Args:
        d: dict to update
        u: second dict

    Returns:
        d: the updated dict
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def generate_train_loss_output(epoch_idx, train_type, s_time, e_time, losses):
    des = 4
    train_loss_output = (
        set_color("epoch %d %s training", "green")
        + " ["
        + set_color("time", "blue")
        + ": %.6fs, "
    ) % (epoch_idx, train_type, e_time - s_time)
    if isinstance(losses, tuple):
        des = set_color("train_loss%d", "blue") + ": %." + str(des) + "f"
        train_loss_output += ", ".join(
            des % (idx + 1, loss) for idx, loss in enumerate(losses)
        )
    else:
        des = "%." + str(des) + "f"
        train_loss_output += set_color("train loss", "blue") + ": " + des % losses
    return train_loss_output + "]"


class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]#.item()

    def __len__(self) -> int:
        return len(self.indices)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def list_batch_collate(batch: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    user_ids, user_indices, items = [], [], []
    for data in batch:
        user_ids.append(data["userId"])
        user_indices.extend(data["indices"][0])
        items.extend(data["indices"][1])
    return {"userId": torch.tensor(user_ids),
            "indices": (torch.tensor(user_indices), torch.tensor(items))}


