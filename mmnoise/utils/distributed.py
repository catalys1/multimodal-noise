import torch


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def rank():
    if is_distributed():
        return torch.distributed.get_rank()
    return 0
