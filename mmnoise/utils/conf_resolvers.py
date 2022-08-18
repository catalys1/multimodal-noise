'''Custom resolvers that can be used in configuration files through OmegaConf.
'''
import os

from omegaconf import OmegaConf

from . import next_run_path


def register(cache=False):
    def decorator_register(func):
        OmegaConf.register_new_resolver(func.__name__, func, use_cache=cache)
        return func
    return decorator_register


@register(cache=True)
def next_run(root, wandb=True):
    '''Determine the name of the next run and setup the run folder.
    '''
    path = next_run_path(root)
    os.makedirs(path)
    if wandb:
        os.makedirs(os.path.join(path, 'wandb'))
    return path


@register(cache=False)
def path_seg(path, seg_idx=-1):
    '''Given a path made up of segments separated by "/", return the segment at seg_idx.
    '''
    segments = str(path).split('/')
    return segments[seg_idx]


@register(cache=False)
def linear_scale_lr(lr, bs, base_bs):
    '''Compute a linearly scaled learning rate based on the ratio of the batch size to
    a base batch size.
    '''
    return lr * bs / base_bs