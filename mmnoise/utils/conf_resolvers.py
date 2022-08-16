'''Custom resolvers that can be used in configuration files through OmegaConf.
'''
import os
import re

from omegaconf import OmegaConf


def register(cache=False):
    def decorator_register(func):
        OmegaConf.register_new_resolver(func.__name__, func, use_cache=cache)
        return func
    return decorator_register


@register(cache=True)
def next_run(root, wandb=True):
    '''Determine the name of the next run and setup the run folder.
    '''
    run = 0
    if os.path.exists(root):
        runs = [x for x in os.listdir(root) if re.match(r'run-\d+', x)]
        if len(runs) > 0:
            runs = sorted(int(x.split('-')[-1]) for x in runs)
            run = runs[-1] + 1
    path = os.path.join(root, f'run-{run}')
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