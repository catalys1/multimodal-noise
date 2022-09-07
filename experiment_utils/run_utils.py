from pathlib import Path
import re

import numpy as np
from omegaconf import OmegaConf

from .config_utils import pselect
from .wandb_utils import *


__all__ = [
    'RunData',
]


class RunData(object):
    def __init__(self, root, wandb_root):
        self.root = Path(root)
        self.wandb_root = Path(wandb_root)
        
        self.wandb_id = self.root.joinpath('wandb_id').open().read().strip()

        self.config = OmegaConf.load(self.root.joinpath('raw_run_config.yaml'))
        
        self._metrics = None

    @property
    def name(self):
        return self.root.name

    @property
    def run_id(self):
        return int(self.name.split('-')[1])
    
    @property
    def metrics(self):
        if self._metrics is None:
            wpath = get_wandb_run_from_id(list(self.wandb_root.glob('*run-*')), self.wandb_id)
            wpath = wpath.joinpath(f'run-{self.wandb_id}.wandb')
            self._metrics = extract_wandb_metrics(wpath)
        return self._metrics

    @property
    def val_metrics(self):
        def _key_sort(k):
            n = int(re.search('\d+', k)[0])
            return (n, 0 if 'i2t' in k else 1)

        val_keys = sorted([k for k in self.metrics[0] if k.startswith('val/')], key=_key_sort)
        return [
            {k: x[k] for k in val_keys} for x in self.metrics
            if all(not np.isnan(x[k]) for k in val_keys)
        ]

    @property
    def max_val_metrics(self):
        metrics = self.val_metrics
        return {k: max(x[k] for x in metrics) for k in metrics[0]}

    def confv(self, val):
        return pselect(self.config, val)

    def print_conf(self, key=None):
        conf = self.config
        if key is not None:
            conf = OmegaConf.select(conf, key)
        print(OmegaConf.to_yaml(conf))

    def __repr__(self):
        s = f'<Run ({self.name})>'
        return s

    def __str__(self):
        return self.__repr__()