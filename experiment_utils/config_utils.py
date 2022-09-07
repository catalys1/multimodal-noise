import omegaconf
from omegaconf import OmegaConf


__all__ = [
    'pselect',
    'remove',
]


def pselect(config, key):
    '''Recursively search for partial matches to `key` in `config`.
    '''
    matches = []
    path = []

    def recurse(conf):
        if isinstance(conf, (dict, omegaconf.DictConfig)):
            _iter = conf
        else:
            _iter = range(len(conf))
        for k in _iter:
            path.append(str(k))
            if key in '.'.join(path):
                matches.append(('.'.join(path), conf[k]))
            elif OmegaConf.is_interpolation(conf, k):
                pass
            elif isinstance(conf[k], (dict, omegaconf.DictConfig, list, omegaconf.ListConfig)):
                recurse(conf[k])
            path.pop()
    
    recurse(config)
    return matches


def remove(conf, path):
    if '.' in path:
        p, x = path.rsplit('.', 1)
        p = OmegaConf.select(conf, p)
        if p and x in p:
            delattr(p, x)
    else:
        if path in conf:
            delattr(conf, path)