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
        for k in conf:
            path.append(k)
            if key in k:
                matches.append(('.'.join(path), conf[k]))
            elif isinstance(conf[k], (dict, omegaconf.dictconfig.DictConfig)):
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