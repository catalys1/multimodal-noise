from importlib import import_module


def identity(input, *args, **kwargs):
    return input


def noop(*args, **kwargs):
    pass


def func_from_str(func_path):
    if not isinstance(func_path, str):
        return func_path
    mod, name = func_path.rsplit('.', 1)
    func = getattr(import_module(mod), name)
    return func


def create_model_from_config(config):
    '''Create and return a model given a config. The config is a dict with the following keys:
        "name": an str, the name of the model
        "create_func": a function that returns a model given a name
        "create_kw": dict of additional keyword args for create_func. Can be omitted or None.
        "load_func": a function that loads pre-trained weights into the model returned by create_func.
            Can be omitted or None.
        "load_kw": dict of additional keyword args for load_func. Can be omitted or None.
    '''
    name = config['name']
    create_func = func_from_str(config['create_func'])
    create_kw = config.get('create_kw', None) or {}
    load_func = func_from_str(config.get('load_func', noop))
    load_kw = config.get('load_kw', None) or {}

    model = create_func(name, **create_kw)
    load_func(model, **load_kw)

    return model


def vision_model(name, **kwargs):
    from torchvision import models
    model_class = getattr(models, name)
    model = model_class(**kwargs)
    return model


def vision_model_no_fc(name, *, fc_key='fc', **kwargs):
    from torch.nn import Identity
    model = vision_model(name, **kwargs)
    setattr(model, fc_key, Identity())
    return model


def huggingface_model(name, *, pretrained=False, **kwargs):
    import transformers
    if pretrained:
        model = transformers.AutoModel.from_pretrained(name, **kwargs)
    else:
        config = transformers.AutoConfig.for_model(name.split('-')[0], **kwargs)
        model = transformers.AutoModel.from_config(config)
    return model


def load_from_moco_pretrained(net, weights_path):
    import torch
    state = torch.load(weights_path, map_location='cpu')['state_dict']
    for k in list(state.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state[k[len("module.encoder_q."):]] = state[k]
        # delete renamed or unused k
        del state[k]
    keys = net.load_state_dict(state, strict=False)
    assert len(keys.unexpected_keys) == 0
    assert len(keys.missing_keys) <= 2
    