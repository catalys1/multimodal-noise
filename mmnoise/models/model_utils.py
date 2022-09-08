from functools import partial
from importlib import import_module
import os

import torch
import transformers

from . import defs


def identity(input, *args, **kwargs):
    return input


def noop(*args, **kwargs):
    pass


# Adapted from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
def init_hf_weights(module, mean=0.0, std=0.02):
    """Initialize the weights"""
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=mean, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=mean, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


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
    model = vision_model(name, **kwargs)
    setattr(model, fc_key, torch.nn.Identity())
    return model


def huggingface_model(name, *, pretrained=False, **kwargs):
    if pretrained:
        if 'pretrained_model_name_or_path' in kwargs:
            name = kwargs.pop('pretrained_model_name_or_path')
        model = transformers.AutoModel.from_pretrained(name, **kwargs)
    else:
        n = name.replace('-', '_')
        if hasattr(defs, n):
            temp = kwargs
            kwargs = getattr(defs, n)()
            kwargs.update(temp)
        config = transformers.AutoConfig.for_model(name.split('-')[0], **kwargs)
        model = transformers.AutoModel.from_config(config)
    return model


def huggingface_model_replace_embeddings(name, *, pretrained=False, config_or_name=None, **kwargs):
    model = huggingface_model(name, pretrained=pretrained, **kwargs)

    if config_or_name is not None:
        if isinstance(config_or_name, str):
            config = transformers.AutoConfig.from_pretrained(config_or_name)
        else:
            config = config_or_name
        model.embeddings = model.embeddings.__class__(config)
    model.embeddings.apply(partial(init_hf_weights, mean=0.0, std=model.config.initializer_range))

    return model


def bert_tokenizer_for_code(name, **kwargs):
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    tokenizer.add_tokens(["[/n]"])
    tokenizer.add_tokens(["[/t]"])
    return tokenizer


def mlp_model(name, in_dim, layer_dims):
    from mmnoise.models.components.mlp import MLP
    model = MLP(in_dim, layer_dims)
    return model


def load_from_basic_checkpoint(net, weights_path, prefix='module.'):
    state = torch.load(weights_path, map_location='cpu')
    state_key = 'state_dict' if 'state_dict' in state else 'model'
    state = state[state_key]
    for k in list(state.keys()):
        if k.startswith(prefix):
            state[k[len(prefix):]] = state[k]
            del state[k]
    keys = net.load_state_dict(state, strict=False)
    if len(keys.unexpected_keys) > 2 or len(keys.missing_keys) > 2:
        raise RuntimeError(f'Error loading model weights:\n{keys}')


def load_from_moco_pretrained(net, weights_path):
    state = torch.load(weights_path, map_location='cpu')['state_dict']
    for k in list(state.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state[k[len("module.encoder_q."):]] = state[k]
        # delete renamed or unused k
        del state[k]
    keys = net.load_state_dict(state, strict=False)
    if len(keys.unexpected_keys) > 2 or len(keys.missing_keys) > 2:
        raise RuntimeError(f'Error loading model weights:\n{keys}')
    

def get_output_dim(model):
    from torchvision import models
    from mmnoise.models.components import mlp
    if isinstance(model, models.ResNet):
        dim = list(model.parameters())[-1].shape[-1] # probably doesn't work in all cases
    elif isinstance(model, transformers.PreTrainedModel):
        dim = model.config.hidden_size
    elif isinstance(model, mlp.MLP):
        dim = model.layers[-4].out_features
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    return dim
