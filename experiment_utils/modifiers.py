from collections import defaultdict
from email.policy import default

from omegaconf import OmegaConf


valid_vis_weights = defaultdict(
    lambda: {
        'imagenet': {'style': 'IMAGENET1K_V2'},
        'none': {'style': None},
    }, {
    'resnet50': {
        'imagenet': {'style': 'IMAGENET1K_V2'},
        'none': {'style': None},
        'places365': {
            'style': 'basic',
            'path': 'outside/encoders/places365/resnet50_places365.pth.tar',
        },
        'dead_leaves-mixed': {
            'style': 'moco',
            'path': 'outside/encoders/dead_leaves-mixed/checkpoint_0199.pth.tar',
        },
        'feature_vis-dead_leaves': {
            'style': 'moco',
            'path': 'outside/encoders/feature_vis-dead_leaves/checkpoint_0199.pth.tar',
        },
        'mixed-4': {
            'style': 'moco',
            'path': 'outside/encoders/mixed-4/checkpoint_0799.pth.tar',
        },
        'stat-spectrum_color_wmm': {
            'style': 'moco',
            'path': 'outside/encoders/stat-spectrum_color_wmm/checkpoint_0199.pth.tar',
        },
        'stylegan-oriented': {
            'style': 'moco',
            'path': 'outside/encoders/stylegan-oriented/checkpoint_0199.pth.tar',
        },
        'shaders21k': {
            'style': 'moco',
            'path': 'outside/encoders/shaders21k/checkpoint_0199.pth.tar',
        }
    },
})

vis_weight_loaders = {
    'moco': 'mmnoise.models.model_utils.load_from_moco_pretrained',
    'basic': 'mmnoise.models.model_utils.load_from_basic_checkpoint',
}

valid_text_weights = defaultdict(
    lambda: {
        'pretrained': {'style': True},
        'none': {'style': False},
    }, {

    }
)


def remove(conf, path):
    if '.' in path:
        p, x = path.rsplit('.', 1)
        p = OmegaConf.select(conf, p)
        if p and x in p:
            delattr(p, x)
    else:
        if path in conf:
            delattr(conf, path)


def set_model(
    config,
    vision_name='resnet50',
    vision_weights='imagenet',
    text_name='bert-base-uncased',
    text_weights='pretrained',
    freeze_vision=False,
    freeze_text=False,
    ft_lr_scale=0.01,
):
    # configure the vision encoder
    if vision_weights not in valid_vis_weights[vision_name]:
        raise RuntimeError(f'vision_weights {vision_weights} is unknown')
    config.model.image_encoder.name = vision_name
    weight_style = valid_vis_weights[vision_name][vision_weights]['style']
    if vision_weights in ('imagenet', 'none'):
        config.model.image_encoder.create_kw = {
            'weights': weight_style,
        }
        remove(config, 'model.image_encoder.load_func')
        remove(config, 'model.image_encoder.load_kw')
    else:
        config.model.image_encoder.load_func = vis_weight_loaders[weight_style]
        config.model.image_encoder.load_kw = {
            'weights_path': valid_vis_weights[vision_name][vision_weights]['path']
        }
        remove(config, 'model.image_encoder.create_kw.weights')

    # configure the text encoder
    assert text_weights in valid_text_weights[text_name]   
    config.model.text_encoder.name = text_name
    weights_style = valid_text_weights[text_name][text_weights]['style']
    if text_weights in ('pretrained', 'none'):
        config.model.text_encoder.create_kw = {
            'pretrained': weights_style
        }
    else:
        raise NotImplementedError()

    # configure parameter groups
    groups = [{
            'scale': 1.0,
            'params': [
                'image_projection',
                'text_projection',
                'temperature',
            ]
        }, {
            'scale': ft_lr_scale,
            'params': []
        }, {
            'scale': 0.0,
            'params': []
        }
    ]
    if freeze_vision:
        groups[2]['params'].append('image_encoder')
    elif vision_weights == 'none':
        groups[0]['params'].append('image_encoder')
    else:
        groups[1]['params'].append('image_encoder')

    if freeze_text:
        groups[2]['params'].append('text_encoder')
    elif text_weights == 'none':
        groups[0]['params'].append('text_encoder')
    else:
        groups[1]['params'].append('text_encoder')

    for i in range(len(groups) - 1, -1, -1):
        if len(groups[i]['params']) == 0:
            groups.pop(i)
    
    config.model.lr_scale = groups
