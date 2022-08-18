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
        'dead_leaves-mixed': {
            'style': 'coco',
            'path': 'outside/encoders/dead_leaves-mixed/checkpoint_0199.pth.tar',
        },
        'feature_vis-dead_leaves': {
            'style': 'coco',
            'path': 'outside/encoders/feature_vis-dead_leaves/checkpoint_0199.pth.tar',
        },
        'mixed-4': {
            'style': 'coco',
            'path': 'outside/encoders/mixed-4/checkpoint_0799.pth.tar',
        },
        'stat-spectrum_color_wmm': {
            'style': 'coco',
            'path': 'outside/encoders/stat-spectrum_color_wmm/checkpoint_0199.pth.tar',
        },
        'stylegan-oriented': {
            'style': 'coco',
            'path': 'outside/encoders/stylegan-oriented/checkpoint_0199.pth.tar',
        },
    },
})

vis_weight_loaders = {
    'coco': 'mmnoise.models.model_utils.load_from_coco_pretrained',
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
):
    # configure the vision encoder
    assert vision_weights in valid_vis_weights[vision_name]
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
    groups = [
        {
            'scale': 1.0,
            'params': [
                'image_projection',
                'text_projection',
                'temperature',
            ]
        },
        {
            'scale': 0.01,
            'params': []
        }
    ]
    if vision_weights == 'none':
        groups[0]['params'].append('image_encoder')
    else:
        groups[1]['params'].append('image_encoder')

    if text_weights == 'none':
        groups[0]['params'].append('text_encoder')
    else:
        groups[1]['params'].append('text_encoder')

    if len(groups[1]['params']) == 0:
        groups.pop(1)
    
    config.model.lr_scale = groups
