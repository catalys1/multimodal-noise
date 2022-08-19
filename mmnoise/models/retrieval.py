from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
import torchvision
import transformers

from mmnoise.utils import losses, distributed
from mmnoise.models import model_utils
from mmnoise.models.components import TemperatureScale


def default_config(name, vision=True):
    config = dict(name=name)
    if vision:
        config['create_func'] = model_utils.vision_model_no_fc
    else:
        config['create_func'] = model_utils.huggingface_model
    return config


def get_model(model_data, vision=True):
    if isinstance(model_data, torch.nn.Module):
        return model_data
    elif isinstance(model_data, str):
        config = default_config(model_data, vision)
    elif isinstance(model_data, dict):
        config = model_data
    return model_utils.create_model_from_config(config)
    

class RetrievalModule(pl.LightningModule):
    '''
    '''
    def __init__(
        self,
        image_encoder: Union[dict, str, torch.nn.Module] = 'resnet18',
        text_encoder: Union[dict, str, torch.nn.Module] = 'bert-base-uncased',
        tokenizer: Union[dict, str, Callable] = 'bert-base-uncased',
        embed_dim: int = 256,
        criterion: Union[str, torch.nn.Module] = 'CLIPLoss',
        lr: float = 0.01,
        lr_scale: Optional[list] = None,
        opt_args: Optional[dict] = None,
        lrsched_args: Optional[List] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.image_encoder = get_model(image_encoder, vision=True)
        self.text_encoder = get_model(text_encoder, vision=False)

        # TODO: still need to modularize tokenizer creation
        if isinstance(tokenizer, str):
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer

        # Pojection layers for text and images
        img_dim = list(self.image_encoder.parameters())[-1].shape[-1] # probably doesn't work in all cases
        text_dim = self.text_encoder.config.hidden_size
        self.image_projection = torch.nn.Linear(img_dim, embed_dim)
        self.text_projection = torch.nn.Linear(text_dim, embed_dim)

        self.temperature = TemperatureScale(0.07)  # learnable temperature

        if isinstance(criterion, str):
            criterion = getattr(losses, criterion)()
        self.criterion = criterion

        self.lr = lr
        self.lr_scale = lr_scale or {}
        self.opt_args = opt_args
        self.lrsched_args = lrsched_args

    def forward(self, imgs, text):
        imgs_feat = self.image_encoder(imgs)

        text = self.tokenizer(text, padding='longest', truncation=True, max_length=30)
        text_in = torch.as_tensor(text.input_ids, device=self.device)
        att_mask = torch.as_tensor(text.attention_mask, device=self.device)
        text_out = self.text_encoder(text_in, attention_mask=att_mask)
        # only take the first token ([CLS]). Not sure if this is the best approach.
        text_feat = text_out.last_hidden_state[:, 0, :]

        return imgs_feat, text_feat

    def _step(self, imgs_feat, text_feat):
        imgs_proj = self.image_projection(imgs_feat)
        imgs_proj = torch.nn.functional.normalize(imgs_proj, dim=-1)
        text_proj = self.text_projection(text_feat)
        text_proj = torch.nn.functional.normalize(text_proj, dim=-1)
        return imgs_proj, text_proj

    def training_step(self, batch, *args, **kwargs):
        imgs, text = batch['images'], batch['text']
        imgs_feat, text_feat = self(imgs, text)
        imgs_proj, text_proj = self._step(imgs_feat, text_feat)

        loss = self.criterion(imgs_proj, text_proj, self.temperature)

        self.log('train/loss', loss, on_step=True)
        self.log('temperature', self.temperature.temp, on_step=True)

        with torch.no_grad():
            logits = imgs_proj @ text_proj.T
            gap = 0.5 * (logits.max(1)[0].sub(logits.mean(1)).mean() + logits.max(0)[0].sub(logits.mean(0)).mean())
            self.log('logit_gap', gap, on_step=True)

        return {'loss': loss}

    def on_after_backward(self):
        img_grad = self.image_projection.weight.grad.abs().mean()
        text_grad = self.text_projection.weight.grad.abs().mean()
        self.log('grads/img-proj-w', img_grad, on_step=True)
        self.log('grads/text-proj-w', text_grad, on_step=True)

    def validation_step(self, batch, *args, **kwargs):
        imgs, text = batch['images'], batch['text']
        idx = batch['index']
        imgs_feat, text_feat = self(imgs, text)
        imgs_proj, text_proj = self._step(imgs_feat, text_feat)

        return {'image_features': imgs_proj, 'text_features': text_proj, 'index': idx}

    def validation_epoch_end(self, outputs):
        img_feat = [x['image_features'] for x in outputs]
        text_feat = [x['text_features'] for x in outputs]
        idx = [x['index'] for x in outputs]
        cumu = 0
        for s, ii in zip([x.shape[0] for x in img_feat[:-1]], idx[1:]):
            cumu += s
            ii += cumu
        img_feat = torch.cat(img_feat, 0)
        text_feat = torch.cat(text_feat, 0)
        idx = torch.cat(idx, 0)
        if distributed.is_distributed():
            img_feat = self.all_gather(img_feat).flatten(0, 1)
            text_feat = self.all_gather(text_feat).flatten(0, 1)
            idx += distributed.rank() * idx.shape[0]
            idx = self.all_gather(idx).flatten(0, 1)

        preds = img_feat @ text_feat.T
        targets = torch.arange(img_feat.shape[0], device=self.device)[:,None].eq(idx)

        ks = [1,5,10]
        recall_i2t = compute_recall(preds, targets, k=ks)
        recall_t2i = compute_recall(preds.T, targets.T, k=ks)
        for k, vi2t, vt2i in zip(ks, recall_i2t, recall_t2i):
            self.log(f'val/recall@{k}-i2t', vi2t, on_epoch=True, on_step=False, prog_bar=False)
            self.log(f'val/recall@{k}-t2i', vt2i, on_epoch=True, on_step=False, prog_bar=False)

        recall = 0.5 * (recall_i2t[1] + recall_t2i[1])
        self.log('val/recall@5', recall, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        # instantiate the optimizer and learning rate schedule
        # self.lr_scales contains a list of dicts, each with a scale and a list of parameter name
        # prefixes, which allows for applying different learning rates to different groups of
        # parameters
        param_groups = []
        named_params = dict(filter(lambda x: x[1].requires_grad, self.named_parameters()))
        for group in self.lr_scale:
            params = []
            keys = [x for x in group['params']]
            for kk in list(named_params.keys()):
                for k in keys:
                    if kk.startswith(k):
                        params.append(named_params.pop(kk))
            if group['scale'] > 0:
                pg = {'params': params, 'lr': self.lr * group['scale']}
                param_groups.append(pg)
            else:
                for p in params: p.requires_grad_(False)

        # default param group for any remaining parameters
        param_groups.append({'params': list(named_params.values()), 'lr': self.lr})

        # create a specified or default optimizer
        if self.opt_args is not None:
            self.opt_args['init_args']['lr'] = self.lr
            optimizer = pl.cli.instantiate_class(param_groups, self.opt_args)
        else:
            optimizer = torch.optim.SGD(param_groups, lr=self.lr, momentum=0.9)

        # create a specified or default learning rate scheduler
        if self.lrsched_args is not None:
            lr_scheduler = pl.cli.instantiate_class(optimizer, self.lrsched_args)
        else:
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr = [g['lr'] for g in optimizer.param_groups],
                    total_steps = self.trainer.estimated_stepping_batches,
                    pct_start = 0.1,
                ),
                'interval': 'step',
            }

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }


def compute_recall(preds, targets, k):
    if isinstance(k, int):
        k = [k]
    maxk = max(k)
    ranked = preds.topk(maxk, dim=1, largest=True, sorted=True).indices
    count = targets.sum(1)
    count[count == 0] = 1.0
    recalls = []
    for kk in k:
        relevant = targets.gather(1, ranked[:, :kk]).sum(dim=1)
        recall = torch.mean(relevant / count)
        recalls.append(recall)
    if len(recalls) == 1:
        recalls = recalls[0]
    return recalls
