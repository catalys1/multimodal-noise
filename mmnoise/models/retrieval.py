from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch
import torchvision
import transformers

from mmnoise.utils import losses, distributed
from mmnoise.models.components import TemperatureScale


class RetrievalModule(pl.LightningModule):
    def __init__(
        self,
        image_encoder: Union[str, torch.nn.Module] = 'resnet18',
        text_encoder: Union[str, torch.nn.Module] = 'bert-base-uncased',
        tokenizer: Optional[Union[str, Callable]] = None,
        embed_dim: int = 256,
        criterion: Union[str, torch.nn.Module] = 'CLIPLoss',
        lr: float = 0.01,
        opt_args: Optional[dict] = None,
        lrsched_args: Optional[dict] = None,
    ):
        super().__init__()

        if tokenizer is None:
            if not isinstance(text_encoder, str):
                raise RuntimeError(
                    f'The tokenizer is None but text_encoder is not a string, no way to create a Tokenizer')
            tokenizer = transformers.AutoTokenizer.from_pretrained(text_encoder)
        self.tokenizer = tokenizer

        if isinstance(image_encoder, str):
           image_encoder = getattr(torchvision.models, image_encoder)()
        image_encoder.fc = torch.nn.Identity()
        self.image_encoder = image_encoder

        if isinstance(text_encoder, str):
            t_config = transformers.AutoConfig.for_model(text_encoder.split('-')[0])
            text_encoder = transformers.AutoModel.from_config(t_config)
        self.text_encoder = text_encoder

        # Need to add projection layers for text and images
        self.image_projection = torch.nn.Linear(512, embed_dim) # hardcoded for now, need to change
        self.text_projection = torch.nn.Linear(text_encoder.config.hidden_size, embed_dim)

        self.temperature = TemperatureScale(0.07)  # learnable temperature

        if isinstance(criterion, str):
            criterion = getattr(losses, criterion)()
        self.criterion = criterion

        self.lr = lr
        self.opt_args = opt_args
        self.lrsched_args = lrsched_args

    def forward(self, imgs, text):
        imgs_feat = self.image_encoder(imgs)

        text = self.tokenizer(text, padding='longest', truncation=True, max_length=30)
        text_in = torch.as_tensor(text.input_ids, device=self.device)
        att_mask = torch.as_tensor(text.attention_mask, device=self.device)
        text_out = self.text_encoder(text_in, attention_mask=att_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]  # only take the first token ([CLS])

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
        return {'loss': loss}

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

        relevant = targets.gather(1, preds.topk(5, dim=1, largest=True).indices).sum(dim=1)
        count = targets.sum(1)
        count[count == 0] = 1.0
        recall = torch.mean(relevant / count)
        self.log('val/recall@5', recall, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        # instantiate the optimizer and learning rate schedule
        # need to handle multiple param groups, for weight decay and for optionally freezing certain layers
        if self.opt_args is not None:
            self.opt_args['init_args']['lr'] = self.lr
            optimizer = pl.cli.instantiate_class(self.parameters(), self.opt_args)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=True)

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
