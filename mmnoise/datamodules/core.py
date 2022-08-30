from itertools import chain
import random
from typing import Callable, Optional, Tuple

from pytorch_lightning.core.datamodule import LightningDataModule
import torch
from torch.utils.data import DataLoader, default_collate
import torchvision


__all__ = [
    'image_text_collate',
    'image_embedding_collate',
    'sample_caption',
    'BaseMultimodalDatamodule',
]


def sample_caption(caption_list):
    '''Randomly sample a single caption from a list of captions.
    '''
    idx = random.randrange(0, len(caption_list))
    return caption_list[idx]


def image_text_collate(batch):
    '''Batch images and text, and create an index mapping each text item to an image index.
    '''
    imgs = default_collate([x[0] for x in batch])
    text = [x[1] for x in batch]
    # there could be a single str or a list of str for each image. in case of single str, convert to list
    if isinstance(text[0], str):
        text = [[x] for x in text]
    # idx associates each item in text with a corresponding item in imgs
    idx = default_collate(list(chain(*list([i]*len(x) for i, x in enumerate(text)))))
    # flatten text into a single list of str
    text = [y for x in text for y in x]
    return {'images': imgs, 'text': text, 'index': idx}


def image_embedding_collate(batch):
    '''Batch images and text embeddings, and create an index mapping each text item to an image index.
    '''
    imgs = default_collate([x[0] for x in batch])
    embed = [x[1] for x in batch]
    # there could be a single str or a list of str for each image. in case of single str, convert to list
    if len(embed[0].shape) == 1:
        embed = [x.reshape(1, -1) for x in embed]
    # idx associates each item in text with a corresponding item in imgs
    idx = default_collate(list(chain(*list([i]*len(x) for i, x in enumerate(embed)))))
    # flatten embeds
    embed = torch.cat([torch.as_tensor(x) for x in embed], 0)
    return {'images': imgs, 'text': embed, 'index': idx}


class BaseMultimodalDatamodule(LightningDataModule):

    dataset_class = None

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 224,
        normalization: Tuple[Tuple, Tuple] = ((0.485, 0.456, 0.406), (0.229, 0.225, 0.226)),
        crop_area: Tuple = (0.2, 1.0),
        hflips: bool = False,
        color_jitter: Optional[Tuple] = None,  # (brightness, contrast, saturation, hue)
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.normalization = normalization
        self.crop_area = crop_area
        self.hflips = hflips
        self.color_jitter = color_jitter

        self._post_init()

    def _post_init(self):
        '''Use for setting dataset-specific stuff'''

    def transforms(self, stage: str = 'train') -> Callable:
        T = torchvision.transforms
        tensorize = T.Compose([
            T.ToTensor(),
            T.Normalize(*self.normalization),
        ])
        if stage == 'train':
            ts = [T.RandomResizedCrop(self.image_size, self.crop_area)]
            if self.hflips:
                ts.append(T.RandomHorizontalFlip(0.5))
            if self.color_jitter is not None:
                ts.append(T.ColorJitter(*self.color_jitter))
            ts.append(tensorize)
            return T.Compose(ts)
        else:
            s = int(8 / 7 * self.image_size)
            return T.Compose([
                T.Resize((s, s)),
                T.CenterCrop(self.image_size),
                tensorize,
            ])

    def setup(self, stage: Optional[str] = None):
        '''Create and cache datasets'''
        raise NotImplementedError()

    @property
    def collate_fn(self):
        return image_text_collate

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, self.batch_size, num_workers=self.num_workers,
            pin_memory=True, drop_last=True, shuffle=True, collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, self.batch_size, num_workers=self.num_workers,
            drop_last=False, pin_memory=True, shuffle=False, collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, self.batch_size, num_workers=self.num_workers,
            drop_last=False, pin_memory=True, shuffle=False, collate_fn=self.collate_fn,
        )
