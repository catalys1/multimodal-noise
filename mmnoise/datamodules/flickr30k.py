from itertools import chain
import os
import random
from typing import Callable, Optional, Tuple

from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, default_collate
import torchvision


__all__ = [
    'Flickr30kDatamodule',
]

torchvision.transforms.ColorJitter()
class Flickr30kDatamodule(LightningDataModule):

    dataset_class = torchvision.datasets.Flickr30k

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 224,
        normalization: Tuple[Tuple, Tuple] = ((0.485, 0.456, 0.406), (0.229, 0.225, 0.226)),
        min_crop_area: float = 0.2,
        hflips: bool = False,
        color_jitter: Optional[Tuple] = None,  # (brightness, contrast, saturation, hue)
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.normalization = normalization
        self.min_crop_area = min_crop_area
        self.hflips = hflips
        self.color_jitter = color_jitter

    @staticmethod
    def _load_split(path):
        with open(path, 'r') as f:
            content = set(f.read().strip().split('\n'))
        return content

    def split_data(self, dataset, split_file):
        '''Set the data to a specific split, using a text file containing a single image
        ID per line.
        '''
        split = self._load_split(split_file)
        dataset.ids = [x for x in dataset.ids if x[:-4] in split]
        dataset.annotations = dict((id, dataset.annotations[id]) for id in dataset.ids)
        return dataset

    def transforms(self, stage: str = 'train') -> Callable:
        T = torchvision.transforms
        tensorize = T.Compose([
            T.ToTensor(),
            T.Normalize(*self.normalization),
        ])
        if stage == 'train':
            ts = [T.RandomResizedCrop(self.image_size, (self.min_crop_area, 1.0))]
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
        root = os.path.join(self.root, 'flickr30k-images')
        ann_file = os.path.join(self.root, 'annotations.txt')
        if stage == 'fit':
            # training data
            train_split = os.path.join(self.root, 'flickr30k_entities', 'train.txt')
            transform = self.transforms(stage='train')
            train_data = self.dataset_class(root, ann_file, transform, sample_caption)
            self.train_data = self.split_data(train_data, train_split)
            # validation data
            val_split = os.path.join(self.root, 'flickr30k_entities', 'val.txt')
            transform = self.transforms(stage='val')
            val_data = self.dataset_class(root, ann_file, transform)
            self.val_data = self.split_data(val_data, val_split)
        else:
            # test data
            test_split = os.path.join(self.root, 'flickr30k_entities', 'test.txt')
            transform = self.transforms(stage='test')
            test_data = self.dataset_class(root, ann_file, transform)
            self.test_data = self.split_data(test_data, test_split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, self.batch_size, num_workers=self.num_workers,
            pin_memory=True, drop_last=True, shuffle=True, collate_fn=image_text_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, self.batch_size, num_workers=self.num_workers,
            drop_last=False, pin_memory=True, shuffle=False, collate_fn=image_text_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, self.batch_size, num_workers=self.num_workers,
            drop_last=False, pin_memory=True, shuffle=False, collate_fn=image_text_collate,
        )


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
