import os
from typing import Optional

from PIL import Image
import torch
import torchvision

from .core import BaseMultimodalDatamodule
from .core import image_embedding_collate, image_text_collate, sample_caption


__all__ = [
    'Flickr30kDatamodule',
    'Flickr30kEmbeddedDatamodule',
]


class Flickr30kDatamodule(BaseMultimodalDatamodule):

    def _post_init(self):
        self.dataset_class = torchvision.datasets.Flickr30k
        self.ann_file = os.path.join(self.root, 'annotations.txt')

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

    def setup(self, stage: Optional[str] = None):
        root = os.path.join(self.root, 'flickr30k-images')
        ann_file = self.ann_file
        if stage == 'fit':
            # training data
            train_split = os.path.join(self.root, 'flickr30k_entities', 'train.txt')
            transform = self.transforms(stage='train')
            train_data = self.dataset_class(root, ann_file, transform, sample_caption)
            self.train_data = self.split_data(train_data, train_split)
        if stage in ('fit', 'validate'):
            # validation data
            val_split = os.path.join(self.root, 'flickr30k_entities', 'val.txt')
            transform = self.transforms(stage='val')
            val_data = self.dataset_class(root, ann_file, transform)
            self.val_data = self.split_data(val_data, val_split)
        if stage == 'test':
            # test data
            test_split = os.path.join(self.root, 'flickr30k_entities', 'test.txt')
            transform = self.transforms(stage='test')
            test_data = self.dataset_class(root, ann_file, transform)
            self.test_data = self.split_data(test_data, test_split)

    @property
    def collate_fn(self):
        return image_text_collate


#############################################
# Flickr30k with caption embeddings
#############################################
class Flickr30kEmbeddingDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        self.annotations = torch.load(self.ann_file)
        self.stats = self.annotations.pop('stats')
        self.ids = sorted(self.annotations.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]

        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = target.float().sub(self.stats['mean']).div_(self.stats['stdev'])

        return img, target

    def __len__(self):
        return len(self.ids)


class Flickr30kEmbeddedDatamodule(Flickr30kDatamodule):

    def _post_init(self):
        self.dataset_class = Flickr30kEmbeddingDataset
        self.ann_file = os.path.join(self.root, 'caption_embeddings.pth')
    
    @property
    def collate_fn(self):
        return image_embedding_collate