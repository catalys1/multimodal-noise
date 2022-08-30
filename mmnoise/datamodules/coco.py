import os

import torchvision

from .core import BaseMultimodalDatamodule, sample_caption


__all__ = [
    'COCODatamodule',
]


class COCODatamodule(BaseMultimodalDatamodule):

    def _post_init(self):
        self.dataset_class = torchvision.datasets.CocoCaptions

    def setup(self, stage):
        if stage == 'fit':
            train_imgs = os.path.join(self.root, 'images/train2014')
            train_ann = os.path.join(self.root, 'annotations/captions_train2014.json')
            transform = self.transforms('train')
            self.train_data = self.dataset_class(train_imgs, train_ann, transform, sample_caption)
        if stage in ('fit', 'validate'):
            val_imgs = os.path.join(self.root, 'images/val2014')
            val_ann = os.path.join(self.root, 'annotations/captions_val2014.json')
            transform = self.transforms('val')
            self.val_data = self.dataset_class(val_imgs, val_ann, transform)
        if stage == 'test':
            test_imgs = os.path.join(self.root, 'images/test2014')
            test_ann = os.path.join(self.root, 'annotations/captions_test2014.json')
            transform = self.transforms('test')
            self.test_data = self.dataset_class(test_imgs, test_ann, transform)