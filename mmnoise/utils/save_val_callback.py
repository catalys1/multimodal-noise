from pathlib import Path

import pytorch_lightning as pl
import torch


class SaveValPredsCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.store_val_predictions = True

    def on_validation_epoch_end(self, trainer, pl_module):
        preds = pl_module.val_predictions.cpu()
        targets = pl_module.val_targets.cpu()

        path = Path(trainer.default_root_dir).joinpath('val_preds.pth')
        torch.save(dict(preds=preds, targets=targets), path)
