import time

import pytorch_lightning as pl


class PrintProgressCallback(pl.callbacks.ProgressBarBase):
    def __init__(self, print_interval=20):
        super().__init__()
        self._enable = True
        self.print_interval = print_interval

    def enable(self):
        self._enable = True
    
    def disable(self):
        self._enable = False

    def on_train_epoch_start(self, *args, **kwargs):
        self.avg_time = 0
        self.steps = 0
        self.train_epoch_time = time.time()

    def on_train_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapse = time.time() - self.start_time
        self.avg_time += elapse
        self.steps += 1

        batch_idx = self.train_batch_idx
        if batch_idx % self.print_interval == 1:
            t = round(self.avg_time / self.steps, 4)
            tb = str(self.total_train_batches)
            b = str(self.train_batch_idx).zfill(len(tb))
            head = 'Train epoch {}/{} (batch {}/{}):  batch_time = {}  '.format(
                self.trainer.current_epoch+1, self.trainer.max_epochs, b, tb, t
            )
            metrics = self.get_metrics(trainer)
            # metrics = {
            #     k: v if isinstance(v, str) else round(v, 4)
            #     for k, v in metrics.items() if not k.startswith('val/')
            # }
            out = '   '.join(f'{k} = {v}' for k, v in metrics.items())
            print(head + out)

    def on_train_epoch_end(self, *args, **kwargs):
        t = round(time.time() - self.train_epoch_time, 4)
        print('Train epoch {} elapsed time: {}'.format(self.trainer.current_epoch+1, t))

    def on_validation_epoch_start(self, *args, **kwargs):
        self.val_epoch_time = time.time()
        self.avg_time = 0
        self.steps = 0

    def on_validation_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        elapse = time.time() - self.start_time
        self.avg_time += elapse
        self.steps += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        t = round(self.avg_time / self.steps, 4)
        head = 'Val epoch {}/{}:  batch_time = {}  '.format(
            self.trainer.current_epoch+1, self.trainer.max_epochs, t
        )
        # metrics = {k: round(v, 4) for k, v in metrics.items() if k.startswith('val/')}
        metrics = self.get_metrics(trainer, val=True)
        out = '   '.join(f'{k} = {v}' for k, v in metrics.items())
        print(head + out)

        t = round(time.time() - self.val_epoch_time, 4)
        print('Val epoch {} elapsed time: {}'.format(self.trainer.current_epoch+1, t))

    def get_metrics(self, trainer, val=False):
        metrics = trainer.callback_metrics
        if val:
            metrics = {k: v for k, v in metrics.items() if k.startswith('val/')}
        else:
            metrics = {k: v for k, v in metrics.items() if not k.startswith('val/')}
        for k in metrics:
            v = metrics[k]
            try:
                metrics[k] = round(v.item(), 4)
            except: pass
        return metrics
