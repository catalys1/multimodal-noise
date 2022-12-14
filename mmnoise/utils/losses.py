import torch

from . import distributed


class CLIPLoss(torch.nn.Module):
    def contrastive_loss(self, x, y, temperature_scale):
        # if distributed training, gather targets from all gpus to serve as negatives       
        # https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
        if distributed.is_distributed():
            rank = distributed.rank()
            with torch.no_grad():
                all_y = [torch.zeros_like(y) for _ in range(distributed.world_size())]
                torch.distributed.all_gather(all_y, y)
            all_y[rank] = y  #
            y = torch.cat(all_y, 0)
            labels = torch.arange(rank * x.shape[0], (rank + 1) * x.shape[0], device=x.device)
        else:
            labels = torch.arange(x.shape[0], device=x.device)

        logits = x @ y.T
        if isinstance(temperature_scale, torch.nn.Module):
            logits = temperature_scale(logits)
        else:
            logits = logits * (1 / temperature_scale)

        return torch.nn.functional.cross_entropy(logits, labels)

    def forward(self, x, y, temperature_scale=1.0):
        loss = 0.5 * (self.contrastive_loss(x, y, temperature_scale) + self.contrastive_loss(y, x, temperature_scale))
        return loss


class CLIPLossFullBidirectional(torch.nn.Module):
    def contrastive_loss(self, x, y, temperature_scale):
        labels = torch.arange(x.shape[0], device=x.device)
        logits = x @ y.T
        if isinstance(temperature_scale, torch.nn.Module):
            logits = temperature_scale(logits)
        else:
            logits = logits * (1 / temperature_scale)
        
        return torch.nn.functional.cross_entropy(logits, labels)

    def forward(self, x, y, temperature_scale=1.0):
        if distributed.is_distributed():
            x = torch.cat(distributed.GatherLayer.apply(x), 0)
            y = torch.cat(distributed.GatherLayer.apply(y), 0)
        loss = 0.5 * (self.contrastive_loss(x, y, temperature_scale) + self.contrastive_loss(y, x, temperature_scale))
        return loss
