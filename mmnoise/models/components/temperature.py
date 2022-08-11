import math

import torch


__all__ = [
    'TemperatureScale',
]


class TemperatureScale(torch.nn.Module):
    def __init__(self, start_val=1.0):
        super().__init__()
        # log-parameterized temperature scalar
        # use 1 / start_val initialization to be compatible with the convention of dividing by the
        # temperature
        self.temp = torch.nn.parameter.Parameter(
            torch.full((1,), math.log(1.0 / start_val), dtype=torch.float32)
        )

    def forward(self, x):
        return x * self.temp.exp()