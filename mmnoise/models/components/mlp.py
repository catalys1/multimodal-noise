from typing import List, Union
import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_dims: Union[int, List[int]],
        activation: torch.nn.Module = torch.nn.ReLU,
        norm: torch.nn.Module = torch.nn.BatchNorm1d,
        dropout: Union[float, List[float]] = 0.0,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        if isinstance(layer_dims, int):
            layer_dims = [layer_dims]
        elif isinstance(layer_dims, tuple):
            layer_dims = list(layer_dims)
        if isinstance(dropout, float):
            dropout = [dropout for _ in range(len(layer_dims))]
        if len(dropout) < len(layer_dims):
            dropout.extend([0.0] * (len(layer_dims) - len(dropout)))

        din = in_dim
        for dim, drop in zip(layer_dims, dropout):
            dout = dim
            self.layers.append(torch.nn.Linear(din, dout))
            self.layers.append(activation(inplace=True))
            self.layers.append(norm(dout))
            self.layers.append(torch.nn.Dropout(drop))
            din = dout

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, 2**0.5)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.InstanceNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
