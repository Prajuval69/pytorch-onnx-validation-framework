import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        factory = {"device": "cpu", "dtype": torch.float32}
        self.layers = nn.Sequential(
            nn.Linear(32, 64, **factory), nn.ReLU(), nn.Linear(64, 8, **factory)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
