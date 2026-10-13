import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        factory = {"device": "cpu", "dtype": torch.float32}
        self.conv = nn.Conv2d(1, 4, 3, padding=1, **factory)
        self.head = nn.Linear(4, 6, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))
