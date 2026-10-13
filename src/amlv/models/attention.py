import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        factory = {"device": "cpu", "dtype": torch.float32}
        self.query = nn.Linear(16, 16, **factory)
        self.key = nn.Linear(16, 16, **factory)
        self.value = nn.Linear(16, 16, **factory)
        self.output = nn.Linear(16, 16, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.query(x), self.key(x).transpose(-2, -1)) / 4.0
        return self.output(torch.matmul(torch.softmax(scores, dim=-1), self.value(x)))
