from dataclasses import dataclass

import torch
from torch import nn

from amlv.errors import ConfigError
from amlv.models.attention import Attention
from amlv.models.cnn import CNN
from amlv.models.mlp import MLP

ARCHITECTURE_VERSION = 1
SHAPES = {"mlp": (32,), "cnn": (1, 16, 16), "attention": (4, 16)}
FACTORIES = {"mlp": MLP, "cnn": CNN, "attention": Attention}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    seed: int = 2026
    input_names: tuple[str, ...] = ("input",)
    output_names: tuple[str, ...] = ("output",)
    architecture_version: int = ARCHITECTURE_VERSION

    def shape(self, batch: int) -> tuple[int, ...]:
        if self.model_id not in SHAPES:
            raise ConfigError(f"Unknown model: {self.model_id}")
        if type(batch) is not int or not 1 <= batch <= 8:
            raise ConfigError("batch must be in [1,8]")
        return (batch, *SHAPES[self.model_id])


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.model_id not in FACTORIES or spec.architecture_version != ARCHITECTURE_VERSION:
        raise ConfigError("Unsupported model or architecture version")
    # Preserve caller RNG state; default device/dtype never determines the reference.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        model = FACTORIES[spec.model_id]().cpu().float().eval()
    model.requires_grad_(False)
    return model
