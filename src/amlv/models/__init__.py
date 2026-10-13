"""Small, native PyTorch models. Weights are seeded, frozen, and never trained."""

from amlv.models.registry import ModelSpec, build_model

__all__ = ["ModelSpec", "build_model"]
