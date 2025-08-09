from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2


@dataclass
class PretrainConfig:
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    temperature: float = 0.2


