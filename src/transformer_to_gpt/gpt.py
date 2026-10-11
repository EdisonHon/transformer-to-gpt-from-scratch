"""Decoder-only character GPT from the COMS 4705 coursework implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import scaled_dot_product_attention
from .data import CharacterTokenizer


@dataclass(frozen=True)
class GPTConfig:
    """Configuration matching the original character-level GPT by default."""

    vocabulary_size: int = 62
    block_size: int = 128
    model_dimension: int = 96
    number_of_heads: int = 6
    number_of_layers: int = 6
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.model_dimension % self.number_of_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_heads")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(
        self,
        model_dimension: int,
        head_dimension: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.W_K = nn.Linear(model_dimension, head_dimension, bias=False)
        self.W_Q = nn.Linear(model_dimension, head_dimension, bias=False)
        self.W_V = nn.Linear(model_dimension, head_dimension, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        sequence_length = inputs.size(1)
        query = self.W_Q(inputs).unsqueeze(1)
        key = self.W_K(inputs).unsqueeze(1)
        value = self.W_V(inputs).unsqueeze(1)
        mask = self.tril[:sequence_length, :sequence_length].bool().view(
            1,
            1,
            sequence_length,
            sequence_length,
        )
        context, _ = scaled_dot_product_attention(
            query,
            key,
            value,
            mask,
            dropout_probability=self.dropout.p,
            training=self.training,
        )
        return context.squeeze(1)


class MultiHeadAttention(nn.Module):
    """Parallel causal attention heads followed by an output projection."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        head_dimension = config.model_dimension // config.number_of_heads
        self.heads = nn.ModuleList(
            Head(
                config.model_dimension,
                head_dimension,
                config.block_size,
                config.dropout,
            )
            for _ in range(config.number_of_heads)
        )
        self.W_O = nn.Linear(config.number_of_heads * head_dimension, config.model_dimension)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        combined = torch.cat([head(inputs) for head in self.heads], dim=-1)
        return self.dropout(self.W_O(combined))


class FeedForward(nn.Module):
    """Position-wise feed-forward network used by each GPT block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.model_dimension, 4 * config.model_dimension),
            nn.ReLU(),
            nn.Linear(4 * config.model_dimension, config.model_dimension),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.ff(inputs)


class DecoderBlock(nn.Module):
    """Pre-LayerNorm causal Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.model_dimension)
        self.ln2 = nn.LayerNorm(config.model_dimension)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.sa(self.ln1(inputs))
        return inputs + self.ff(self.ln2(inputs))


class CharacterGPT(nn.Module):
    """Decoder-only Transformer trained to predict the next character."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        # These names intentionally match the original state dictionary.
        self.token_embedding_table = nn.Embedding(
            config.vocabulary_size,
            config.model_dimension,
        )
        self.position_embedding_table = nn.Embedding(
            config.block_size,
            config.model_dimension,
        )
        self.blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.number_of_layers)]
        )
        self.ln = nn.LayerNorm(config.model_dimension)
        self.ff = nn.Linear(config.model_dimension, config.vocabulary_size)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")
        sequence_length = token_ids.size(1)
        if sequence_length > self.config.block_size:
            raise ValueError("input sequence exceeds the configured block size")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding_table(token_ids)
        hidden = hidden + self.position_embedding_table(positions)
        logits = self.ff(self.ln(self.blocks(hidden)))

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        maximum_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Sample characters autoregressively from the model."""
        if maximum_new_tokens < 0:
            raise ValueError("maximum_new_tokens must not be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        was_training = self.training
        self.eval()
        for _ in range(maximum_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_logits = logits[:, -1] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive")
                count = min(top_k, next_logits.size(-1))
                threshold = torch.topk(next_logits, count).values[:, -1].unsqueeze(1)
                next_logits = next_logits.masked_fill(
                    next_logits < threshold,
                    float("-inf"),
                )
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        self.train(was_training)
        return token_ids


def save_checkpoint(
    path: str | Path,
    model: CharacterGPT,
    tokenizer: CharacterTokenizer,
    *,
    training_metadata: dict[str, Any] | None = None,
) -> None:
    """Save model configuration, tokenizer, weights, and verified metadata."""
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state": model.state_dict(),
            "training_metadata": training_metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[CharacterGPT, CharacterTokenizer, dict[str, Any]]:
    """Load a self-contained checkpoint using PyTorch's weights-only mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"model_config", "tokenizer", "model_state"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(
            "checkpoint does not include its tokenizer and configuration; "
            "use load_original_state_dict for the archived coursework weights"
        )
    model = CharacterGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = CharacterTokenizer.from_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint.get("training_metadata", {})


def load_original_state_dict(
    path: str | Path,
    *,
    device: torch.device,
) -> CharacterGPT:
    """Load the original 62-character coursework weights without a tokenizer."""
    model = CharacterGPT(GPTConfig()).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

