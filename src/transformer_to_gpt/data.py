"""Tokenization and synthetic data helpers for the coursework models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CharacterTokenizer:
    """Deterministic character-level tokenizer."""

    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("text must not be empty")
        return cls(tuple(sorted(set(text))))

    @property
    def vocabulary_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        character_to_id = {
            character: token_id for token_id, character in enumerate(self.characters)
        }
        try:
            return [character_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character not in vocabulary: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.characters[token_id] for token_id in token_ids)
        except IndexError as error:
            raise ValueError("token id is outside the vocabulary") from error

    def to_dict(self) -> dict[str, list[str]]:
        return {"characters": list(self.characters)}

    @classmethod
    def from_dict(cls, values: dict[str, list[str]]) -> "CharacterTokenizer":
        return cls(tuple(values["characters"]))


def split_language_data(
    encoded_text: Tensor,
    training_fraction: float = 0.9,
) -> tuple[Tensor, Tensor]:
    """Split an encoded corpus sequentially into training and validation data."""
    if encoded_text.ndim != 1:
        raise ValueError("encoded_text must be one-dimensional")
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must be between zero and one")
    split_index = int(training_fraction * len(encoded_text))
    if split_index == 0 or split_index == len(encoded_text):
        raise ValueError("encoded_text is too short for the requested split")
    return encoded_text[:split_index], encoded_text[split_index:]


def sample_language_model_batch(
    data: Tensor,
    batch_size: int,
    block_size: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample input sequences and one-token-shifted targets."""
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive")
    if len(data) <= block_size:
        raise ValueError("data must contain more tokens than block_size")
    starts = torch.randint(
        len(data) - block_size,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([data[index : index + block_size] for index in starts])
    targets = torch.stack([data[index + 1 : index + block_size + 1] for index in starts])
    return inputs.to(device), targets.to(device)


def generate_copy_batch(
    batch_size: int,
    sequence_length: int,
    vocabulary_size: int,
    *,
    start_token_id: int = 1,
    end_token_id: int = 2,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate a synthetic sequence batch for the encoder-decoder copy task."""
    if batch_size <= 0 or sequence_length < 2:
        raise ValueError("batch_size must be positive and sequence_length must be at least two")
    if vocabulary_size <= max(start_token_id, end_token_id) + 1:
        raise ValueError("vocabulary_size is too small for the reserved token ids")
    random_tokens = torch.randint(
        max(start_token_id, end_token_id) + 1,
        vocabulary_size,
        (batch_size, sequence_length - 2),
        generator=generator,
    )
    start = torch.full((batch_size, 1), start_token_id)
    end = torch.full((batch_size, 1), end_token_id)
    return torch.cat((start, random_tokens, end), dim=1).long().to(device)

