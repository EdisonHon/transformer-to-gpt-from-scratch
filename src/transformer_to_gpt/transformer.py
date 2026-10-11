"""Encoder-decoder Transformer from the COMS 4701 coursework implementation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import causal_mask, combine_masks, padding_mask, scaled_dot_product_attention


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the encoder-decoder Transformer."""

    source_vocab_size: int = 100
    target_vocab_size: int = 100
    model_dimension: int = 64
    number_of_heads: int = 2
    number_of_layers: int = 2
    feed_forward_dimension: int = 128
    maximum_sequence_length: int = 20
    dropout: float = 0.1
    pad_token_id: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TokenAndPositionalEmbedding(nn.Module):
    """Learned token embeddings with fixed sinusoidal positional encodings."""

    def __init__(self, vocab_size: int, model_dimension: int, maximum_sequence_length: int) -> None:
        super().__init__()
        if vocab_size <= 0 or model_dimension <= 0 or maximum_sequence_length <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.model_dimension = model_dimension
        self.maximum_sequence_length = maximum_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, model_dimension)

        positions = torch.arange(maximum_sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_dimension, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / model_dimension)
        )
        encoding = torch.zeros(maximum_sequence_length, model_dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        odd_width = encoding[:, 1::2].size(1)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:odd_width])
        self.register_buffer("position_encoding", encoding.unsqueeze(0))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch_size, sequence_length)")
        sequence_length = tokens.size(1)
        if sequence_length > self.maximum_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds maximum "
                f"{self.maximum_sequence_length}"
            )
        token_embeddings = self.token_embedding(tokens) * math.sqrt(self.model_dimension)
        return token_embeddings + self.position_encoding[:, :sequence_length]


class MultiHeadAttention(nn.Module):
    """Multi-head attention for self-attention and cross-attention."""

    def __init__(self, model_dimension: int, number_of_heads: int) -> None:
        super().__init__()
        if model_dimension % number_of_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_heads")
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.query_projection = nn.Linear(model_dimension, model_dimension, bias=False)
        self.key_projection = nn.Linear(model_dimension, model_dimension, bias=False)
        self.value_projection = nn.Linear(model_dimension, model_dimension, bias=False)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=False)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, sequence_length, _ = values.shape
        values = values.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_dimension,
        )
        return values.transpose(1, 2)

    def forward(
        self,
        query_input: Tensor,
        key_input: Tensor,
        value_input: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        query = self._split_heads(self.query_projection(query_input))
        key = self._split_heads(self.key_projection(key_input))
        value = self._split_heads(self.value_projection(value_input))
        context, _ = scaled_dot_product_attention(query, key, value, mask)
        batch_size, _, query_length, _ = context.shape
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.model_dimension,
        )
        return self.output_projection(context)


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network applied independently to each token."""

    def __init__(
        self,
        model_dimension: int,
        feed_forward_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.linear2 = nn.Linear(feed_forward_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(inputs))))


class TransformerEncoderBlock(nn.Module):
    """Post-LayerNorm Transformer encoder block."""

    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dimension, number_of_heads)
        self.feed_forward = PositionwiseFeedForward(
            model_dimension,
            feed_forward_dimension,
            dropout,
        )
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, source_mask: Tensor | None = None) -> Tensor:
        attention_output = self.self_attention(inputs, inputs, inputs, source_mask)
        hidden = self.norm1(inputs + self.dropout1(attention_output))
        feed_forward_output = self.feed_forward(hidden)
        return self.norm2(hidden + self.dropout2(feed_forward_output))


class TransformerDecoderBlock(nn.Module):
    """Post-LayerNorm decoder block with masked and cross-attention."""

    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        feed_forward_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(model_dimension, number_of_heads)
        self.cross_attention = MultiHeadAttention(model_dimension, number_of_heads)
        self.feed_forward = PositionwiseFeedForward(
            model_dimension,
            feed_forward_dimension,
            dropout,
        )
        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
        self.norm3 = nn.LayerNorm(model_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        inputs: Tensor,
        encoder_output: Tensor,
        target_mask: Tensor,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        attention_output = self.masked_self_attention(
            inputs,
            inputs,
            inputs,
            target_mask,
        )
        hidden = self.norm1(inputs + self.dropout1(attention_output))
        cross_output = self.cross_attention(
            hidden,
            encoder_output,
            encoder_output,
            source_mask,
        )
        hidden = self.norm2(hidden + self.dropout2(cross_output))
        feed_forward_output = self.feed_forward(hidden)
        return self.norm3(hidden + self.dropout3(feed_forward_output))


class EncoderDecoderTransformer(nn.Module):
    """Transformer for sequence-to-sequence learning."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedding = TokenAndPositionalEmbedding(
            config.source_vocab_size,
            config.model_dimension,
            config.maximum_sequence_length,
        )
        self.target_embedding = TokenAndPositionalEmbedding(
            config.target_vocab_size,
            config.model_dimension,
            config.maximum_sequence_length,
        )
        self.encoder_layers = nn.ModuleList(
            TransformerEncoderBlock(
                config.model_dimension,
                config.number_of_heads,
                config.feed_forward_dimension,
                config.dropout,
            )
            for _ in range(config.number_of_layers)
        )
        self.decoder_layers = nn.ModuleList(
            TransformerDecoderBlock(
                config.model_dimension,
                config.number_of_heads,
                config.feed_forward_dimension,
                config.dropout,
            )
            for _ in range(config.number_of_layers)
        )
        self.output_projection = nn.Linear(
            config.model_dimension,
            config.target_vocab_size,
        )
        self.dropout = nn.Dropout(config.dropout)

    def encode(self, source: Tensor, source_mask: Tensor | None = None) -> Tensor:
        source_mask = (
            padding_mask(source, self.config.pad_token_id)
            if source_mask is None
            else source_mask
        )
        hidden = self.dropout(self.source_embedding(source))
        for layer in self.encoder_layers:
            hidden = layer(hidden, source_mask)
        return hidden

    def decode(
        self,
        target: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor | None = None,
        target_padding_mask: Tensor | None = None,
    ) -> Tensor:
        visible_past = causal_mask(target.size(1), device=target.device)
        visible_target_tokens = (
            padding_mask(target, self.config.pad_token_id)
            if target_padding_mask is None
            else target_padding_mask
        )
        target_mask = combine_masks(visible_past, visible_target_tokens)
        hidden = self.dropout(self.target_embedding(target))
        for layer in self.decoder_layers:
            hidden = layer(hidden, encoder_output, target_mask, source_mask)
        return hidden

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mask: Tensor | None = None,
        target_padding_mask: Tensor | None = None,
    ) -> Tensor:
        source_mask = (
            padding_mask(source, self.config.pad_token_id)
            if source_mask is None
            else source_mask
        )
        memory = self.encode(source, source_mask)
        hidden = self.decode(target, memory, source_mask, target_padding_mask)
        return self.output_projection(hidden)

    @torch.no_grad()
    def greedy_decode(
        self,
        source: Tensor,
        start_token_id: int,
        end_token_id: int,
        maximum_length: int,
    ) -> Tensor:
        """Decode a batch autoregressively using the highest-probability token."""
        if maximum_length > self.config.maximum_sequence_length:
            raise ValueError("maximum_length exceeds configured positional encoding length")
        was_training = self.training
        self.eval()
        source_mask = padding_mask(source, self.config.pad_token_id)
        memory = self.encode(source, source_mask)
        generated = torch.full(
            (source.size(0), 1),
            start_token_id,
            dtype=torch.long,
            device=source.device,
        )
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(maximum_length - 1):
            hidden = self.decode(generated, memory, source_mask)
            next_token = self.output_projection(hidden[:, -1]).argmax(dim=-1, keepdim=True)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, end_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token), dim=1)
            finished |= next_token.squeeze(1).eq(end_token_id)
            if finished.all():
                break
        self.train(was_training)
        return generated

