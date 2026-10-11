import pytest
import torch

from transformer_to_gpt.transformer import (
    EncoderDecoderTransformer,
    TokenAndPositionalEmbedding,
    TransformerConfig,
)


def small_config() -> TransformerConfig:
    return TransformerConfig(
        source_vocab_size=20,
        target_vocab_size=20,
        model_dimension=12,
        number_of_heads=3,
        number_of_layers=2,
        feed_forward_dimension=24,
        maximum_sequence_length=12,
        dropout=0.0,
    )


def test_positional_embedding_supports_odd_dimensions():
    layer = TokenAndPositionalEmbedding(10, 7, 8)
    output = layer(torch.tensor([[1, 2, 3]]))

    assert output.shape == (1, 3, 7)
    assert layer.position_encoding.shape == (1, 8, 7)


def test_encoder_decoder_forward_shape_with_padding():
    model = EncoderDecoderTransformer(small_config())
    source = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 7, 8, 2]])
    target = torch.tensor([[1, 4, 5, 0], [1, 6, 7, 8]])

    logits = model(source, target)

    assert logits.shape == (2, 4, 20)
    assert torch.isfinite(logits).all()


def test_greedy_decode_supports_batches_and_restores_training_mode():
    model = EncoderDecoderTransformer(small_config())
    model.train()
    source = torch.tensor([[1, 4, 2], [1, 5, 2]])

    output = model.greedy_decode(source, 1, 2, maximum_length=6)

    assert output.shape[0] == 2
    assert 1 <= output.shape[1] <= 6
    assert model.training


def test_embedding_rejects_sequences_beyond_configured_length():
    layer = TokenAndPositionalEmbedding(10, 8, 3)

    with pytest.raises(ValueError, match="exceeds maximum"):
        layer(torch.tensor([[1, 2, 3, 4]]))

