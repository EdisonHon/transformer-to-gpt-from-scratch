import math

import torch

from transformer_to_gpt.attention import causal_mask, padding_mask, scaled_dot_product_attention


def test_scaled_dot_product_attention_returns_expected_shapes_and_probabilities():
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 5, 8)
    value = torch.randn(2, 4, 5, 8)

    context, weights = scaled_dot_product_attention(query, key, value)

    assert context.shape == (2, 4, 5, 8)
    assert weights.shape == (2, 4, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5))


def test_attention_matches_a_small_manual_example():
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[3.0, 1.0], [1.0, 5.0]]]])

    context, weights = scaled_dot_product_attention(query, key, value)
    expected_weights = torch.softmax(torch.tensor([1.0, 0.0]) / math.sqrt(2), dim=0)

    assert torch.allclose(weights[0, 0, 0], expected_weights)
    assert torch.allclose(context[0, 0, 0], expected_weights @ value[0, 0])


def test_causal_mask_removes_future_attention():
    query = torch.ones(1, 1, 4, 2)
    key = torch.ones(1, 1, 4, 2)
    value = torch.arange(8, dtype=torch.float32).view(1, 1, 4, 2)

    _, weights = scaled_dot_product_attention(query, key, value, causal_mask(4))

    assert torch.count_nonzero(torch.triu(weights[0, 0], diagonal=1)) == 0


def test_padding_mask_hides_padding_keys():
    tokens = torch.tensor([[7, 8, 0, 0]])
    mask = padding_mask(tokens, pad_token_id=0)

    assert mask.shape == (1, 1, 1, 4)
    assert mask.tolist() == [[[[True, True, False, False]]]]


def test_fully_masked_attention_returns_zero_context_without_nan():
    query = torch.ones(1, 1, 1, 2)
    key = torch.ones(1, 1, 2, 2)
    value = torch.ones(1, 1, 2, 3)
    mask = torch.zeros(1, 1, 1, 2, dtype=torch.bool)

    context, weights = scaled_dot_product_attention(query, key, value, mask)

    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.equal(context, torch.zeros_like(context))

