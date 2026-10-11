import pytest
import torch

from transformer_to_gpt.data import (
    CharacterTokenizer,
    generate_copy_batch,
    sample_language_model_batch,
    split_language_data,
)


def test_character_tokenizer_round_trip():
    tokenizer = CharacterTokenizer.from_text("cab cab")
    text = "abc cab"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_language_model_batch_is_shifted_by_one_token():
    data = torch.arange(20)
    inputs, targets = sample_language_model_batch(
        data,
        batch_size=4,
        block_size=5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(2),
    )

    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_copy_batch_has_reserved_boundary_tokens():
    batch = generate_copy_batch(
        3,
        8,
        20,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(2),
    )

    assert batch.shape == (3, 8)
    assert torch.equal(batch[:, 0], torch.ones(3, dtype=torch.long))
    assert torch.equal(batch[:, -1], torch.full((3,), 2, dtype=torch.long))


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="between zero and one"):
        split_language_data(torch.arange(10), training_fraction=1.0)

