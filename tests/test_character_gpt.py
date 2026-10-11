import torch

from transformer_to_gpt.data import CharacterTokenizer
from transformer_to_gpt.gpt import (
    CharacterGPT,
    GPTConfig,
    load_checkpoint,
    save_checkpoint,
)


def small_config() -> GPTConfig:
    return GPTConfig(
        vocabulary_size=8,
        block_size=12,
        model_dimension=16,
        number_of_heads=4,
        number_of_layers=2,
        dropout=0.0,
    )


def test_gpt_forward_returns_logits_and_loss():
    model = CharacterGPT(small_config())
    tokens = torch.randint(0, 8, (3, 6))

    logits, loss = model(tokens, tokens)

    assert logits.shape == (3, 6, 8)
    assert loss is not None
    assert loss.ndim == 0


def test_gpt_is_causal():
    torch.manual_seed(7)
    model = CharacterGPT(small_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 7, 6]])

    first_logits, _ = model(first)
    second_logits, _ = model(second)

    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)


def test_generation_restores_training_mode():
    model = CharacterGPT(small_config())
    model.train()

    generated = model.generate(torch.tensor([[1]]), 4, top_k=3)

    assert generated.shape == (1, 5)
    assert model.training


def test_checkpoint_round_trip_includes_tokenizer_and_metadata(tmp_path):
    model = CharacterGPT(small_config())
    tokenizer = CharacterTokenizer(tuple("abcdefgh"))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, tokenizer, training_metadata={"seed": 1337})

    restored, restored_tokenizer, metadata = load_checkpoint(
        path,
        device=torch.device("cpu"),
    )

    assert restored.config == model.config
    assert restored_tokenizer == tokenizer
    assert metadata == {"seed": 1337}
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_original_checkpoint_parameter_names_remain_compatible():
    keys = CharacterGPT(GPTConfig()).state_dict().keys()

    assert "token_embedding_table.weight" in keys
    assert "blocks.0.sa.heads.0.W_K.weight" in keys
    assert "blocks.5.ff.ff.2.weight" in keys
    assert "ff.bias" in keys

