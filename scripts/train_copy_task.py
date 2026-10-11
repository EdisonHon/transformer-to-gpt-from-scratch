"""Train the encoder-decoder Transformer on the synthetic copy task."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from transformer_to_gpt.data import generate_copy_batch
from transformer_to_gpt.transformer import EncoderDecoderTransformer, TransformerConfig


START_TOKEN_ID = 1
END_TOKEN_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransformerConfig(
        maximum_sequence_length=max(20, args.sequence_length + 1)
    )
    model = EncoderDecoderTransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    model.train()
    for epoch in range(1, args.epochs + 1):
        sequence = generate_copy_batch(
            args.batch_size,
            args.sequence_length,
            config.source_vocab_size,
            device=device,
            generator=generator,
        )
        decoder_input = sequence[:, :-1]
        expected_output = sequence[:, 1:]
        logits = model(sequence, decoder_input)
        loss = criterion(
            logits.reshape(-1, config.target_vocab_size),
            expected_output.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    example = generate_copy_batch(
        1,
        args.sequence_length,
        config.source_vocab_size,
        device=device,
        generator=generator,
    )
    prediction = model.greedy_decode(
        example,
        START_TOKEN_ID,
        END_TOKEN_ID,
        args.sequence_length + 1,
    )
    print("input:     ", example[0].tolist())
    print("prediction:", prediction[0].tolist())


if __name__ == "__main__":
    main()

