"""Train the encoder-decoder Transformer on the synthetic copy task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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
    parser.add_argument("--evaluation-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/copy_task_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.log_interval <= 0 or args.evaluation_batches <= 0:
        raise ValueError("training and evaluation counts must be positive")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TransformerConfig(
        maximum_sequence_length=max(20, args.sequence_length + 1)
    )
    model = EncoderDecoderTransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

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
        if epoch == 1 or epoch % args.log_interval == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")
            history.append({"epoch": epoch, "training_loss": loss.item()})

    model.eval()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    exact_sequences = 0
    total_sequences = 0
    example: torch.Tensor | None = None
    prediction: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(args.evaluation_batches):
            batch = generate_copy_batch(
                args.batch_size,
                args.sequence_length,
                config.source_vocab_size,
                device=device,
                generator=generator,
            )
            expected = batch[:, 1:]
            logits = model(batch, batch[:, :-1])
            total_loss += criterion(
                logits.reshape(-1, config.target_vocab_size),
                expected.reshape(-1),
            ).item()
            predicted_tokens = logits.argmax(dim=-1)
            correct_tokens += predicted_tokens.eq(expected).sum().item()
            total_tokens += expected.numel()

            decoded = model.greedy_decode(
                batch,
                START_TOKEN_ID,
                END_TOKEN_ID,
                args.sequence_length,
            )
            if decoded.size(1) < args.sequence_length:
                decoded = torch.cat(
                    (
                        decoded,
                        torch.full(
                            (decoded.size(0), args.sequence_length - decoded.size(1)),
                            END_TOKEN_ID,
                            dtype=torch.long,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
            decoded = decoded[:, : args.sequence_length]
            exact_sequences += decoded.eq(batch).all(dim=1).sum().item()
            total_sequences += batch.size(0)
            if example is None:
                example = batch[0].detach().cpu()
                prediction = decoded[0].detach().cpu()

    metrics = {
        "task": "synthetic sequence copy",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "configuration": {
            **config.to_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "evaluation_batches": args.evaluation_batches,
            "log_interval": args.log_interval,
        },
        "elapsed_seconds": time.perf_counter() - started_at,
        "history": history,
        "evaluation": {
            "teacher_forced_loss": total_loss / args.evaluation_batches,
            "teacher_forced_token_accuracy": correct_tokens / total_tokens,
            "greedy_sequence_accuracy": exact_sequences / total_sequences,
            "evaluated_sequences": total_sequences,
        },
        "example": {
            "input": example.tolist(),
            "prediction": prediction.tolist(),
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("input:     ", example.tolist())
    print("prediction:", prediction.tolist())
    print(f"saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()

