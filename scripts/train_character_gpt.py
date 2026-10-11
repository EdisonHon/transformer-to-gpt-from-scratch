"""Train the decoder-only GPT on a user-provided UTF-8 text corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer_to_gpt.data import (
    CharacterTokenizer,
    sample_language_model_batch,
    split_language_data,
)
from transformer_to_gpt.gpt import CharacterGPT, GPTConfig, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="local UTF-8 text corpus")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/character_gpt.pt"),
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--evaluation-interval", type=int, default=500)
    parser.add_argument("--evaluation-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: CharacterGPT,
    training_data: torch.Tensor,
    validation_data: torch.Tensor,
    *,
    batch_size: int,
    evaluation_batches: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Estimate mean training and validation losses over sampled batches."""
    was_training = model.training
    model.eval()
    estimates: dict[str, float] = {}
    for name, data in (("train", training_data), ("validation", validation_data)):
        losses = []
        for _ in range(evaluation_batches):
            inputs, targets = sample_language_model_batch(
                data,
                batch_size,
                model.config.block_size,
                device=device,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            losses.append(loss.item())
        estimates[name] = sum(losses) / len(losses)
    model.train(was_training)
    return estimates


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.evaluation_interval <= 0 or args.evaluation_batches <= 0:
        raise ValueError("training and evaluation counts must be positive")
    text = args.corpus.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    training_data, validation_data = split_language_data(encoded)
    if min(len(training_data), len(validation_data)) <= args.block_size:
        raise ValueError("both data splits must be longer than block_size")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        block_size=args.block_size,
    )
    model = CharacterGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    latest_metrics: dict[str, float] = {}

    model.train()
    for step in range(args.steps):
        if step % args.evaluation_interval == 0 or step == args.steps - 1:
            if step:
                scheduler.step()
            latest_metrics = estimate_loss(
                model,
                training_data,
                validation_data,
                batch_size=args.batch_size,
                evaluation_batches=args.evaluation_batches,
                device=device,
                generator=generator,
            )
            print(
                f"step={step:05d} train_loss={latest_metrics['train']:.4f} "
                f"validation_loss={latest_metrics['validation']:.4f}"
            )

        inputs, targets = sample_language_model_batch(
            training_data,
            args.batch_size,
            args.block_size,
            device=device,
            generator=generator,
        )
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.output,
        model,
        tokenizer,
        training_metadata={
            "steps": args.steps,
            "seed": args.seed,
            "final_metrics": latest_metrics,
        },
    )
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

