"""Train the decoder-only GPT on a user-provided UTF-8 text corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

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
    parser.add_argument("--early-stopping-tolerance", type=int, default=1)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.2)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/gpt_training_metrics.json"),
    )
    return parser.parse_args()


class EarlyStopping:
    """Match the relative train/validation gap rule used in the notebook."""

    def __init__(self, tolerance: int = 1, minimum_delta: float = 0.2) -> None:
        if tolerance <= 0 or minimum_delta < 0:
            raise ValueError("early-stopping parameters must be non-negative")
        self.tolerance = tolerance
        self.minimum_delta = minimum_delta
        self.counter = 0

    def should_stop(self, training_loss: float, validation_loss: float) -> bool:
        relative_gap = (validation_loss - training_loss) / training_loss
        if relative_gap > self.minimum_delta:
            self.counter += 1
        return self.counter >= self.tolerance


@torch.no_grad()
def estimate_loss(
    model: CharacterGPT,
    training_data: torch.Tensor,
    validation_data: torch.Tensor,
    *,
    batch_size: int,
    evaluation_batches: int,
    device: torch.device,
    generator: torch.Generator | None,
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
    # The notebook uses PyTorch's global generator for initialization and data
    # sampling. Keeping that sequence here makes the reproduction more faithful.
    generator = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        block_size=args.block_size,
    )
    model = CharacterGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    latest_metrics: dict[str, float] = {}
    history: list[dict[str, float | int]] = []
    early_stopping = EarlyStopping(
        args.early_stopping_tolerance,
        args.early_stopping_min_delta,
    )
    stopped_early = False
    started_at = time.perf_counter()

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
            history.append(
                {
                    "step": step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "training_loss": latest_metrics["train"],
                    "validation_loss": latest_metrics["validation"],
                }
            )
            if early_stopping.should_stop(
                latest_metrics["train"],
                latest_metrics["validation"],
            ):
                print(f"early stopping at step {step}")
                stopped_early = True
                break

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
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(
            {
                "task": "character-level language modeling",
                "device": str(device),
                "torch_version": torch.__version__,
                "seed": args.seed,
                "corpus_filename": args.corpus.name,
                "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                "corpus_characters": len(text),
                "vocabulary_size": tokenizer.vocabulary_size,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "configuration": {
                    **config.to_dict(),
                    "batch_size": args.batch_size,
                    "requested_steps": args.steps,
                    "evaluation_interval": args.evaluation_interval,
                    "evaluation_batches": args.evaluation_batches,
                    "initial_learning_rate": args.learning_rate,
                    "early_stopping_tolerance": args.early_stopping_tolerance,
                    "early_stopping_min_delta": args.early_stopping_min_delta,
                },
                "stopped_early": stopped_early,
                "completed_steps": history[-1]["step"] if history else 0,
                "elapsed_seconds": time.perf_counter() - started_at,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved checkpoint to {args.output}")
    print(f"saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()

