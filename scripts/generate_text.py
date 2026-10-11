"""Generate text from a self-contained character GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer_to_gpt.gpt import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=1_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=device)
    prompt = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long,
        device=device,
    )
    generated = model.generate(
        prompt,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

