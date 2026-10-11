"""Generate text from a self-contained character GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from transformer_to_gpt.gpt import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--prompt",
        help="initial text; when omitted, use token zero as in the notebook",
    )
    parser.add_argument("--tokens", type=int, default=10_000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, help="optional UTF-8 output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=device)
    initial_tokens = [0] if args.prompt is None else tokenizer.encode(args.prompt)
    if not initial_tokens:
        raise ValueError("prompt must not be empty")
    prompt = torch.tensor([initial_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"saved generated text to {args.output}")


if __name__ == "__main__":
    main()

