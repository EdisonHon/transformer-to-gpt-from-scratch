"""Download the text corpus referenced by the archived GPT notebook."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gdown


COURSEWORK_FILE_ID = "1RlmRmXiWVKpZq98ftdtOIdM2lsA1uw3j"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hemingway.txt"),
        help="local destination (ignored by Git)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(id=COURSEWORK_FILE_ID, output=str(args.output), quiet=False)
    if downloaded is None or not args.output.is_file():
        raise RuntimeError("the coursework corpus could not be downloaded")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
