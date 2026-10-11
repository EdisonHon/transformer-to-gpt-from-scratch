# From Transformer to GPT: Building Attention Architectures from Scratch

A from-scratch PyTorch implementation of the Transformer architecture, progressing from encoder-decoder attention to a decoder-only character-level GPT.

## Project information

| Field | Details |
| --- | --- |
| Courses | COMS 4701 Artificial Intelligence and COMS 4705 Natural Language Processing |
| Institution | Columbia University |
| Term | Spring 2026 |
| Project type | Refactored individual coursework |
| Stack | Python 3.14, PyTorch |

## Overview

This repository combines two related coursework implementations into one learning progression:

1. Build the attention primitives and a complete encoder-decoder Transformer.
2. Retain the decoder's causal self-attention and remove encoder cross-attention.
3. Train the resulting decoder-only architecture as a character-level GPT language model.

The goal is architectural understanding rather than benchmark performance. The code makes masks, projections, residual connections, normalization, and autoregressive generation explicit instead of wrapping PyTorch's built-in Transformer modules.

## Architecture progression

| Stage | Encoder-decoder Transformer | Character GPT |
| --- | --- | --- |
| Input | Source and shifted target sequences | A single character sequence |
| Attention | Encoder self-attention, causal decoder self-attention, and cross-attention | Causal self-attention only |
| Position representation | Fixed sinusoidal encoding | Learned position embeddings |
| Prediction objective | Next target token conditioned on the source | Next character conditioned on prior characters |
| Demonstration | Synthetic sequence copy task | User-provided UTF-8 text corpus |

Both models share the same explicit scaled dot-product attention primitive. Boolean masks use `True` for visible keys, which keeps padding and causal-mask behavior consistent across the project.

## Repository structure

```text
.
|-- data/
|   `-- README.md
|-- results/
|   |-- copy_task_metrics.json
|   |-- generated_hemingway.txt
|   `-- gpt_training_metrics.json
|-- scripts/
|   |-- download_coursework_corpus.py
|   |-- generate_text.py
|   |-- train_character_gpt.py
|   `-- train_copy_task.py
|-- src/transformer_to_gpt/
|   |-- attention.py
|   |-- data.py
|   |-- gpt.py
|   `-- transformer.py
|-- tests/
|-- .gitignore
|-- pyproject.toml
|-- RESULTS.md
`-- README.md
```

Local coursework instructions, context notes, archived submissions, datasets, and checkpoints are intentionally excluded from version control.

## Setup

Python 3.14 is the supported runtime.

```bash
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev,reproduction]"
```

Run the test suite:

```bash
.venv\Scripts\python -m pytest
```

## Encoder-decoder copy task

The copy task provides a small, self-contained check that information can flow through the encoder, cross-attention, and autoregressive decoder.

```bash
.venv\Scripts\python scripts/train_copy_task.py --epochs 10000 --log-interval 100
```

The script generates synthetic token sequences, trains with teacher forcing, evaluates held-out sequences, and records one greedy-decoded example. The reported metrics are an educational verification rather than a general benchmark.

## Character-level GPT

The training corpus is not included. Reproduce the notebook's explicit download, or supply another local UTF-8 text file that you are permitted to use:

```bash
.venv\Scripts\python scripts/download_coursework_corpus.py
```

```bash
.venv\Scripts\python scripts/train_character_gpt.py path/to/corpus.txt
```

The resulting checkpoint contains the model configuration, tokenizer vocabulary, weights, and training metadata. Generate text from that self-contained checkpoint with:

```bash
.venv\Scripts\python scripts/generate_text.py checkpoints/character_gpt.pt --tokens 10000 --output results/generated_hemingway.txt
```

See [`data/README.md`](data/README.md) for the data policy and reproducibility limitation.

## Results

Both model families were trained and evaluated on an RTX 4080 SUPER. The encoder-decoder run reached 83.96% teacher-forced token accuracy and 15.63% greedy exact-sequence accuracy on held-out synthetic sequences. The GPT reproduction reached train loss 1.2705 and validation loss 1.3689, closely matching the archived notebook's 1.2677 and 1.3653.

See [`RESULTS.md`](RESULTS.md) for configurations, loss curves, comparisons, interpretation, and an unedited generated-text sample.

## Improvements made during consolidation

- Unified mask semantics and corrected source-padding and target-causal masking.
- Added batched greedy decoding and restored model mode after generation.
- Supported odd positional-encoding dimensions safely.
- Replaced ambiguous standalone GPT weights with a self-contained checkpoint format for new runs.
- Added input validation and focused tests for attention, masking, model shapes, causality, tokenization, data batching, and checkpoint round trips.

These are organizational, refactoring, and correctness improvements to the coursework implementations. They do not add a new model family or present post-course research extensions.

## Limitations

- The notebook's external corpus is downloadable and its exact file hash is recorded, but its original publisher and redistribution terms are undocumented, so it is not committed.
- Character-level tokenization is intentionally simple and is not representative of modern subword tokenization.
- The implementations prioritize readability and architectural transparency over training efficiency.
- Recorded losses depend on the exact externally hosted corpus and should not be compared directly with runs on other text.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018.
