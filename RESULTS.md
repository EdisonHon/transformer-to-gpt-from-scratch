# Results

This document reports fresh runs of both parts of the consolidated project: the encoder-decoder Transformer copy task and the decoder-only character GPT. The runs were completed on October 11, 2026, using Python 3.14.5, PyTorch 2.14.1 with CUDA 13.0, and an NVIDIA GeForce RTX 4080 SUPER.

## 1. Encoder-decoder Transformer

### Task

The model receives a synthetic sequence containing start and end tokens and learns to reproduce it. This exercises encoder self-attention, masked decoder self-attention, encoder-decoder cross-attention, sinusoidal positions, and greedy autoregressive decoding.

The original coursework used 100 epochs as a short demonstration and explicitly noted that exact copying was not expected. The consolidated verification run retained the architecture and data generator but used 10,000 epochs so that held-out behavior could be measured meaningfully.

### Configuration

| Setting | Value |
| --- | ---: |
| Parameters | 185,188 |
| Vocabulary size | 100 |
| Model dimension | 64 |
| Attention heads | 2 |
| Encoder / decoder layers | 2 / 2 |
| Sequence length | 10 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 10,000 |
| Evaluation sequences | 320 |
| Seed | 1337 |
| Runtime | 99.42 seconds |

### Outcome

| Metric | Result |
| --- | ---: |
| Initial sampled training loss | 4.7284 |
| Final sampled training loss | 0.8589 |
| Held-out teacher-forced loss | 0.4233 |
| Held-out teacher-forced token accuracy | 83.96% |
| Greedy exact-sequence accuracy | 15.63% |

One held-out example was:

```text
Input:      [1, 53, 10, 18, 12, 73, 72, 34, 28, 2]
Prediction: [1, 53, 18, 10, 73, 12, 72, 34, 28, 2]
```

The loss reduction and token accuracy show that the model learned strong copying behavior. The lower exact-sequence accuracy exposes a useful limitation: teacher-forced token predictions are easier than fully autoregressive decoding, where one error changes the context for later positions. The model often preserved the correct token set while swapping nearby positions.

Full machine-readable metrics are in [`results/copy_task_metrics.json`](results/copy_task_metrics.json).

## 2. Decoder-only character GPT

### Task and data

The model predicts the next character from up to 128 preceding characters. The corpus was downloaded using the exact Google Drive file ID embedded in the archived notebook.

| Data property | Value |
| --- | ---: |
| Local filename | `hemingway.txt` |
| File size | 133,232 bytes |
| Decoded characters | 133,230 |
| Vocabulary size | 62 |
| SHA-256 | `edd96c922264c2ac38e40ff25e42f9caae917d10c417463df79ff20977db2bf4` |
| Split | First 90% train, final 10% validation |

The corpus itself is not committed because its original provenance and redistribution terms are not documented.

### Configuration

| Setting | Value |
| --- | ---: |
| Parameters | 693,758 |
| Model dimension | 96 |
| Attention heads | 6 |
| Decoder blocks | 6 |
| Context length | 128 |
| Batch size | 64 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Training steps | 5,000 |
| Evaluation interval / batches | 500 / 200 |
| Seed | 1337 |
| Runtime | 209.86 seconds |

### Loss curve

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.3103 | 4.3113 |
| 500 | 2.1631 | 2.1695 |
| 1,000 | 1.8817 | 1.8768 |
| 1,500 | 1.6821 | 1.6893 |
| 2,000 | 1.5596 | 1.5860 |
| 2,500 | 1.4774 | 1.5156 |
| 3,000 | 1.4103 | 1.4649 |
| 3,500 | 1.3675 | 1.4311 |
| 4,000 | 1.3231 | 1.4042 |
| 4,500 | 1.2962 | 1.3861 |
| 4,999 | **1.2705** | **1.3689** |

### Comparison with the archived notebook

| Run | Final train loss | Final validation loss |
| --- | ---: | ---: |
| Archived coursework output | 1.2677 | 1.3653 |
| Consolidated reproduction | 1.2705 | 1.3689 |
| Absolute difference | 0.0028 | 0.0036 |

The close agreement supports that the external corpus, tokenizer, architecture, optimizer, scheduler, evaluation routine, and random-number sequence were reproduced correctly. Small differences are expected across PyTorch, CUDA, and hardware versions.

### Generated output

Starting from token ID zero, the trained model sampled 10,000 new characters. An excerpt is shown below without correction:

```text
"Eat against sandil it was a beencale see him and I have had left fear
clent luck and a line wase in the sky of Freect I crabbes a to seeth in
the malrhops. His morich boat not more has for beforab that against hanp
proled gren. The shark cat you right."
```

The output learns dialogue-like punctuation, paragraph structure, recurring maritime vocabulary, and short-range English patterns. It also contains invented words, grammatical failures, repetition, and weak long-range coherence, which are expected from a small character-level model trained on a short corpus.

The complete unedited sample is stored in [`results/generated_hemingway.txt`](results/generated_hemingway.txt), and the full training history is in [`results/gpt_training_metrics.json`](results/gpt_training_metrics.json). The trained checkpoint is excluded from Git because generated weight files are large and environment-specific.

## Reproduction commands

```bash
.venv\Scripts\python scripts/download_coursework_corpus.py
.venv\Scripts\python scripts/train_copy_task.py --epochs 10000 --log-interval 100
.venv\Scripts\python scripts/train_character_gpt.py data/hemingway.txt --steps 5000
.venv\Scripts\python scripts/generate_text.py checkpoints/character_gpt.pt --tokens 10000 --output results/generated_hemingway.txt
```

CUDA was used for the reported runs. The scripts automatically fall back to CPU, but runtime and small floating-point differences will change.
