# Data

The character-level GPT requires a local UTF-8 text corpus. The original coursework downloaded a Hemingway corpus from an external Google Drive link, but its provenance and redistribution terms are not documented. That file is therefore not included in this repository and the training script does not download it automatically.

To train the model, provide a text file that you are permitted to use:

```bash
python scripts/train_character_gpt.py path/to/corpus.txt
```

Local corpus files are ignored by Git. The tokenizer vocabulary is derived deterministically by sorting the unique characters in the supplied text.

