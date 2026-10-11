# Data

The character-level GPT requires a local UTF-8 text corpus. Cell 1 of the archived coursework notebook downloads `hemingway.txt` from Google Drive file ID `1RlmRmXiWVKpZq98ftdtOIdM2lsA1uw3j`. The external file remains available, but its original publisher and redistribution terms are not documented in the coursework materials. It is therefore downloaded only on explicit request and is never tracked by Git.

Install the reproduction dependency and retrieve the same external file:

```bash
.venv\Scripts\python -m pip install -e ".[reproduction]"
.venv\Scripts\python scripts/download_coursework_corpus.py
```

The file retrieved for the documented run was 133,232 bytes with SHA-256 `edd96c922264c2ac38e40ff25e42f9caae917d10c417463df79ff20977db2bf4`. Recording the hash distinguishes this run from any future change to the externally hosted file.

Alternatively, provide another text file that you are permitted to use:

```bash
python scripts/train_character_gpt.py path/to/corpus.txt
```

Local corpus files are ignored by Git. The tokenizer vocabulary is derived deterministically by sorting the unique characters in the supplied text.

The archived `fake_hemingway.txt` is not training data. The notebook creates it after training by sampling 10,000 new characters, then deletes it in the following cell.

