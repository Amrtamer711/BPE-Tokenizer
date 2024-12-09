# BPE Tokenizer

![image](https://github.com/user-attachments/assets/8b1f4ba2-12b3-4ca0-8e54-8ff0dab4be4b)


A custom implementation of a Byte Pair Encoding (BPE) tokenizer for text processing. This tokenizer supports creating and saving tokenization vocabularies, encoding and decoding text, and padding sequences for batch processing.


## Features

- **Trainable Tokenizer**: Train a BPE tokenizer from raw text with a specified vocabulary size.
- **Encoding and Decoding**: Efficiently tokenize and detokenize text.
- **Padding and Attention Masks**: Handle batch processing with padding and attention masks.
- **Save and Load Vocabulary**: Save the vocabulary and merges to files, and reload them later.
- **UTF-8 Compatibility**: Supports UTF-8 encoding for non-ASCII characters.

# Usage
### 1. Train & Save Tokenizer
```python
from Tokenizer import BPE_Tokenizer

text = "This is an example sentence to demonstrate the BPE tokenizer."
vocab_size = 1000

tokenizer = BPE_Tokenizer(text=text, vocab_size=vocab_size)
tokenizer.save_vocab(directory="path_to_save_vocab")
```

### 2. Load Tokenizer

```python
tokenizer = BPE_Tokenizer(directory="path_to_vocab")
```

### 3. Tokenize Text
```python
encoded = tokenizer.encode("This is a test.", max_length=10)
```

### 4. Decode Tokenized Text

```python
decoded = tokenizer.decode(encoded)
```

# Implementation

This tokenizer implements padding by having the last token of the trained vocabulary be a pad token. The tokenizer creates a ```vocab.txt``` file and ```merges.txt``` file once it trains the vocabulary to store it. It also stores a special tokens mapping in ```special.json```

# Future Work

Currently working on applying special tokens to the tokenizer
