# BPE Tokenizer

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

![image](https://github.com/user-attachments/assets/dda7fcc9-3e56-403b-8b91-b2db9c549045)


### 3. Tokenize Text

![image](https://github.com/user-attachments/assets/348eaa9a-6e29-49ca-a02c-7b0d75ff9184)


### 4. Decode Tokenized Text

![image](https://github.com/user-attachments/assets/4c7a5037-3868-4918-bc97-d5f536ba6482)

# Implementation

This tokenizer implements padding by having the last token of the trained vocabulary be a pad token. The tokenizer creates a ```vocab.txt``` file and ```merges.txt``` file once it trains the vocabulary to store it.

# Future Work

Currently working on applying special tokens to the tokenizer
