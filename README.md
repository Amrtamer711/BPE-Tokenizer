# BPE Tokenizer

A custom implementation of a Byte Pair Encoding (BPE) tokenizer for text processing. This tokenizer supports creating and saving tokenization vocabularies, encoding and decoding text, and padding sequences for batch processing.


## Features
- **Trainable Tokenizer**: Train a BPE tokenizer from raw text with a specified vocabulary size.
- **Encoding and Decoding**: Efficiently tokenize and detokenize text.
- **Padding and Attention Masks**: Handle batch processing with padding and attention masks.
- **Save and Load Vocabulary**: Save the vocabulary and merges to files, and reload them later.
- **UTF-8 Compatibility**: Supports UTF-8 encoding for non-ASCII characters.
