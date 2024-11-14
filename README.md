# BPE Tokenizer

A custom implementation of a Byte Pair Encoding (BPE) tokenizer for text processing. This tokenizer supports creating and saving tokenization vocabularies, encoding and decoding text, and padding sequences for batch processing.


## Features

- **Distributed Training (DDP)**: Easily scale training across multiple GPUs using torch.distributed.
- **LoRA/DoRA Fine-Tuning**: Use PEFT methods to efficiently fine-tune large language models with minimal trainable parameters.
- **Quantization Support**: Optimize memory usage with BitsAndBytesConfig for 4-bit quantization.
- **Custom Tokenization**: Tokenize and preprocess datasets for efficient causal language modeling.
- **Gradient Checkpointing**: Reduce VRAM usage during training.
- **Evaluation and Inference**: Comprehensive tools for validation, testing, and collecting results with metrics like accuracy and F1 score.
- **Custom Training and Dataloaders**: Fine-tuned dataloader creation for tokenized and non-tokenized datasets, and an adaptable training loop.

# Usage
### 1. Train & Save Tokenizer
![image](https://github.com/user-attachments/assets/f876fbbf-736b-4c8d-ae2e-5a78e42e99bf)

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
