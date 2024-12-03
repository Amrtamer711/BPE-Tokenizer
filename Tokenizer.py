import regex as re
import numpy as np
import torch
import base64
import json
from collections import defaultdict


class BPE_Tokenizer:
    def __init__(self, text=None, vocab_size=None, directory=None, logging=None, 
                 special_tokens=None, notify_every=None, save_every=None, save_directory=None):
        self.vocab = {}
        self.merges = {}
        self.padding_idx = None
        self.special_tokens = special_tokens or {}
        self.notify_every = notify_every  # Optional parameter to notify every nth vocab
        self.save_every = save_every      # Save every nth vocab
        self.save_directory = save_directory  # Directory to save vocab checkpoints
        self.text_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        if (text is None and directory is None) or (text is not None and directory is not None):
            raise Exception("You can only pass either the model's tokenization vocabulary or the training text file.")

        if directory:
            self.load_vocab(directory)
        else:
            if not vocab_size:
                raise Exception("You must specify the tokenization vocabulary size.")
            else:
                self.create_vocab(text, vocab_size, logging)

    def create_vocab(self, text, vocab_size, logging=None):
        # Initialize with 256 basic UTF-8 tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        current_size = len(self.vocab)

        # Merge pairs until the target vocab size
        num_merges = vocab_size - current_size
        tokens = re.findall(self.text_regex, text)
        ids = [np.frombuffer(token.encode("utf-8"), dtype=np.uint8) for token in tokens]

        for i in range(current_size, current_size + num_merges):
            stats = defaultdict(int)
            for chunk_ids in ids:
                chunk_stats = self.get_stats(chunk_ids)
                for pair, count in chunk_stats.items():
                    stats[pair] += count

            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = i
            ids = [self.merge(chunk_ids, pair, new_id) for chunk_ids in ids]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Notify every nth vocab if notify_every is set
            if self.notify_every and new_id % self.notify_every == 0:  # Adjust for initial vocab size of 256
                print(f"Notification: Vocab ID {new_id} has been created.")

            if logging:
                print(f"New token created: {self.vocab[new_id]}")

            # Save vocabulary every n tokens if save_every and save_directory are set
            if self.save_every and self.save_directory and new_id % self.save_every == 0:
                self.save_vocab(self.save_directory)
                print(f"Checkpoint saved at vocab ID {new_id}.")

        # Add special tokens at the very end
        self.add_special_tokens()

    def add_special_tokens(self):
        """Adds special tokens to the vocabulary at the very end."""
        next_idx = len(self.vocab)  # Start from the current size of the vocabulary
        for token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<URL>', '<TITLE>']:
            if token not in self.special_tokens:
                self.special_tokens[token] = next_idx
                self.vocab[next_idx] = token.encode('utf-8')
                next_idx += 1

        # Set the padding index as vocab_size + 1
        self.padding_idx = next_idx  # The final index after all tokens
        self.special_tokens['<PAD>'] = self.padding_idx

    def load_vocab(self, directory):
        # Load vocab and decode Base64-encoded strings back to bytes
        with open(f"{directory}/vocab.txt", "r") as f:
            vocab_serializable = json.load(f)
        self.vocab = {int(key): base64.b64decode(value) for key, value in vocab_serializable.items()}

        # Load merges and convert JSON lists back to tuples and ints
        with open(f"{directory}/merges.txt", "r") as f:
            merges_serializable = json.load(f)
        self.merges = {tuple([int(k) for k in json.loads(key)]): int(value) for key, value in merges_serializable.items()}

        # Load special tokens if they exist
        special_tokens_file = f"{directory}/special_tokens.json"
        if os.path.exists(special_tokens_file):
            with open(special_tokens_file, "r") as f:
                self.special_tokens = json.load(f)

        # Set the padding token as vocab_size + 1
        self.add_special_tokens()

    def save_vocab(self, directory):
        # Convert vocab bytes to Base64-encoded string
        vocab_serializable = {key: base64.b64encode(value).decode('utf-8') for key, value in self.vocab.items()}

        # Convert tuple keys and uint8 values in merges to lists and ints
        merges_serializable = {json.dumps([int(k) for k in key]): int(value) for key, value in self.merges.items()}

        with open(f"{directory}/vocab.txt", "w") as f:
            json.dump(vocab_serializable, f)

        with open(f"{directory}/merges.txt", "w") as f:
            json.dump(merges_serializable, f)

        # Save special tokens
        with open(f"{directory}/special_tokens.json", "w") as f:
            json.dump(self.special_tokens, f)

    def get_stats(self, stream):
        """Counts the frequency of adjacent pairs in the token stream using numpy for efficiency."""
        counts = defaultdict(int)
        stream = np.array(stream)
        for i in range(len(stream) - 1):
            pair = (stream[i], stream[i + 1])
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, id):
        """Merges token pairs into a single token."""
        merged = []
        i = 0
        ids = np.array(ids)
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(id)
                i += 2  # Skip the next token since it's merged
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def encode(self, text, max_length=None, return_attention_mask=True, add_special_tokens=True):
        is_single_text = isinstance(text, str) or (isinstance(text, list) and len(text) == 1)
        if isinstance(text, str):
            text = [text]

        tokenized_batch = []
        for t in text:
            # Tokenize and encode in one pass
            tokens = [
                self.special_tokens.get(match.group(), np.frombuffer(match.group().encode('utf-8'), dtype=np.uint8).tolist())
                for match in re.finditer(self.text_regex, t)
            ]

            # Flatten list of tokens and add special tokens
            tokens = [item for sublist in tokens for item in (sublist if isinstance(sublist, list) else [sublist])]
            if add_special_tokens:
                tokens = [self.special_tokens['<BOS>']] + tokens + [self.special_tokens['<EOS>']]

            tokenized_batch.append(tokens)

        # Pad sequences and generate attention masks
        max_length = max_length or max(len(seq) for seq in tokenized_batch)
        tokenized_batch, attention_masks = self.pad_sequences(tokenized_batch, max_length)

        # Convert to tensors
        tokenized_batch_tensor = torch.tensor(tokenized_batch, dtype=torch.long)
        attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long) if return_attention_mask else None

        if is_single_text:
            return {
                'input_ids': tokenized_batch_tensor[0],
                'attention_mask': attention_masks_tensor[0] if attention_masks_tensor is not None else None
            }

        return {
            'input_ids': tokenized_batch_tensor,
            'attention_mask': attention_masks_tensor
        } if return_attention_mask else {'input_ids': tokenized_batch_tensor}

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        attention_masks = []
        for seq in sequences:
            padded_seq = seq + [self.padding_idx] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
            attention_masks.append([1] * len(seq) + [0] * (max_length - len(seq)))
        return padded_sequences, attention_masks

    def create_attention_masks(self, sequences):
        """Creates attention masks for each sequence in a batch (1 for token, 0 for padding)."""
        attention_masks = []
        for seq in sequences:
            # Create a mask of 1s for all tokens that are not the padding_idx, 0 otherwise
            mask = [1 if token != self.padding_idx else 0 for token in seq]
            attention_masks.append(mask)
        return attention_masks

    def decode(self, encoded_input, skip_special_tokens=True):
        """
        Decode token IDs into text, preserving trailing spaces and handling special tokens.

        Args:
            encoded_input (dict, torch.Tensor, or list): Encoded tokens or a dictionary containing 'input_ids'.
            skip_special_tokens (bool): Whether to skip special tokens during decoding.

        Returns:
            str or list of str: The decoded text. Returns a single string if input is a single sequence,
            otherwise returns a list of strings.
        """
        # Extract input IDs
        if isinstance(encoded_input, dict):  # If it's a dictionary
            ids = encoded_input['input_ids']
        else:
            ids = encoded_input

        # Convert tensor to list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Normalize input to batch format
        is_single_sequence = isinstance(ids[0], int)  # Check if it's a single sequence
        if is_single_sequence:
            ids = [ids]  # Wrap single sequence in a list for consistent batch processing

        decoded_texts = []
        for seq in ids:
            if skip_special_tokens:
                # Filter out special tokens
                seq = [id for id in seq if id not in self.special_tokens.values()]

            # Decode tokens
            decoded_tokens = [self.vocab[id] for id in seq if id != self.padding_idx]

            # Combine tokens while preserving spaces exactly as they were encoded
            stream = b"".join(decoded_tokens)
            decoded_text = stream.decode("utf-8", errors="replace")
            decoded_texts.append(decoded_text)

        # Return single string if input was a single sequence
        if is_single_sequence:
            return decoded_texts[0]
        return decoded_texts
