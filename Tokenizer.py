import regex as re
import numpy as np
import torch
import base64
import json
from collections import defaultdict


class BPE_Tokenizer:
    def __init__(self, text=None, vocab_size=None, directory=None, logging=None):
        self.vocab = {}
        self.merges = {}
        self.padding_idx = None  # We will set this to vocab_size + 1 when the vocab is finalized
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

    def load_vocab(self, directory):
        # Load vocab and decode Base64-encoded strings back to bytes
        with open(f"{directory}/vocab.txt", "r") as f:
            vocab_serializable = json.load(f)
        self.vocab = {int(key): base64.b64decode(value) for key, value in vocab_serializable.items()}

        # Load merges and convert JSON lists back to tuples and ints
        with open(f"{directory}/merges.txt", "r") as f:
            merges_serializable = json.load(f)
        self.merges = {tuple([int(k) for k in json.loads(key)]): int(value) for key, value in merges_serializable.items()}

        # Set the padding token as vocab_size + 1
        self.padding_idx = max(self.vocab.keys()) + 1

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

    def create_vocab(self, text, vocab_size, logging=None):
        # If the vocabulary already exists, resume the process
        if self.vocab:
            print(f"Resuming vocabulary creation from size {len(self.vocab)}...")
            current_size = len(self.vocab)
        else:
            # If the vocabulary does not exist, initialize it
            current_size = 256
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            self.merges = {}

        num_merges = vocab_size - current_size

        tokens = re.findall(self.text_regex, text)
        
        # Precompute UTF-8 encodings and store them
        ids = [np.frombuffer(token.encode("utf-8"), dtype=np.uint8) for token in tokens]
        
        # Continue merging process from current_size
        for i in range(current_size, current_size + num_merges):
            stats = defaultdict(int)
            for chunk_ids in ids:
                chunk_stats = self.get_stats(chunk_ids)
                for pair, count in chunk_stats.items():
                    stats[pair] += count
            
            # Find the most frequent pair
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = i
            
            # Merge the most frequent pair across all chunks
            ids = [self.merge(chunk_ids, pair, new_id) for chunk_ids in ids]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if logging:
                print(f"New token created: {self.vocab[new_id]}")

        # Finalize the padding token to vocab_size + 1
        self.padding_idx = max(self.vocab.keys()) + 1

        print(f"Vocabulary creation completed with size {len(self.vocab)}. Padding token set to index {self.padding_idx}.")

    def encode(self, text, max_length=None, return_attention_mask=True):
        """Encode new text(s) based on the current state of the vocabulary, with optional padding and attention masks."""
        # Check if input is a list with only one string or a single string
        is_single_text = isinstance(text, str) or (isinstance(text, list) and len(text) == 1)

        # If it's a single string, wrap it in a list to handle uniformly
        if isinstance(text, str):
            text = [text]

        tokenized_batch = []
        for t in text:
            tokens = re.findall(self.text_regex, t)
            unicode_streams = [np.frombuffer(token.encode("utf-8"), dtype=np.uint8) for token in tokens]
            merged_streams = []

            for token_stream in unicode_streams:
                while len(token_stream) >= 2:
                    stats = self.get_stats(token_stream)
                    pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
                    if pair not in self.merges:
                        break
                    id = self.merges[pair]
                    token_stream = self.merge(token_stream, pair, id)
                merged_streams.append(token_stream)

            tokenized_stream = [int(item) for sublist in merged_streams for item in sublist]
            tokenized_batch.append(tokenized_stream)

        # Only pad if it's a batch (multiple texts) or max_length is explicitly specified
        if (not is_single_text or max_length is not None):
            if max_length is None:
                max_length = max(len(seq) for seq in tokenized_batch)  # Longest sequence length
            tokenized_batch = self.pad_sequences(tokenized_batch, max_length)

        # Generate attention mask if required
        attention_masks = None
        if return_attention_mask:
            attention_masks = self.create_attention_masks(tokenized_batch)

        # Convert to tensors
        tokenized_batch_tensor = torch.tensor(tokenized_batch)
        attention_masks_tensor = torch.tensor(attention_masks) if attention_masks is not None else None

        # If a single text was passed, return only the first result (unbatch the result)
        if is_single_text:
            if return_attention_mask:
                return {'input_ids': tokenized_batch_tensor[0], 'attention_mask': attention_masks_tensor[0]}
            return {'input_ids': tokenized_batch_tensor[0]}

        # Return tokenized batch and attention masks as tensors in a dictionary
        return {'input_ids': tokenized_batch_tensor, 'attention_mask': attention_masks_tensor} if return_attention_mask else {'input_ids': tokenized_batch_tensor}

    def pad_sequences(self, sequences, max_length):
        """Pads all tokenized sequences in a batch to the specified max_length, using the padding index."""
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [self.padding_idx] * (max_length - len(seq))  # Pad with the padding_idx
            padded_sequences.append(padded_seq)
        return padded_sequences

    def create_attention_masks(self, sequences):
        """Creates attention masks for each sequence in a batch (1 for token, 0 for padding)."""
        attention_masks = []
        for seq in sequences:
            # Create a mask of 1s for all tokens that are not the padding_idx, 0 otherwise
            mask = [1 if token != self.padding_idx else 0 for token in seq]
            attention_masks.append(mask)
        return attention_masks

    def decode(self, encoded_input):
        """Decode token IDs from the 'input_ids' key into text, ignoring the padding token."""
        # Extract the 'input_ids' from the encoded input dictionary (handles tensors)
        ids = encoded_input['input_ids']

        # If input is a tensor, convert it to a list of integers
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # If a single list of ids is passed, wrap it into a list for batch handling
        is_single_ids = isinstance(ids[0], int)
        if is_single_ids:
            ids = [ids]  # Wrap it in a list to handle uniformly

        decoded_texts = []
        for seq in ids:
            # Ignore padding token
            decoded_tokens = [id for id in seq if id != self.padding_idx]
            stream = b"".join(self.vocab[id] for id in decoded_tokens)
            decoded_texts.append(stream.decode("utf-8", errors="replace"))

        # If a single sequence was passed, return only the first result (unbatch the result)
        if is_single_ids:
            return decoded_texts[0]

        return decoded_texts

    def save_vocab(self, directory):
        # Convert vocab bytes to Base64-encoded string
        vocab_serializable = {key: base64.b64encode(value).decode('utf-8') for key, value in self.vocab.items()}
        
        # Convert tuple keys and uint8 values in merges to lists and ints
        merges_serializable = {json.dumps([int(k) for k in key]): int(value) for key, value in self.merges.items()}

        with open(f"{directory}/vocab.txt", "w") as f:
            json.dump(vocab_serializable, f)

        with open(f"{directory}/merges.txt", "w") as f:
            json.dump(merges_serializable, f)