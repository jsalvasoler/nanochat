"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) (not here yet) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")
    
    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be a string or list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError("text must be a string or list of strings")
        
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)
    
    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
        
        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "if the first message is system, the second message must be user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        
        assert len(messages) > 0, "conversation must have at least one message"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # let's now tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"message {i} must be from {must_be_from}"

            # content can be a simple string or a list of parts (e.g. tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "user message content must be a string"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => wrap tokens
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => wrap tokens
                            add_tokens(output_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(output_end, 1)
                        else:
                            raise ValueError(f"unknown assistant content part type: {part['type']}")
                else:
                    raise ValueError("assistant message content must be a string or list of parts")
                add_tokens(assistant_end, 0)

        # truncate to max_tokens (helps prevent OOMs)
        return ids[:max_tokens], mask[:max_tokens]

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        raise NotImplementedError("render_for_completion is not implemented")

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes


if __name__ == "__main__":
    # quick test of the tokenizer
    tokenizer = RustBPETokenizer.from_pretrained("gpt2")
    test_text = """Hello world! This is a test."""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(tokenizer.visualize_tokenization(encoded, mask=[0]*len(encoded), with_token_id=True))
    assert decoded == test_text
    print("Tokenizer test passed!")