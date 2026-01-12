"""Custom Byte-Pair Encoding (BPE) tokenizer implementation from scratch.

This module implements a production-ready BPE tokenizer for text and multimodal data,
without any dependency on transformers or other tokenizer libraries.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer built from scratch.

    This tokenizer learns subword units through iterative merging of frequent
    character pairs, providing robust handling of rare words and OOV tokens.

    Features:
    - Custom BPE training on your corpus
    - Special token support (BOS, EOS, PAD, UNK, multimodal markers)
    - Character-level fallback for unknown words
    - Fast encoding with merge caching
    - UTF-8 Unicode support
    """

    # Special tokens (assigned IDs 0-9 for consistency)
    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,  # Beginning of sequence
        "<eos>": 3,  # End of sequence
        # Multimodal special tokens
        "<image_start>": 4,
        "<image_end>": 5,
        "<audio_start>": 6,
        "<audio_end>": 7,
        "<video_start>": 8,
        "<video_end>": 9,
    }

    # Word-end marker for BPE
    WORD_END = "</w>"

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        max_merge_depth: int = 50,
        lowercase: bool = False,
    ):
        """Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (default: 32,000)
            min_frequency: Minimum frequency for a pair to be merged
            max_merge_depth: Maximum number of consecutive merges per word
            lowercase: Whether to lowercase text during encoding
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_merge_depth = max_merge_depth
        self.lowercase = lowercase

        # Core BPE components (populated during training)
        self.vocab: dict[str, int] = {}  # token -> ID
        self.id_to_token: dict[int, str] = {}  # ID -> token
        self.merge_rules: list[tuple[str, str]] = []  # Ordered list of (pair1, pair2) merges

        # Cache for faster encoding
        self._word_cache: dict[str, list[str]] = {}  # word -> subword tokens

        # Initialize with special tokens
        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        self.vocab = self.SPECIAL_TOKENS.copy()
        self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}

    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<eos>"]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        if self.lowercase:
            text = text.lower()

        return text

    def _get_words(self, text: str) -> list[tuple[str, int]]:
        """Split text into words and count frequencies.

        Returns:
            List of (word, frequency) tuples
        """
        # Split on whitespace and punctuation, but keep punctuation as separate tokens
        word_pattern = re.compile(r"\w+|[^\w\s]")
        words = word_pattern.findall(text)

        # Count frequencies
        word_counts = Counter(words)

        return list(word_counts.items())

    def _get_char_vocab(self, words: list[tuple[str, int]]) -> set[str]:
        """Extract base character vocabulary from words."""
        chars = set()
        for word, _ in words:
            for char in word:
                chars.add(char)

        # Add word-end marker
        chars.add(self.WORD_END)

        return chars

    def _word_to_symbols(self, word: str) -> list[str]:
        """Convert word to list of symbols (characters + word-end marker)."""
        return list(word) + [self.WORD_END]

    def _get_pair_frequencies(
        self, word_symbols: dict[tuple[str, ...], int]
    ) -> Counter[tuple[str, str]]:
        """Count frequencies of all symbol pairs in words.

        Args:
            word_symbols: Dict mapping word symbol tuples to their frequencies

        Returns:
            Counter of (symbol1, symbol2) -> frequency
        """
        pair_counts: Counter[tuple[str, str]] = Counter()

        for symbols, freq in word_symbols.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq

        return pair_counts

    def _merge_pair(
        self,
        word_symbols: dict[tuple[str, ...], int],
        pair: tuple[str, str],
    ) -> dict[tuple[str, ...], int]:
        """Merge a specific pair in all words.

        Args:
            word_symbols: Current word representations
            pair: The pair to merge (a, b)

        Returns:
            Updated word_symbols with pair merged to 'ab'
        """
        new_word_symbols = {}
        merged_symbol = pair[0] + pair[1]

        for symbols, freq in word_symbols.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                # Check if we can merge at position i
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(merged_symbol)
                    i += 2  # Skip both elements of the pair
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            new_word_symbols[tuple(new_symbols)] = freq

        return new_word_symbols

    def train(self, corpus: str, verbose: bool = True) -> None:
        """Train BPE tokenizer on a text corpus.

        Args:
            corpus: Large text string to train on (ideally 1-10 GB)
            verbose: Whether to log training progress
        """
        if verbose:
            logger.info("Starting BPE tokenizer training...")
            logger.info(f"Target vocab size: {self.vocab_size}")

        # Step 1: Preprocess and split into words
        corpus = self._preprocess_text(corpus)
        words = self._get_words(corpus)

        if verbose:
            logger.info(f"Extracted {len(words)} unique words")

        # Step 2: Build initial character vocabulary
        char_vocab = self._get_char_vocab(words)

        # Add characters to vocab (after special tokens)
        next_id = max(self.vocab.values()) + 1
        for char in sorted(char_vocab):
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        if verbose:
            logger.info(f"Initial vocab size: {len(self.vocab)} (includes {len(char_vocab)} chars)")

        # Step 3: Initialize word representations as character sequences
        word_symbols: dict[tuple[str, ...], int] = {}
        for word, freq in words:
            symbols = tuple(self._word_to_symbols(word))
            word_symbols[symbols] = word_symbols.get(symbols, 0) + freq

        # Step 4: Iteratively learn merge rules
        merge_count = 0
        while len(self.vocab) < self.vocab_size:
            # Calculate pair frequencies
            pair_freqs = self._get_pair_frequencies(word_symbols)

            if not pair_freqs:
                if verbose:
                    logger.info("No more pairs to merge")
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            # Stop if frequency too low
            if best_freq < self.min_frequency:
                if verbose:
                    logger.info(
                        f"Stopping: best pair frequency {best_freq} < min {self.min_frequency}"
                    )
                break

            # Merge the pair
            word_symbols = self._merge_pair(word_symbols, best_pair)
            self.merge_rules.append(best_pair)

            # Add merged token to vocab
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = next_id
            self.id_to_token[next_id] = new_token
            next_id += 1

            merge_count += 1

            if verbose and merge_count % 1000 == 0:
                logger.info(f"Merges: {merge_count}, Vocab size: {len(self.vocab)}")

        if verbose:
            logger.info(f"Training complete! Final vocab size: {len(self.vocab)}")
            logger.info(f"Total merge rules: {len(self.merge_rules)}")

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        """Apply all merge rules to a symbol sequence.

        Args:
            symbols: List of symbols (characters initially)

        Returns:
            List of merged symbols (subwords)
        """
        # Apply merges in order (order matters!)
        for merge_depth, (a, b) in enumerate(self.merge_rules):
            if merge_depth >= self.max_merge_depth:
                break

            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode_word(self, word: str) -> list[int]:
        """Encode a single word to token IDs.

        Args:
            word: Word to encode

        Returns:
            List of token IDs
        """
        # Check cache first
        if word in self._word_cache:
            cached_tokens = self._word_cache[word]
            return [self.vocab.get(token, self.unk_token_id) for token in cached_tokens]

        # Convert to character symbols
        symbols = self._word_to_symbols(word)

        # Apply merge rules
        symbols = self._apply_merges(symbols)

        # Cache the result
        self._word_cache[word] = symbols

        # Convert to IDs (with fallback to <unk>)
        ids = []
        for symbol in symbols:
            if symbol in self.vocab:
                ids.append(self.vocab[symbol])
            else:
                # Character-level fallback for unknown symbols
                for char in symbol:
                    if char in self.vocab:
                        ids.append(self.vocab[char])
                    else:
                        ids.append(self.unk_token_id)

        return ids

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add <bos> and <eos> tokens
            max_length: Maximum sequence length (truncate if exceeded)

        Returns:
            List of token IDs
        """
        # Preprocess
        text = self._preprocess_text(text)

        # Split into words
        word_pattern = re.compile(r"\w+|[^\w\s]")
        words = word_pattern.findall(text)

        # Encode each word
        ids = []
        for word in words:
            ids.extend(self.encode_word(word))

        # Add special tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        tokens = []

        for token_id in ids:
            if token_id not in self.id_to_token:
                continue

            token = self.id_to_token[token_id]

            # Skip special tokens if requested
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue

            tokens.append(token)

        # Join tokens and handle word boundaries
        text = "".join(tokens)

        # Remove word-end markers and convert to spaces
        text = text.replace(self.WORD_END, " ")

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text.strip())

        return text

    def save(self, save_dir: str | Path) -> None:
        """Save tokenizer to directory.

        Args:
            save_dir: Directory to save tokenizer artifacts
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save vocab
        vocab_path = save_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merge rules
        merges_path = save_dir / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merge_rules:
                f.write(f"{a} {b}\n")

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "max_merge_depth": self.max_merge_depth,
            "lowercase": self.lowercase,
            "special_tokens": self.SPECIAL_TOKENS,
            "word_end_marker": self.WORD_END,
        }
        config_path = save_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Tokenizer saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str | Path) -> "BPETokenizer":
        """Load tokenizer from directory.

        Args:
            load_dir: Directory containing tokenizer artifacts

        Returns:
            Loaded BPETokenizer instance
        """
        load_dir = Path(load_dir)

        # Load config
        config_path = load_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Create tokenizer with config
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            max_merge_depth=config["max_merge_depth"],
            lowercase=config["lowercase"],
        )

        # Load vocab
        vocab_path = load_dir / "vocab.json"
        with open(vocab_path, encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)

        # Convert string keys to ints for id_to_token
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}

        # Load merge rules
        merges_path = load_dir / "merges.txt"
        tokenizer.merge_rules = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ", 1)
                    tokenizer.merge_rules.append((a, b))

        logger.info(f"Tokenizer loaded from {load_dir}")
        logger.info(f"Vocab size: {len(tokenizer.vocab)}")
        logger.info(f"Merge rules: {len(tokenizer.merge_rules)}")

        return tokenizer

    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)

    def train_from_file(self, file_path: str | Path, encoding: str = "utf-8") -> None:
        """Train tokenizer from a text file.

        Args:
            file_path: Path to text file
            encoding: File encoding (default: utf-8)
        """
        logger.info(f"Loading training corpus from {file_path}...")

        with open(file_path, encoding=encoding) as f:
            corpus = f.read()

        logger.info(f"Corpus size: {len(corpus)} characters")
        self.train(corpus)

    def train_from_files(
        self,
        file_paths: list[str | Path],
        max_size: int | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Train tokenizer from multiple text files.

        Args:
            file_paths: List of paths to text files
            max_size: Maximum corpus size in characters (for sampling)
            encoding: File encoding (default: utf-8)
        """
        logger.info(f"Loading training corpus from {len(file_paths)} files...")

        corpus_parts = []
        total_size = 0

        for file_path in file_paths:
            with open(file_path, encoding=encoding) as f:
                text = f.read()
                corpus_parts.append(text)
                total_size += len(text)

                if max_size and total_size >= max_size:
                    logger.info(f"Reached max corpus size: {max_size}")
                    break

        corpus = "\n".join(corpus_parts)

        # Sample if needed
        if max_size and len(corpus) > max_size:
            logger.info(f"Sampling corpus from {len(corpus)} to {max_size} characters")
            corpus = corpus[:max_size]

        logger.info(f"Final corpus size: {len(corpus)} characters")
        self.train(corpus)
