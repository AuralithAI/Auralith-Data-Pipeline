"""Tokenization module for text and multimodal content."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterator
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizedSample:
    """A tokenized sample ready for training."""
    
    input_ids: List[int]
    attention_mask: List[int]
    metadata: Dict[str, Any]
    
    @property
    def length(self) -> int:
        return len(self.input_ids)


class Tokenizer:
    """Text tokenizer using SentencePiece or HuggingFace tokenizers."""
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 50257,
        max_length: int = 2048,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ):
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.truncation = truncation
        
        self._tokenizer = None
        self._use_hf = False
    
    def _load_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is not None:
            return
        
        if self.tokenizer_path:
            # Try SentencePiece first
            if self.tokenizer_path.endswith(".model"):
                try:
                    import sentencepiece as spm
                    self._tokenizer = spm.SentencePieceProcessor()
                    self._tokenizer.Load(self.tokenizer_path)
                    self._use_hf = False
                    logger.info(f"Loaded SentencePiece tokenizer from {self.tokenizer_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load SentencePiece: {e}")
            
            # Try HuggingFace tokenizer
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                self._use_hf = True
                logger.info(f"Loaded HuggingFace tokenizer from {self.tokenizer_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace tokenizer: {e}")
        
        # Default to GPT-2 tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._use_hf = True
            logger.info("Using default GPT-2 tokenizer")
        except ImportError:
            logger.warning("transformers not installed, using basic tokenization")
            self._tokenizer = None
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        self._load_tokenizer()
        
        if self._tokenizer is None:
            # Basic fallback tokenization
            return [hash(word) % self.vocab_size for word in text.split()]
        
        if self._use_hf:
            return self._tokenizer.encode(
                text,
                add_special_tokens=self.add_special_tokens,
                truncation=self.truncation,
                max_length=self.max_length,
            )
        else:
            # SentencePiece
            ids = self._tokenizer.Encode(text)
            if self.truncation and len(ids) > self.max_length:
                ids = ids[:self.max_length]
            return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        self._load_tokenizer()
        
        if self._tokenizer is None:
            return ""
        
        if self._use_hf:
            return self._tokenizer.decode(ids, skip_special_tokens=True)
        else:
            return self._tokenizer.Decode(ids)
    
    def tokenize(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenizedSample:
        """Tokenize text into a training sample."""
        input_ids = self.encode(text)
        
        # Padding
        if self.padding and len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            input_ids = input_ids + [0] * pad_length
        else:
            attention_mask = [1] * len(input_ids)
        
        return TokenizedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            metadata=metadata or {},
        )
    
    def batch_tokenize(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TokenizedSample]:
        """Tokenize a batch of texts."""
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        return [
            self.tokenize(text, meta)
            for text, meta in zip(texts, metadata_list)
        ]


class TokenizationPipeline:
    """Pipeline for tokenizing data samples."""
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 50257,
        max_length: int = 2048,
        chunk_overlap: int = 128,
    ):
        self.tokenizer = Tokenizer(
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            max_length=max_length,
        )
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap
    
    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[TokenizedSample]:
        """Process text into tokenized samples, chunking if necessary."""
        from auralith_pipeline.sources.data_sources import DataSample
        
        # Get initial tokens to check length
        ids = self.tokenizer.encode(text)
        
        if len(ids) <= self.max_length:
            # Single sample
            yield self.tokenizer.tokenize(text, metadata)
        else:
            # Need to chunk
            for chunk in self._chunk_text(text):
                yield self.tokenizer.tokenize(chunk, metadata)
    
    def _chunk_text(self, text: str) -> Iterator[str]:
        """Chunk text for long documents."""
        words = text.split()
        
        # Estimate words per chunk based on average token per word ratio
        words_per_chunk = int(self.max_length * 0.7)  # Conservative estimate
        overlap_words = int(self.chunk_overlap * 0.7)
        
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk = " ".join(words[start:end])
            yield chunk
            
            if end >= len(words):
                break
            
            start = end - overlap_words
