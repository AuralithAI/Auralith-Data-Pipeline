"""Data preprocessing module with deduplication, quality filtering, and PII removal."""

from dataclasses import dataclass
from typing import Iterator, Set, Optional, List, Callable
import re
import logging
import hashlib

from auralith_pipeline.sources.data_sources import DataSample
from auralith_pipeline.config.pipeline_config import QualityConfig, DeduplicationConfig

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Normalize text for consistent processing."""
    
    def __init__(self):
        # Common replacements
        self.replacements = [
            (r'\s+', ' '),  # Multiple spaces to single
            (r'\n{3,}', '\n\n'),  # Multiple newlines to double
            (r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ''),  # Control chars
        ]
    
    def normalize(self, text: str) -> str:
        """Normalize text."""
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            pass
        
        for pattern, replacement in self.replacements:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()


class QualityFilter:
    """Filter samples based on quality criteria."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self._lang_detector = None
    
    def _get_lang_detector(self):
        if self._lang_detector is None:
            try:
                from langdetect import detect
                self._lang_detector = detect
            except ImportError:
                self._lang_detector = lambda x: "en"
        return self._lang_detector
    
    def passes_filter(self, sample: DataSample) -> bool:
        """Check if sample passes quality filters."""
        text = sample.content
        
        # Length check
        if len(text) < self.config.min_text_length:
            return False
        if len(text) > self.config.max_text_length:
            return False
        
        # Word count
        words = text.split()
        word_count = len(words)
        if word_count < self.config.min_word_count:
            return False
        if word_count > self.config.max_word_count:
            return False
        
        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0 and special_chars / len(text) > self.config.max_special_char_ratio:
            return False
        
        # Digit ratio
        digits = sum(1 for c in text if c.isdigit())
        if len(text) > 0 and digits / len(text) > self.config.max_digit_ratio:
            return False
        
        # Uppercase ratio
        uppercase = sum(1 for c in text if c.isupper())
        alpha = sum(1 for c in text if c.isalpha())
        if alpha > 0 and uppercase / alpha > self.config.max_uppercase_ratio:
            return False
        
        # Language detection
        if self.config.allowed_languages:
            try:
                detect = self._get_lang_detector()
                lang = detect(text[:1000])  # Use first 1000 chars for speed
                if lang not in self.config.allowed_languages:
                    return False
            except Exception:
                pass  # If detection fails, allow the sample
        
        return True


class MinHashDeduplicator:
    """Deduplicate using MinHash LSH."""
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self._lsh = None
        self._seen_hashes: Set[str] = set()
        self._count = 0
    
    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Get k-shingles from text."""
        text = text.lower()
        words = text.split()
        if len(words) < k:
            return {text}
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}
    
    def _get_minhash(self, text: str):
        """Get MinHash signature."""
        try:
            from datasketch import MinHash
            
            shingles = self._get_shingles(text)
            m = MinHash(num_perm=self.config.minhash_num_perm)
            for s in shingles:
                m.update(s.encode('utf-8'))
            return m
        except ImportError:
            # Fallback to simple hash
            return hashlib.md5(text.encode()).hexdigest()
    
    def is_duplicate(self, sample: DataSample) -> bool:
        """Check if sample is a duplicate."""
        if not self.config.enabled:
            return False
        
        text = sample.content
        
        if self.config.method == "exact":
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._seen_hashes:
                return True
            self._seen_hashes.add(text_hash)
            return False
        
        elif self.config.method == "minhash":
            try:
                from datasketch import MinHashLSH
                
                if self._lsh is None:
                    self._lsh = MinHashLSH(
                        threshold=self.config.minhash_threshold,
                        num_perm=self.config.minhash_num_perm,
                    )
                
                minhash = self._get_minhash(text)
                
                # Check for similar documents
                result = self._lsh.query(minhash)
                if result:
                    return True
                
                # Add to index
                self._lsh.insert(f"doc_{self._count}", minhash)
                self._count += 1
                return False
                
            except ImportError:
                # Fallback to exact matching
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self._seen_hashes:
                    return True
                self._seen_hashes.add(text_hash)
                return False
        
        return False
    
    def reset(self):
        """Reset deduplication state."""
        self._lsh = None
        self._seen_hashes.clear()
        self._count = 0


class PIIRemover:
    """Remove Personally Identifiable Information."""
    
    def __init__(self):
        # PII patterns
        self.patterns = [
            # Email
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            # Phone numbers (various formats)
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
            (r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', '[PHONE]'),
            # SSN
            (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]'),
            # Credit card
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]'),
            # IP addresses
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]'),
            # URLs with credentials
            (r'https?://[^:]+:[^@]+@', 'https://[CREDENTIALS]@'),
        ]
    
    def remove_pii(self, text: str) -> str:
        """Remove PII from text."""
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(
        self,
        quality_config: Optional[QualityConfig] = None,
        dedup_config: Optional[DeduplicationConfig] = None,
        normalize: bool = True,
        remove_pii: bool = True,
    ):
        self.quality_config = quality_config or QualityConfig()
        self.dedup_config = dedup_config or DeduplicationConfig()
        
        self.normalizer = TextNormalizer() if normalize else None
        self.quality_filter = QualityFilter(self.quality_config)
        self.deduplicator = MinHashDeduplicator(self.dedup_config)
        self.pii_remover = PIIRemover() if remove_pii else None
        
        # Stats
        self.stats = {
            "total_processed": 0,
            "passed_quality": 0,
            "duplicates_removed": 0,
            "pii_removed": 0,
        }
    
    def process(self, samples: Iterator[DataSample]) -> Iterator[DataSample]:
        """Process samples through the preprocessing pipeline."""
        for sample in samples:
            self.stats["total_processed"] += 1
            
            # Normalize
            if self.normalizer:
                sample.content = self.normalizer.normalize(sample.content)
            
            # Quality filter
            if not self.quality_filter.passes_filter(sample):
                continue
            self.stats["passed_quality"] += 1
            
            # Deduplication
            if self.deduplicator.is_duplicate(sample):
                self.stats["duplicates_removed"] += 1
                continue
            
            # PII removal
            if self.pii_remover:
                original_len = len(sample.content)
                sample.content = self.pii_remover.remove_pii(sample.content)
                if len(sample.content) != original_len:
                    self.stats["pii_removed"] += 1
            
            yield sample
    
    def reset(self):
        """Reset preprocessor state."""
        self.deduplicator.reset()
        self.stats = {k: 0 for k in self.stats}
