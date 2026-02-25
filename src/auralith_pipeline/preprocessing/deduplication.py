"""Embedding-based deduplication using FAISS on top of MinHash.

Semantic near-duplicate detection using sentence embeddings + FAISS IVF index
for sub-linear search.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingDeduplicator:
    """Semantic deduplication using sentence embeddings + FAISS.

    Two-stage pipeline:
      1. MinHash (already in preprocessor.py) catches exact / near-exact dups.
      2. This class catches semantic duplicates that differ in wording
         but convey the same information.

    Requires: sentence-transformers, faiss-cpu (or faiss-gpu).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        nprobe: int = 10,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        """Initialize embedding deduplicator.

        Args:
            model_name: sentence-transformers model
            similarity_threshold: Cosine similarity above which two samples
                                  are considered duplicates (0–1)
            index_type: FAISS index type ('Flat', 'IVFFlat', 'IVFPQ')
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to probe at search time
            batch_size: Batch size for embedding computation
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.batch_size = batch_size
        self.device = device

        self._model = None
        self._index = None
        self._dim: int | None = None
        self._count = 0
        self._trained = False

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model {self.model_name} (dim={self._dim})")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — embedding dedup disabled. "
                "Install with: pip install sentence-transformers faiss-cpu"
            )

    def _build_index(self) -> None:
        """Build or rebuild the FAISS index."""
        if self._index is not None:
            return
        try:
            import faiss

            if self._dim is None:
                raise ValueError("Embedding dimension unknown — load model first")

            if self.index_type == "Flat":
                self._index = faiss.IndexFlatIP(self._dim)
            elif self.index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(self._dim)
                self._index = faiss.IndexIVFFlat(quantizer, self._dim, self.nlist)
                self._index.nprobe = self.nprobe
            elif self.index_type == "IVFPQ":
                quantizer = faiss.IndexFlatIP(self._dim)
                m = min(16, self._dim)
                self._index = faiss.IndexIVFPQ(quantizer, self._dim, self.nlist, m, 8)
                self._index.nprobe = self.nprobe
            else:
                self._index = faiss.IndexFlatIP(self._dim)

            logger.info(f"Built FAISS index: {self.index_type} (dim={self._dim})")
        except ImportError:
            logger.warning("faiss not installed — embedding dedup disabled")

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Compute normalized embeddings."""
        self._load_model()
        if self._model is None:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a semantic duplicate of anything in the index."""
        self._load_model()
        if self._model is None:
            return False

        self._build_index()
        if self._index is None:
            return False

        embedding = self._embed([text])
        if embedding.size == 0:
            return False

        # Need enough vectors to train IVF
        import faiss

        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            if self._count < self.nlist:
                # Can't search yet, just add
                self._index_add_raw(embedding)
                return False
            # Train the index with accumulated vectors
            self._train_index()

        if self._count == 0:
            self._index.add(embedding)
            self._count += 1
            return False

        # Search for nearest neighbor
        scores, _ = self._index.search(embedding, 1)
        max_sim = float(scores[0][0])

        # Add to index regardless
        self._index.add(embedding)
        self._count += 1

        return max_sim >= self.similarity_threshold

    def _index_add_raw(self, embedding: np.ndarray) -> None:
        """Add to a staging buffer for IVF training."""
        if not hasattr(self, "_staging"):
            self._staging: list[np.ndarray] = []
        self._staging.append(embedding)
        self._count += 1

    def _train_index(self) -> None:
        """Train IVF index from staging buffer."""
        if not hasattr(self, "_staging") or not self._staging:
            return

        import faiss

        all_vecs = np.concatenate(self._staging, axis=0)
        self._index.train(all_vecs)
        self._index.add(all_vecs)
        self._trained = True
        self._staging.clear()
        logger.info(f"Trained FAISS IVF index with {self._count} vectors")

    def batch_deduplicate(self, texts: list[str]) -> list[bool]:
        """Check a batch of texts for duplicates.

        Returns:
            List of booleans, True = duplicate, False = unique.
        """
        return [self.is_duplicate(t) for t in texts]

    def reset(self) -> None:
        """Reset the FAISS index."""
        self._index = None
        self._count = 0
        self._trained = False
        if hasattr(self, "_staging"):
            self._staging.clear()

    def save_index(self, path: str | Path) -> None:
        """Save FAISS index to disk."""
        if self._index is None:
            return
        try:
            import faiss

            faiss.write_index(self._index, str(path))
            logger.info(f"Saved FAISS index to {path}")
        except Exception as e:
            logger.warning(f"Failed to save FAISS index: {e}")

    def load_index(self, path: str | Path) -> None:
        """Load FAISS index from disk."""
        try:
            import faiss

            self._index = faiss.read_index(str(path))
            self._count = self._index.ntotal
            logger.info(f"Loaded FAISS index from {path} ({self._count} vectors)")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
