"""Embedding-based deduplication using FAISS on top of MinHash.

Semantic near-duplicate detection using sentence embeddings + FAISS IVF index
for sub-linear search.
"""

import logging
from pathlib import Path

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
        # Buffer for vectors accumulated before IVF index can be trained.
        # FAISS recommends 10-40× nlist training vectors for good cluster quality.
        self._staging: list[np.ndarray] = []
        self._min_train_size: int = nlist * 10

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
        """Compute L2-normalized embeddings.

        Returns unit-length vectors so that ``dot(a, b)`` equals cosine
        similarity — the metric used by both the staging-buffer brute-force
        check and the FAISS ``IndexFlatIP`` / ``IndexIVFFlat`` index.

        We pass ``normalize_embeddings=True`` to the model *and* re-normalise
        afterwards so the invariant holds even if the underlying model ignores
        the flag or a future refactor changes the model backend.
        """
        self._load_model()
        if self._model is None:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Defensive L2-normalisation: guarantees unit-length vectors regardless
        # of whether the model actually honoured ``normalize_embeddings``.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        embeddings = embeddings / norms

        return embeddings

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a semantic duplicate of anything in the index.

        Pre-training phase (IVF not yet trained):
          Vectors accumulate in a staging buffer.  Deduplication uses brute-force
          dot-product against the buffer.  Once _min_train_size vectors are collected
          (nlist × 10 = 1000 by default), the IVF index is trained and the buffer
          is flushed into it.

        Post-training phase:
          Normal approximate nearest-neighbour search via FAISS.
        """
        self._load_model()
        if self._model is None:
            return False

        self._build_index()
        if self._index is None:
            return False

        embedding = self._embed([text])
        if embedding.size == 0:
            return False

        # IVF indexes require training before they can be searched.
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            # Brute-force cosine-similarity check within the pending buffer
            # (O(N) — N < _min_train_size ≈ 1000).  Safe because _embed()
            # guarantees L2-normalised vectors, so dot(a, b) == cos(a, b).
            for prev in self._staging:
                if float((prev @ embedding.T).squeeze()) >= self.similarity_threshold:
                    return True  # duplicate — do NOT add to buffer

            self._staging.append(embedding)
            self._count += 1

            # Trigger training once we have enough vectors for reliable clustering.
            if len(self._staging) >= self._min_train_size:
                self._train_index()

            return False

        # Index is trained — use approximate nearest-neighbour search.
        if self._count == 0:
            # Flat indexes report is_trained=True even when empty.
            self._index.add(embedding)
            self._count += 1
            return False

        scores, _ = self._index.search(embedding, 1)
        is_dup = float(scores[0][0]) >= self.similarity_threshold

        if not is_dup:
            self._index.add(embedding)
            self._count += 1

        return is_dup

    def _train_index(self) -> None:
        """Train IVF index from the staging buffer, then flush all staged vectors into it."""
        if not self._staging:
            return

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
