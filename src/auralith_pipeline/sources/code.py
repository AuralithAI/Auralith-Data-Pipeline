"""Code data sources — local directories and GitHub repositories.

Provides ``LocalCodeSource`` for ingesting code from a local directory
tree, and ``GitHubCodeSource`` for shallow-cloning a **public** GitHub
repository and iterating over its code files via ``LocalCodeSource``.

Both sources produce ``DataSample`` objects with ``modality="code"``
and rich metadata (language, line range, function/class name, etc.)
powered by :class:`~auralith_pipeline.preprocessing.code_chunker.CodeChunker`.

.. note::
   ``GitHubCodeSource`` only supports public repositories.  Training
   data pipelines should never ingest private/proprietary code — doing
   so risks leaking credentials and violating licence terms.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from auralith_pipeline.config.pipeline_config import CodeConfig
from auralith_pipeline.preprocessing.code_chunker import CodeChunker
from auralith_pipeline.sources.data_sources import DataSample, DataSource
from auralith_pipeline.utils.file_types import CODE_EXTS

logger = logging.getLogger(__name__)

# Directories that should always be skipped when walking a code tree.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "venv",
        "env",
        ".eggs",
        "dist",
        "build",
        ".next",
        "target",  # Rust / Java
        "vendor",  # Go / PHP
        "Pods",  # iOS
        ".gradle",
        ".idea",
        ".vscode",
    }
)


class LocalCodeSource(DataSource):
    """Iterate over code files in a local directory.

    Each file is chunked via :class:`CodeChunker` — AST-aware when
    tree-sitter is available, fixed-size otherwise — producing one
    ``DataSample`` per chunk.

    Parameters
    ----------
    root_dir:
        Root directory to walk.
    config:
        ``CodeConfig`` controlling chunk sizes, tree-sitter usage, etc.
    git_commit_hash:
        Optional Git commit hash to stamp on every chunk's metadata.
    repo_url:
        Optional repository URL to stamp on every chunk's metadata.
    max_samples:
        Stop after emitting this many samples (``None`` = unlimited).
    """

    def __init__(
        self,
        root_dir: str | Path,
        config: CodeConfig | None = None,
        git_commit_hash: str | None = None,
        repo_url: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.config = config or CodeConfig()
        self.git_commit_hash = git_commit_hash
        self.repo_url = repo_url
        self.max_samples = max_samples

        self._chunker = CodeChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            min_chunk_chars=self.config.min_chunk_chars,
            max_chunk_chars=self.config.max_chunk_chars,
            use_tree_sitter=self.config.use_tree_sitter,
        )

        # Pre-scan to get file count for __len__
        self._files: list[Path] = list(self._iter_code_files())

    # ── DataSource interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"code:{self.root_dir.name}"

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[DataSample]:
        emitted = 0
        for file_path in self._files:
            if self.max_samples is not None and emitted >= self.max_samples:
                return

            # Skip files over the size limit
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            if size > self.config.max_file_size_bytes:
                logger.debug("Skipping oversized file (%d bytes): %s", size, file_path)
                continue

            chunks = self._chunker.chunk_file(
                file_path,
                git_commit_hash=self.git_commit_hash,
                repo_url=self.repo_url,
            )

            for chunk in chunks:
                yield DataSample(
                    content=chunk.content,
                    source=str(file_path),
                    metadata=chunk.metadata.to_dict(),
                    modality="code",
                )
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return

    # ── helpers ───────────────────────────────────────────────────────

    def _iter_code_files(self) -> Iterator[Path]:
        """Walk the directory tree yielding code files."""
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Prune skipped directories *in-place* so os.walk doesn't descend
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in CODE_EXTS:
                    yield fpath


class GitHubCodeSource(DataSource):
    """Clone a public GitHub repository and iterate over its code files.

    Performs a shallow ``git clone --depth 1`` into a temporary directory
    (or a user-specified path) and delegates to :class:`LocalCodeSource`.

    Only **public** repositories are supported.  Training data pipelines
    should never ingest private/proprietary code.

    Use as a context manager to ensure cleanup of the temp clone::

        with GitHubCodeSource("https://github.com/org/repo") as src:
            for sample in src:
                ...

    Parameters
    ----------
    url:
        HTTPS GitHub URL (e.g. ``https://github.com/owner/repo``).
    ref:
        Branch / tag to clone (default: repository default branch).
    output_dir:
        Persistent clone destination.  When ``None`` a temp directory is
        used and cleaned up on context-manager exit.
    config:
        ``CodeConfig`` forwarded to the inner ``LocalCodeSource``.
    max_samples:
        Cap forwarded to the inner ``LocalCodeSource``.
    """

    def __init__(
        self,
        url: str,
        ref: str | None = None,
        output_dir: str | Path | None = None,
        config: CodeConfig | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.url = url
        self.ref = ref
        self.config = config or CodeConfig()
        self.max_samples = max_samples

        self._output_dir = Path(output_dir) if output_dir else None
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._inner: LocalCodeSource | None = None
        self._commit_hash: str | None = None

    # ── context manager ───────────────────────────────────────────────

    def __enter__(self) -> GitHubCodeSource:
        self._clone()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    # ── DataSource interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"github:{self.url.rstrip('/').split('/')[-1]}"

    def __len__(self) -> int:
        if self._inner is None:
            self._clone()
        assert self._inner is not None
        return len(self._inner)

    def __iter__(self) -> Iterator[DataSample]:
        if self._inner is None:
            self._clone()
        assert self._inner is not None
        yield from self._inner

    # ── cloning ───────────────────────────────────────────────────────

    def _clone(self) -> None:
        """Shallow-clone the repository."""
        if self._inner is not None:
            return  # already cloned

        # Decide target directory
        if self._output_dir is not None:
            clone_dir = self._output_dir
            clone_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="auralith_code_")
            clone_dir = Path(self._tmp_dir.name)

        cmd = ["git", "clone", "--depth", "1"]
        if self.ref:
            cmd += ["--branch", self.ref]
        cmd += [self.url, str(clone_dir)]

        logger.info("Cloning %s …", self.url)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git clone failed for {self.url}: {exc.stderr or ''}") from exc

        # Grab commit hash for metadata
        try:
            result = subprocess.run(
                ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            self._commit_hash = result.stdout.strip()
        except Exception:
            self._commit_hash = None

        self._inner = LocalCodeSource(
            root_dir=clone_dir,
            config=self.config,
            git_commit_hash=self._commit_hash,
            repo_url=self.url,
            max_samples=self.max_samples,
        )
        logger.info(
            "Cloned %s (%s) — %d code files found",
            self.url,
            self._commit_hash or "?",
            len(self._inner),
        )

    def cleanup(self) -> None:
        """Remove the temporary clone directory, if any."""
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except Exception as exc:
                logger.debug("temp dir cleanup failed: %s", exc)
            self._tmp_dir = None
        self._inner = None
