"""AST-aware code chunking for the code modality.

Splits source files into semantically meaningful chunks at function/class
boundaries using tree-sitter when available, with a fixed-size sliding
window fallback for languages without a grammar or when tree-sitter is
not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Language detection ────────────────────────────────────────────────────

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # ── Mainstream / general-purpose ───────────────────────────────
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    ".pxd": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".erb": "ruby",
    ".cs": "c_sharp",
    ".csx": "c_sharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".r": "r",
    ".rmd": "r",
    ".lua": "lua",
    ".zig": "zig",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".lhs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".fs": "fsharp",
    ".fsi": "fsharp",
    ".fsx": "fsharp",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".edn": "clojure",
    ".lisp": "commonlisp",
    ".cl": "commonlisp",
    ".el": "elisp",
    ".scm": "scheme",
    ".ss": "scheme",
    ".rkt": "racket",
    ".jl": "julia",
    ".nim": "nim",
    ".nims": "nim",
    ".cr": "crystal",
    ".v": "v",
    ".vh": "v",
    ".d": "d",
    ".pl": "perl",
    ".pm": "perl",
    ".groovy": "groovy",
    ".gvy": "groovy",
    ".gy": "groovy",
    # ── C / C++ family ────────────────────────────────────────────
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    ".m": "objc",
    ".mm": "objc",
    # ── Legacy / mainframe ────────────────────────────────────────
    ".cob": "cobol",
    ".cbl": "cobol",
    ".cpy": "cobol",
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".for": "fortran",
    ".ada": "ada",
    ".adb": "ada",
    ".ads": "ada",
    ".pas": "pascal",
    ".pp": "pascal",
    ".vb": "vb",
    ".bas": "vb",
    ".vbs": "vb",
    ".rpg": "rpg",
    ".rpgle": "rpg",
    ".abap": "abap",
    ".rexx": "rexx",
    ".rex": "rexx",
    ".clu": "clu",
    ".pro": "prolog",
    # ── GPU / HPC / shaders ───────────────────────────────────────
    ".cu": "cuda",
    ".cuh": "cuda",
    ".hlsl": "hlsl",
    ".glsl": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
    ".comp": "glsl",
    ".metal": "metal",
    ".wgsl": "wgsl",
    ".ptx": "ptx",
    # ── DevOps / IaC / build ──────────────────────────────────────
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".ksh": "bash",
    ".csh": "bash",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".hcl": "hcl",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".dockerfile": "dockerfile",
    ".nix": "nix",
    ".bzl": "starlark",
    ".starlark": "starlark",
    ".cmake": "cmake",
    ".just": "just",
    # ── Config-as-code ────────────────────────────────────────────
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "conf",
    ".env": "dotenv",
    ".properties": "properties",
    ".gradle": "groovy",
    ".sbt": "scala",
    ".cabal": "cabal",
    # ── Web / frontend ────────────────────────────────────────────
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".css": "css",
    ".scss": "scss",
    ".sass": "scss",
    ".less": "less",
    ".styl": "stylus",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".wasm": "wasm",
    ".wat": "wasm",
    # ── Data / science / query ────────────────────────────────────
    ".sql": "sql",
    ".psql": "sql",
    ".plsql": "sql",
    ".sas": "sas",
    ".do": "stata",
    ".ado": "stata",
    ".pig": "pig",
    # ── Scripting / text processing ───────────────────────────────
    ".awk": "awk",
    ".sed": "sed",
    ".tcl": "tcl",
    ".expect": "tcl",
    # ── Other notable languages ───────────────────────────────────
    ".sol": "solidity",
    ".move": "move",
    ".vy": "vyper",
    ".cairo": "cairo",
    ".elm": "elm",
    ".purs": "purescript",
    ".hack": "hack",
    ".hhi": "hack",
    ".lean": "lean",
    ".idr": "idris",
    ".agda": "agda",
    ".coq": "coq",
    ".sv": "systemverilog",
    ".svh": "systemverilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
    ".asm": "asm",
    ".s": "asm",
    ".ll": "llvm",
    ".proto": "protobuf",
    ".thrift": "thrift",
    ".capnp": "capnproto",
    ".fbs": "flatbuffers",
    ".smithy": "smithy",
    ".cue": "cue",
    ".dhall": "dhall",
    ".jsonnet": "jsonnet",
    ".libsonnet": "jsonnet",
    ".pkl": "pkl",
    ".rego": "rego",
    ".zeek": "zeek",
}

# AST node types that represent natural chunk boundaries per language.
# Languages not listed here still work — the fixed-size fallback handles them.
_CHUNK_NODE_TYPES: dict[str, set[str]] = {
    # ── Mainstream ────────────────────────────────────────────────
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
    "c": {"function_definition", "struct_specifier"},
    "ruby": {"method", "class", "module"},
    "c_sharp": {"method_declaration", "class_declaration", "interface_declaration"},
    "php": {"function_definition", "class_declaration", "method_declaration"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "kotlin": {"function_declaration", "class_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition"},
    "lua": {"function_declaration", "local_function"},
    "dart": {"function_signature", "class_definition", "method_signature"},
    "zig": {"fn_decl", "container_decl"},
    # ── Functional ────────────────────────────────────────────────
    "elixir": {"def", "defmodule", "defmacro", "defp"},
    "erlang": {"function", "attribute"},
    "haskell": {"function", "type_alias", "data_declaration"},
    "ocaml": {"let_binding", "type_definition", "module_definition"},
    "elm": {"function_declaration_left", "type_alias_declaration", "type_declaration"},
    "clojure": {"defn", "def", "defmacro"},
    # ── C / C++ family ────────────────────────────────────────────
    "objc": {"function_definition", "class_interface", "class_implementation", "method_definition"},
    "cuda": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
    # ── Legacy ────────────────────────────────────────────────────
    "fortran": {"function", "subroutine", "module", "program"},
    "pascal": {"function_declaration", "procedure_declaration", "class_declaration"},
    # ── DevOps / IaC ──────────────────────────────────────────────
    "bash": {"function_definition"},
    "hcl": {"block"},
    "yaml": {"block_mapping_pair"},
    # ── Web ───────────────────────────────────────────────────────
    "css": {"rule_set", "media_statement"},
    "scss": {"rule_set", "mixin_statement", "function_statement"},
    "graphql": {"type_definition", "input_type_definition", "enum_type_definition"},
    # ── Data / query ──────────────────────────────────────────────
    "sql": {"create_table_statement", "select_statement", "create_function_statement"},
    "julia": {"function_definition", "struct_definition", "macro_definition"},
    # ── Other ─────────────────────────────────────────────────────
    "solidity": {"function_definition", "contract_declaration", "event_definition"},
    "protobuf": {"message", "service", "enum"},
}


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every code chunk."""

    file_path: str
    language: str
    line_start: int
    line_end: int
    function_name: str | None = None
    class_name: str | None = None
    git_commit_hash: str | None = None
    repo_url: str | None = None
    chunk_index: int = 0
    total_chunks: int = 0

    def to_dict(self) -> dict[str, str | int | None]:
        """Serialise for storage in DataSample.metadata."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "git_commit_hash": self.git_commit_hash,
            "repo_url": self.repo_url,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


@dataclass
class CodeChunk:
    """A single code chunk with its metadata."""

    content: str
    metadata: ChunkMetadata


# ── Parser cache ──────────────────────────────────────────────────────────

_parser_cache: dict[str, object] = {}


def _get_parser(language: str) -> object | None:
    """Return a tree-sitter parser for *language*, or ``None``.

    Tries ``tree_sitter_languages`` first (single package with many
    grammars), then falls back to individual ``tree_sitter_{lang}``
    packages.  All exceptions are caught so the caller can fall back
    to fixed-size chunking.
    """
    if language in _parser_cache:
        return _parser_cache[language]

    parser = None
    try:
        from tree_sitter_languages import get_parser  # type: ignore[import-untyped]

        parser = get_parser(language)
    except Exception:
        try:
            import importlib

            lang_mod = importlib.import_module(f"tree_sitter_{language}")
            import tree_sitter  # type: ignore[import-untyped]

            parser = tree_sitter.Parser()
            parser.set_language(tree_sitter.Language(lang_mod.language()))
        except Exception:
            pass

    _parser_cache[language] = parser
    return parser


# ── Chunker ───────────────────────────────────────────────────────────────


class CodeChunker:
    """Language-aware code chunker.

    Parameters
    ----------
    chunk_size:
        Target chunk size **in tokens** (estimated as ``len(text) // 4``).
    overlap:
        Overlap between consecutive fixed-size chunks, in tokens.
    min_chunk_chars:
        Chunks shorter than this are discarded.
    max_chunk_chars:
        Oversized AST nodes are split via the fixed-size fallback.
    use_tree_sitter:
        Set ``False`` to always use the fixed-size fallback.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 256,
        min_chunk_chars: int = 50,
        max_chunk_chars: int = 16_000,
        use_tree_sitter: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.use_tree_sitter = use_tree_sitter

    # ── public API ────────────────────────────────────────────────────

    def detect_language(self, file_path: str | Path) -> str | None:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext)

    def chunk_file(
        self,
        file_path: str | Path,
        git_commit_hash: str | None = None,
        repo_url: str | None = None,
    ) -> list[CodeChunk]:
        """Chunk a source file into semantically meaningful pieces."""
        file_path = Path(file_path)
        language = self.detect_language(file_path)
        if language is None:
            return []

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return []

        if not source.strip():
            return []

        chunks: list[CodeChunk] = []

        if self.use_tree_sitter:
            chunks = self._chunk_with_tree_sitter(
                source,
                language,
                str(file_path),
                git_commit_hash,
                repo_url,
            )

        # Fall back to fixed-size if tree-sitter unavailable or yielded nothing
        if not chunks:
            chunks = self._chunk_fixed_size(
                source,
                language,
                str(file_path),
                git_commit_hash,
                repo_url,
            )

        # Stamp total_chunks on every metadata object
        for i, chunk in enumerate(chunks):
            chunk.metadata.chunk_index = i
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    # ── tree-sitter chunking ──────────────────────────────────────────

    def _chunk_with_tree_sitter(
        self,
        source: str,
        language: str,
        file_path: str,
        git_commit_hash: str | None,
        repo_url: str | None,
    ) -> list[CodeChunk]:
        parser = _get_parser(language)
        if parser is None:
            return []

        try:
            source_bytes = source.encode("utf-8")
            tree = parser.parse(source_bytes)
        except Exception as exc:
            logger.debug("tree-sitter parse failed for %s: %s", file_path, exc)
            return []

        boundary_types = _CHUNK_NODE_TYPES.get(language, set())
        if not boundary_types:
            return []

        chunks: list[CodeChunk] = []
        for child in tree.root_node.children:
            if child.type not in boundary_types:
                continue

            text = source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
            if len(text) < self.min_chunk_chars:
                continue

            fn_name, cls_name = self._extract_symbol_name(child, source_bytes)

            # If the node is too large, split via fixed-size
            if len(text) > self.max_chunk_chars:
                sub_chunks = self._chunk_fixed_size(
                    text,
                    language,
                    file_path,
                    git_commit_hash,
                    repo_url,
                    base_line=child.start_point[0] + 1,
                )
                # Carry over symbol names
                for sc in sub_chunks:
                    sc.metadata.function_name = sc.metadata.function_name or fn_name
                    sc.metadata.class_name = sc.metadata.class_name or cls_name
                chunks.extend(sub_chunks)
                continue

            meta = ChunkMetadata(
                file_path=file_path,
                language=language,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                function_name=fn_name,
                class_name=cls_name,
                git_commit_hash=git_commit_hash,
                repo_url=repo_url,
            )
            chunks.append(CodeChunk(content=text, metadata=meta))

        return chunks

    @staticmethod
    def _extract_symbol_name(
        node: object,
        source_bytes: bytes,
    ) -> tuple[str | None, str | None]:
        """Extract function and class names from an AST node."""
        fn_name: str | None = None
        cls_name: str | None = None

        node_type = getattr(node, "type", "")

        # Walk direct children looking for the identifier / name child
        for child in getattr(node, "children", []):
            child_type = getattr(child, "type", "")
            if child_type in ("identifier", "name", "property_identifier"):
                name = source_bytes[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="replace"
                )
                if "class" in node_type:
                    cls_name = name
                else:
                    fn_name = name
                break

        return fn_name, cls_name

    # ── fixed-size fallback ───────────────────────────────────────────

    def _chunk_fixed_size(
        self,
        source: str,
        language: str,
        file_path: str,
        git_commit_hash: str | None,
        repo_url: str | None,
        *,
        base_line: int = 1,
    ) -> list[CodeChunk]:
        """Sliding-window chunking with newline alignment."""
        # Convert token counts to approximate character counts (1 token ~ 4 chars)
        window_chars = self.chunk_size * 4
        overlap_chars = self.overlap * 4

        chunks: list[CodeChunk] = []
        start = 0
        source_len = len(source)

        while start < source_len:
            end = min(start + window_chars, source_len)

            # Snap to next newline to avoid splitting mid-line
            if end < source_len:
                nl = source.find("\n", end)
                if nl != -1 and nl - end < 200:  # don't overshoot too far
                    end = nl + 1

            chunk_text = source[start:end]
            if len(chunk_text.strip()) < self.min_chunk_chars:
                start = end
                continue

            line_start = base_line + source[:start].count("\n")
            line_end = line_start + chunk_text.count("\n")

            meta = ChunkMetadata(
                file_path=file_path,
                language=language,
                line_start=line_start,
                line_end=line_end,
                git_commit_hash=git_commit_hash,
                repo_url=repo_url,
            )
            chunks.append(CodeChunk(content=chunk_text, metadata=meta))

            # Advance by (window - overlap)
            step = window_chars - overlap_chars
            if step <= 0:
                step = window_chars  # safety: never get stuck
            start += step

        return chunks
