"""Shared file-type constants and classification logic.

Centralised here so that ``cli.py``, ``worker.py``, and any future
consumer agree on which extensions map to which modality.

**.npy** files are ambiguous — they can represent either image arrays
(H, W, 3) or audio waveforms (1-D / 2-D time-series).  The function
:func:`_classify_file` resolves the ambiguity by inspecting the
**parent directory name**.  If it contains ``image`` or ``img`` the
file is classified as ``"image"``; if it contains ``audio`` or
``speech`` it is ``"audio"``.  When no keyword matches the directory
name, ``None`` is returned so the caller can decide how to handle it.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── extension sets ─────────────────────────────────────────────────────
# .npy is intentionally absent — it is handled by directory-based
# routing inside _classify_file().

TEXT_EXTS: frozenset[str] = frozenset(
    {".txt", ".md", ".rst", ".csv", ".json", ".jsonl", ".tsv", ".xml", ".html"}
)
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
AUDIO_EXTS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
CODE_EXTS: frozenset[str] = frozenset(
    {
        # ── Mainstream / general-purpose ───────────────────────────
        ".py",
        ".pyi",
        ".pyx",
        ".pxd",  # Python, Cython
        ".js",
        ".mjs",
        ".cjs",
        ".jsx",  # JavaScript
        ".ts",
        ".tsx",
        ".mts",
        ".cts",  # TypeScript
        ".java",  # Java
        ".go",  # Go
        ".rs",  # Rust
        ".rb",
        ".erb",  # Ruby
        ".cs",
        ".csx",  # C#
        ".php",  # PHP
        ".swift",  # Swift
        ".kt",
        ".kts",  # Kotlin
        ".scala",
        ".sc",  # Scala
        ".r",
        ".rmd",  # R
        ".lua",  # Lua
        ".zig",  # Zig
        ".dart",  # Dart
        ".ex",
        ".exs",  # Elixir
        ".erl",
        ".hrl",  # Erlang
        ".hs",
        ".lhs",  # Haskell
        ".ml",
        ".mli",  # OCaml
        ".fs",
        ".fsi",
        ".fsx",  # F#
        ".clj",
        ".cljs",
        ".cljc",
        ".edn",  # Clojure
        ".lisp",
        ".cl",
        ".el",  # Common Lisp, Emacs Lisp
        ".scm",
        ".ss",
        ".rkt",  # Scheme, Racket
        ".jl",  # Julia
        ".nim",
        ".nims",  # Nim
        ".cr",  # Crystal
        ".v",
        ".vh",  # V  (also Verilog – fine)
        ".d",  # D
        ".pl",
        ".pm",  # Perl
        ".groovy",
        ".gvy",
        ".gy",  # Groovy
        # ── C / C++ family ────────────────────────────────────────
        ".c",
        ".h",  # C
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hxx",
        ".hh",  # C++
        ".m",
        ".mm",  # Objective-C / C++
        # ── Legacy / mainframe ────────────────────────────────────
        ".cob",
        ".cbl",
        ".cpy",  # COBOL
        ".f",
        ".f90",
        ".f95",
        ".f03",
        ".f08",
        ".for",  # Fortran
        ".ada",
        ".adb",
        ".ads",  # Ada
        ".pas",
        ".pp",  # Pascal / Delphi
        ".vb",
        ".bas",
        ".vbs",  # Visual Basic / VBScript
        ".rpg",
        ".rpgle",  # RPG (IBM i)
        ".abap",  # SAP ABAP
        ".rexx",
        ".rex",  # REXX (IBM)
        ".clu",  # CLU
        ".pro",  # Prolog
        # ── GPU / HPC / shaders ───────────────────────────────────
        ".cu",
        ".cuh",  # CUDA
        # OpenCL  (shares .cl w/ Lisp)
        ".hlsl",
        ".glsl",
        ".vert",
        ".frag",
        ".comp",  # Shader languages
        ".metal",  # Apple Metal
        ".wgsl",  # WebGPU Shader Language
        ".ptx",  # NVIDIA PTX
        # ── DevOps / IaC / build ──────────────────────────────────
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ksh",
        ".csh",  # Shell
        ".ps1",
        ".psm1",
        ".psd1",  # PowerShell
        ".bat",
        ".cmd",  # Windows batch
        ".tf",
        ".tfvars",  # Terraform
        # Puppet  (also Pascal)
        ".yml",
        ".yaml",  # YAML (Ansible, CI, K8s, etc.)
        ".dockerfile",  # Dockerfile (explicit ext)
        ".hcl",  # HashiCorp HCL
        ".nix",  # Nix
        ".bzl",  # Bazel / Starlark
        ".cmake",  # CMake script files
        ".just",  # Justfile
        # ── Config-as-code ────────────────────────────────────────
        ".toml",  # TOML
        ".ini",
        ".cfg",  # INI-style config
        ".conf",  # Generic conf (Nginx, etc.)
        ".env",  # Dotenv
        ".properties",  # Java properties
        ".gradle",  # Gradle (Groovy DSL)
        ".sbt",  # SBT (Scala build tool)
        ".cabal",  # Haskell Cabal
        # ── Web / frontend ────────────────────────────────────────
        ".vue",  # Vue SFC
        ".svelte",  # Svelte
        ".astro",  # Astro
        ".css",  # CSS
        ".scss",
        ".sass",
        ".less",
        ".styl",  # CSS preprocessors
        ".graphql",
        ".gql",  # GraphQL
        ".wasm",
        ".wat",  # WebAssembly (text format)
        # ── Data / science / query ────────────────────────────────
        ".sql",
        ".psql",
        ".plsql",  # SQL dialects
        # MATLAB (shares w/ ObjC)
        ".sas",  # SAS
        ".do",
        ".ado",  # Stata
        ".pig",  # Apache Pig
        # ── Scripting / text processing ───────────────────────────
        ".awk",  # AWK
        ".sed",  # sed
        ".tcl",  # Tcl
        ".expect",  # Expect
        # ── Other notable languages ───────────────────────────────
        ".sol",  # Solidity (Ethereum)
        ".move",  # Move (blockchain)
        ".vy",  # Vyper  (Ethereum)
        ".cairo",  # Cairo  (StarkNet)
        ".elm",  # Elm
        ".purs",  # PureScript
        ".hack",
        ".hhi",  # Hack (Meta)
        ".lean",  # Lean (theorem prover)
        ".idr",  # Idris
        ".agda",  # Agda
        ".coq",  # Coq
        # Coq / Verilog (intentional dup)
        ".sv",
        ".svh",  # SystemVerilog
        ".vhd",
        ".vhdl",  # VHDL
        ".asm",
        ".s",  # Assembly
        ".ll",  # LLVM IR
        # also Terraform (dup OK in set)
        ".proto",  # Protocol Buffers
        ".thrift",  # Apache Thrift
        ".capnp",  # Cap'n Proto
        ".fbs",  # FlatBuffers
        ".smithy",  # Smithy (AWS)
        ".cue",  # CUE
        ".dhall",  # Dhall
        ".jsonnet",
        ".libsonnet",  # Jsonnet
        ".pkl",  # Apple Pkl
        ".starlark",  # Starlark
        ".rego",  # OPA Rego
        ".zeek",  # Zeek (network security)
    }
)

# ── modality mask IDs ──────────────────────────────────────────────────

MODALITY_ID: dict[str, int] = {"text": 0, "image": 1, "audio": 2, "video": 3, "code": 4}

# ── token offsets ──────────────────────────────────────────────────────
# Separate ID ranges prevent collisions across modalities inside a
# single vocabulary.

IMAGE_TOKEN_OFFSET: int = 100_000
AUDIO_TOKEN_OFFSET: int = 200_000
VIDEO_TOKEN_OFFSET: int = 300_000

# ── directory keyword patterns for .npy disambiguation ─────────────────

_IMAGE_DIR_RE = re.compile(r"image|img|picture|photo|visual", re.IGNORECASE)
_AUDIO_DIR_RE = re.compile(r"audio|speech|sound|music|waveform", re.IGNORECASE)


def classify_file(file_path: Path) -> str | None:
    """Return the modality name for *file_path*, or ``None`` if unsupported.

    For ``.npy`` files the parent directory name is inspected:

    * ``images/data.npy``  → ``"image"``
    * ``audio/data.npy``   → ``"audio"``
    * ``misc/data.npy``    → ``None`` (ambiguous — caller must decide)

    All other extensions are matched against the canonical sets above.
    """
    ext = file_path.suffix.lower()

    # ── .npy: directory-based routing ──────────────────────────────
    if ext == ".npy":
        parent = file_path.parent.name.lower()
        if _IMAGE_DIR_RE.search(parent):
            return "image"
        if _AUDIO_DIR_RE.search(parent):
            return "audio"
        # Ambiguous — cannot decide from extension alone
        return None

    # ── standard extension lookup ──────────────────────────────────
    if ext in CODE_EXTS:
        return "code"
    if ext in TEXT_EXTS:
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return None
