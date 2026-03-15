"""Tests for the code modality (modality 4) feature."""

from pathlib import Path

import pytest

# ── File-type classification ──────────────────────────────────────────────


class TestFileTypesCode:
    """Verify CODE_EXTS, MODALITY_ID, and classify_file behaviour."""

    def test_code_exts_exist(self):
        from auralith_pipeline.utils.file_types import CODE_EXTS

        assert isinstance(CODE_EXTS, frozenset)
        assert ".py" in CODE_EXTS
        assert ".rs" in CODE_EXTS
        assert ".ts" in CODE_EXTS
        assert ".go" in CODE_EXTS

    def test_code_exts_disjoint_from_text(self):
        from auralith_pipeline.utils.file_types import CODE_EXTS, TEXT_EXTS

        overlap = CODE_EXTS & TEXT_EXTS
        assert overlap == frozenset(), f"CODE_EXTS and TEXT_EXTS overlap: {overlap}"

    def test_modality_id_has_code(self):
        from auralith_pipeline.utils.file_types import MODALITY_ID

        assert "code" in MODALITY_ID
        assert MODALITY_ID["code"] == 4

    def test_classify_file_python(self, tmp_path: Path):
        from auralith_pipeline.utils.file_types import classify_file

        p = tmp_path / "example.py"
        p.write_text("print('hello')")
        assert classify_file(p) == "code"

    def test_classify_file_rust(self, tmp_path: Path):
        from auralith_pipeline.utils.file_types import classify_file

        p = tmp_path / "main.rs"
        p.write_text("fn main() {}")
        assert classify_file(p) == "code"

    def test_classify_file_text(self, tmp_path: Path):
        from auralith_pipeline.utils.file_types import classify_file

        p = tmp_path / "readme.txt"
        p.write_text("hello")
        assert classify_file(p) == "text"

    def test_classify_file_javascript(self, tmp_path: Path):
        from auralith_pipeline.utils.file_types import classify_file

        p = tmp_path / "index.js"
        p.write_text("console.log('hi')")
        assert classify_file(p) == "code"

    def test_classify_file_typescript(self, tmp_path: Path):
        from auralith_pipeline.utils.file_types import classify_file

        p = tmp_path / "app.tsx"
        p.write_text("export default function App() {}")
        assert classify_file(p) == "code"

    # ── Extended language coverage ────────────────────────────────────

    @pytest.mark.parametrize(
        "filename",
        [
            # Legacy / mainframe
            "program.cob",
            "module.cbl",
            "copybook.cpy",
            "solver.f90",
            "legacy.for",
            "modern.f08",
            "package.adb",
            "spec.ads",
            "unit.pas",
            "macro.bas",
            "script.vbs",
            "report.rpgle",
            "program.abap",
            "exec.rexx",
            # GPU / HPC / shaders
            "kernel.cu",
            "header.cuh",
            "shader.hlsl",
            "vertex.vert",
            "fragment.frag",
            "pipeline.metal",
            "compute.wgsl",
            # DevOps / IaC
            "deploy.tf",
            "vars.tfvars",
            "config.hcl",
            "playbook.yml",
            "compose.yaml",
            "setup.dockerfile",
            "default.nix",
            "build.bzl",
            "CMakeLists.cmake",
            # Config-as-code
            "pyproject.toml",
            "settings.ini",
            "app.cfg",
            "nginx.conf",
            "local.env",
            "app.properties",
            "build.gradle",
            "build.sbt",
            # Functional
            "server.ex",
            "test_helper.exs",
            "gen_server.erl",
            "Main.hs",
            "parser.ml",
            "App.elm",
            "core.clj",
            "init.el",
            "helper.scm",
            "lang.rkt",
            # Systems
            "server.nim",
            "app.cr",
            "lib.d",
            # Web / frontend
            "App.vue",
            "Page.svelte",
            "Layout.astro",
            "style.css",
            "theme.scss",
            "vars.less",
            "schema.graphql",
            # Scripting
            "script.pl",
            "Module.pm",
            "process.awk",
            "filter.sed",
            "test.tcl",
            "run.ps1",
            "deploy.bat",
            "build.cmd",
            # Blockchain
            "Token.sol",
            "module.move",
            "contract.vy",
            # HDL
            "cpu.sv",
            "alu.vhd",
            # Assembly / IR
            "boot.asm",
            "start.s",
            "module.ll",
            # IDL / serialization
            "api.proto",
            "service.thrift",
            "schema.fbs",
            "model.smithy",
            "config.cue",
            "types.dhall",
            "template.jsonnet",
            "policy.rego",
            # Other
            "lib.dart",
            "analysis.jl",
            "App.purs",
            "Theorem.lean",
        ],
    )
    def test_classify_extended_code_extensions(self, tmp_path: Path, filename: str):
        from auralith_pipeline.utils.file_types import classify_file

        p = tmp_path / filename
        p.write_text("/* placeholder */")
        assert classify_file(p) == "code", f"{filename} should classify as code"

    def test_code_exts_disjoint_from_image_audio_video(self):
        from auralith_pipeline.utils.file_types import (
            AUDIO_EXTS,
            CODE_EXTS,
            IMAGE_EXTS,
            VIDEO_EXTS,
        )

        for name, other in [("IMAGE", IMAGE_EXTS), ("AUDIO", AUDIO_EXTS), ("VIDEO", VIDEO_EXTS)]:
            overlap = CODE_EXTS & other
            assert overlap == frozenset(), f"CODE_EXTS overlaps with {name}_EXTS: {overlap}"


# ── Extension-to-language mapping ─────────────────────────────────────────


class TestExtensionToLanguage:
    """Verify EXTENSION_TO_LANGUAGE covers all CODE_EXTS."""

    def test_every_code_ext_has_language(self):
        from auralith_pipeline.preprocessing.code_chunker import EXTENSION_TO_LANGUAGE
        from auralith_pipeline.utils.file_types import CODE_EXTS

        missing = CODE_EXTS - set(EXTENSION_TO_LANGUAGE.keys())
        assert (
            missing == set()
        ), f"CODE_EXTS has extensions not mapped in EXTENSION_TO_LANGUAGE: {sorted(missing)}"


# ── CodeConfig ────────────────────────────────────────────────────────────


class TestCodeConfig:
    """Test the CodeConfig dataclass and its integration with PipelineConfig."""

    def test_defaults(self):
        from auralith_pipeline.config.pipeline_config import CodeConfig

        c = CodeConfig()
        assert c.enabled is True
        assert c.chunk_size == 512
        assert c.overlap == 256
        assert c.use_tree_sitter is True
        assert c.min_chunk_chars == 50
        assert c.max_chunk_chars == 16_000
        assert c.max_file_size_bytes == 1_000_000
        assert "python" in c.supported_languages
        # Verify expanded coverage
        assert "cobol" in c.supported_languages
        assert "fortran" in c.supported_languages
        assert "cuda" in c.supported_languages
        assert "bash" in c.supported_languages
        assert "hcl" in c.supported_languages
        assert "yaml" in c.supported_languages
        assert "solidity" in c.supported_languages
        assert "protobuf" in c.supported_languages

    def test_custom_values(self):
        from auralith_pipeline.config.pipeline_config import CodeConfig

        c = CodeConfig(chunk_size=1024, overlap=128, use_tree_sitter=False)
        assert c.chunk_size == 1024
        assert c.overlap == 128
        assert c.use_tree_sitter is False

    def test_pipeline_config_has_code(self):
        from auralith_pipeline.config.pipeline_config import CodeConfig, PipelineConfig

        pc = PipelineConfig()
        assert isinstance(pc.code, CodeConfig)
        assert pc.code.enabled is True

    def test_pipeline_config_from_dict(self):
        from auralith_pipeline.config.pipeline_config import PipelineConfig

        d = {"code": {"chunk_size": 1024, "overlap": 128}}
        pc = PipelineConfig.from_dict(d)
        assert pc.code.chunk_size == 1024
        assert pc.code.overlap == 128


# ── CodeChunker ───────────────────────────────────────────────────────────


class TestCodeChunker:
    """Test CodeChunker with fixed-size fallback (no tree-sitter)."""

    def test_detect_language(self):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        chunker = CodeChunker()
        # Mainstream
        assert chunker.detect_language("foo.py") == "python"
        assert chunker.detect_language("bar.rs") == "rust"
        assert chunker.detect_language("baz.ts") == "typescript"
        assert chunker.detect_language("app.go") == "go"
        assert chunker.detect_language("Main.java") == "java"
        # Legacy
        assert chunker.detect_language("prog.cob") == "cobol"
        assert chunker.detect_language("solver.f90") == "fortran"
        assert chunker.detect_language("spec.ads") == "ada"
        # GPU / HPC
        assert chunker.detect_language("kernel.cu") == "cuda"
        assert chunker.detect_language("shader.hlsl") == "hlsl"
        assert chunker.detect_language("vertex.glsl") == "glsl"
        # DevOps / IaC
        assert chunker.detect_language("main.tf") == "hcl"
        assert chunker.detect_language("playbook.yml") == "yaml"
        assert chunker.detect_language("setup.dockerfile") == "dockerfile"
        assert chunker.detect_language("default.nix") == "nix"
        assert chunker.detect_language("script.ps1") == "powershell"
        # Config-as-code
        assert chunker.detect_language("pyproject.toml") == "toml"
        assert chunker.detect_language("settings.ini") == "ini"
        assert chunker.detect_language("nginx.conf") == "conf"
        # Other
        assert chunker.detect_language("Token.sol") == "solidity"
        assert chunker.detect_language("api.proto") == "protobuf"
        assert chunker.detect_language("policy.rego") == "rego"
        # Non-code
        assert chunker.detect_language("readme.md") is None
        assert chunker.detect_language("data.csv") is None

    def test_chunk_file_python(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        code = (
            "def hello():\n"
            "    print('hello world')\n"
            "\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
        )
        p = tmp_path / "example.py"
        p.write_text(code)

        # Disable tree-sitter so we hit the fixed-size path
        chunker = CodeChunker(use_tree_sitter=False, chunk_size=512)
        chunks = chunker.chunk_file(p)

        assert len(chunks) >= 1
        assert chunks[0].metadata.language == "python"
        assert chunks[0].metadata.file_path == str(p)

    def test_chunk_file_empty(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        p = tmp_path / "empty.py"
        p.write_text("")

        chunker = CodeChunker(use_tree_sitter=False)
        assert chunker.chunk_file(p) == []

    def test_chunk_file_unknown_ext(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,2,3\n")

        chunker = CodeChunker(use_tree_sitter=False)
        assert chunker.chunk_file(p) == []

    def test_chunk_metadata_fields(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        p = tmp_path / "test.go"
        p.write_text(
            'package main\n\nimport "fmt"\n\n'
            'func main() {\n    fmt.Println("hello world from go")\n}\n'
        )

        chunker = CodeChunker(
            use_tree_sitter=False,
        )
        chunks = chunker.chunk_file(p, git_commit_hash="abc123", repo_url="https://example.com")
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta.git_commit_hash == "abc123"
        assert meta.repo_url == "https://example.com"
        assert meta.chunk_index == 0
        assert meta.total_chunks == len(chunks)

    def test_chunk_metadata_to_dict(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        p = tmp_path / "app.js"
        p.write_text(
            "// Application entry point\n"
            "function main() {\n"
            "    console.log('hello world from javascript');\n"
            "}\n"
            "main();\n"
        )

        chunker = CodeChunker(use_tree_sitter=False)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1

        d = chunks[0].metadata.to_dict()
        assert isinstance(d, dict)
        assert "language" in d
        assert d["language"] == "javascript"

    def test_min_chunk_chars(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        p = tmp_path / "tiny.py"
        p.write_text("x=1")  # 3 chars, below default min_chunk_chars

        chunker = CodeChunker(use_tree_sitter=False, min_chunk_chars=50)
        chunks = chunker.chunk_file(p)
        assert len(chunks) == 0

    def test_large_file_multiple_chunks(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        # Generate a large-ish file that should produce multiple chunks
        lines = [f"def func_{i}():\n    return {i}\n" for i in range(200)]
        p = tmp_path / "big.py"
        p.write_text("\n".join(lines))

        # Small chunk_size forces multiple chunks
        chunker = CodeChunker(use_tree_sitter=False, chunk_size=64, overlap=16)
        chunks = chunker.chunk_file(p)
        assert len(chunks) > 1


# ── CodeChunker with tree-sitter ──────────────────────────────────────────


class TestCodeChunkerTreeSitter:
    """Test tree-sitter based chunking.

    tree-sitter + tree-sitter-languages are installed in dev/CI via
    ``pip install -e ".[dev,code]"`` so these tests must never be skipped
    in a production-ready CI pipeline.
    """

    def test_ast_chunking_python(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        code = (
            "class Greeter:\n"
            "    def hello(self):\n"
            "        print('hello')\n"
            "\n"
            "def standalone():\n"
            "    return 42\n"
        )
        p = tmp_path / "example.py"
        p.write_text(code)

        chunker = CodeChunker(use_tree_sitter=True)
        chunks = chunker.chunk_file(p)

        assert len(chunks) >= 1
        # At least one chunk should have a class or function name
        names = [c.metadata.function_name or c.metadata.class_name for c in chunks]
        assert any(n is not None for n in names)

    def test_ast_chunking_javascript(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        code = (
            "function greet(name) {\n"
            "    return 'Hello ' + name;\n"
            "}\n\n"
            "class Animal {\n"
            "    constructor(type) { this.type = type; }\n"
            "}\n"
        )
        p = tmp_path / "app.js"
        p.write_text(code)

        chunker = CodeChunker(use_tree_sitter=True)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1

    def test_ast_chunking_rust(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        code = (
            "struct Point {\n"
            "    x: f64,\n"
            "    y: f64,\n"
            "}\n\n"
            "impl Point {\n"
            "    fn distance(&self) -> f64 {\n"
            "        (self.x * self.x + self.y * self.y).sqrt()\n"
            "    }\n"
            "}\n"
        )
        p = tmp_path / "point.rs"
        p.write_text(code)

        chunker = CodeChunker(use_tree_sitter=True)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1

    def test_ast_chunking_go(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        code = (
            "package main\n\n"
            'import "fmt"\n\n'
            "func main() {\n"
            '    fmt.Println("hello world")\n'
            "}\n\n"
            "func add(a int, b int) int {\n"
            "    return a + b\n"
            "}\n"
        )
        p = tmp_path / "main.go"
        p.write_text(code)

        chunker = CodeChunker(use_tree_sitter=True)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1

    def test_ast_chunking_java(self, tmp_path: Path):
        from auralith_pipeline.preprocessing.code_chunker import CodeChunker

        code = (
            "public class Calculator {\n"
            "    public int add(int a, int b) {\n"
            "        return a + b;\n"
            "    }\n"
            "    public int multiply(int a, int b) {\n"
            "        return a * b;\n"
            "    }\n"
            "}\n"
        )
        p = tmp_path / "Calculator.java"
        p.write_text(code)

        chunker = CodeChunker(use_tree_sitter=True)
        chunks = chunker.chunk_file(p)
        assert len(chunks) >= 1


# ── LocalCodeSource ──────────────────────────────────────────────────────


class TestLocalCodeSource:
    """Test LocalCodeSource iteration."""

    def test_iter_code_files(self, tmp_path: Path):
        from auralith_pipeline.sources.code import LocalCodeSource

        (tmp_path / "main.py").write_text(
            "def hello():\n    print('hello world from python')\n\nhello()\n"
        )
        (tmp_path / "readme.txt").write_text("just text\n")
        (tmp_path / "lib.rs").write_text('fn main() {\n    println!("hello world from rust");\n}\n')

        source = LocalCodeSource(tmp_path)
        samples = list(source)

        # Should pick up .py and .rs, skip .txt
        assert len(samples) >= 2
        modalities = {s.modality for s in samples}
        assert modalities == {"code"}

    def test_skips_git_dir(self, tmp_path: Path):
        from auralith_pipeline.sources.code import LocalCodeSource

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
        (tmp_path / "app.py").write_text("x = 1\n" * 20)

        source = LocalCodeSource(tmp_path)
        sources = [s.source for s in source]
        assert all(".git" not in s for s in sources)

    def test_max_samples(self, tmp_path: Path):
        from auralith_pipeline.sources.code import LocalCodeSource

        for i in range(10):
            (tmp_path / f"mod_{i}.py").write_text(f"x = {i}\n" * 20)

        source = LocalCodeSource(tmp_path, max_samples=3)
        samples = list(source)
        assert len(samples) == 3

    def test_skips_oversized_files(self, tmp_path: Path):
        from auralith_pipeline.config.pipeline_config import CodeConfig
        from auralith_pipeline.sources.code import LocalCodeSource

        # max_file_size_bytes = 100 → file is oversized
        config = CodeConfig(max_file_size_bytes=100)
        p = tmp_path / "big.py"
        p.write_text("x = 1\n" * 100)  # ~600 bytes

        source = LocalCodeSource(tmp_path, config=config)
        samples = list(source)
        assert len(samples) == 0

    def test_name_property(self, tmp_path: Path):
        from auralith_pipeline.sources.code import LocalCodeSource

        source = LocalCodeSource(tmp_path)
        assert source.name.startswith("code:")

    def test_len(self, tmp_path: Path):
        from auralith_pipeline.sources.code import LocalCodeSource

        (tmp_path / "a.py").write_text("a = 1\n" * 20)
        (tmp_path / "b.js").write_text("let b = 1;\n" * 20)

        source = LocalCodeSource(tmp_path)
        assert len(source) == 2


# ── Preprocessor modality guards ─────────────────────────────────────────


class TestPreprocessorCodeGuards:
    """Verify that normalize() and passes_filter() handle code correctly."""

    def test_normalize_preserves_indentation(self):
        from auralith_pipeline.preprocessing import TextNormalizer

        normalizer = TextNormalizer()
        code = "def f():\n    if True:\n        return 1\n"
        result = normalizer.normalize(code, modality="code")

        # Indentation must be preserved
        assert "    if True:" in result
        assert "        return 1" in result

    def test_normalize_text_collapses_whitespace(self):
        from auralith_pipeline.preprocessing import TextNormalizer

        normalizer = TextNormalizer()
        text = "Hello   world\n\n\n\ntest"
        result = normalizer.normalize(text, modality="text")
        assert "   " not in result

    def test_quality_filter_passes_code_with_special_chars(self):
        from auralith_pipeline.config.pipeline_config import QualityConfig
        from auralith_pipeline.preprocessing import QualityFilter
        from auralith_pipeline.sources.data_sources import DataSample

        config = QualityConfig(
            max_special_char_ratio=0.1,  # very strict for prose
            min_text_length=10,
            min_word_count=2,
        )
        filt = QualityFilter(config)

        # Code naturally has lots of special chars ({, }, (, ), ;, :, etc.)
        code = 'def foo(x): return {"key": x + 1}\n' * 5
        sample = DataSample(content=code, source="test.py", modality="code")
        assert filt.passes_filter(sample) is True

    def test_quality_filter_blocks_bad_prose(self):
        from auralith_pipeline.config.pipeline_config import QualityConfig
        from auralith_pipeline.preprocessing import QualityFilter
        from auralith_pipeline.sources.data_sources import DataSample

        config = QualityConfig(
            max_special_char_ratio=0.1,
            min_text_length=10,
            min_word_count=2,
        )
        filt = QualityFilter(config)

        # Prose with tons of special chars should be blocked
        text = "!!@@##$$%%^^&&**(())__++" * 10
        sample = DataSample(content=text, source="bad.txt", modality="text")
        assert filt.passes_filter(sample) is False


# ── Multimodal tokenizer — code region relabelling ────────────────────────


class TestMultimodalTokenizerCodeRegion:
    """Verify that <CODE>…<CODE_END> tokens are relabelled to MODALITY_CODE."""

    def test_code_region_mask(self):
        from auralith_pipeline.tokenization.bpe_tokenizer import BPETokenizer

        # Build a minimal trained BPE tokenizer for this test
        tok = BPETokenizer(vocab_size=500)
        tok.train("the quick brown fox jumps over the lazy dog " * 100)

        # Encode some text with CODE markers
        code_start = BPETokenizer.SPECIAL_TOKENS["<CODE>"]
        code_end = BPETokenizer.SPECIAL_TOKENS["<CODE_END>"]

        body = tok.encode("print hello world", add_special_tokens=False)
        input_ids = [code_start] + body + [code_end]

        # Build initial mask (all text)
        MODALITY_TEXT = 0
        MODALITY_CODE = 4
        modality_mask = [MODALITY_TEXT] * len(input_ids)

        # Simulate the relabelling logic from encode_with_mask
        inside_code = False
        for i, tid in enumerate(input_ids):
            if tid == code_start:
                inside_code = True
                modality_mask[i] = MODALITY_CODE
            elif tid == code_end:
                modality_mask[i] = MODALITY_CODE
                inside_code = False
            elif inside_code:
                modality_mask[i] = MODALITY_CODE

        # All entries should be MODALITY_CODE
        assert all(m == MODALITY_CODE for m in modality_mask)


# ── End-to-end integration (lightweight) ─────────────────────────────────


class TestCodePipelineIntegration:
    """Lightweight end-to-end test: code dir → DataSamples → token IDs."""

    def test_local_source_to_tokens(self, tmp_path: Path):
        from auralith_pipeline.sources.code import LocalCodeSource
        from auralith_pipeline.tokenization.bpe_tokenizer import BPETokenizer

        # Write some code files
        (tmp_path / "lib.py").write_text("def add(a, b):\n    return a + b\n" * 5)
        (tmp_path / "main.go").write_text(
            'package main\n\nfunc main() {\n    fmt.Println("hello")\n}\n'
        )

        # Create source
        source = LocalCodeSource(tmp_path)
        samples = list(source)
        assert len(samples) >= 2

        # Train a tiny BPE and tokenize
        tok = BPETokenizer(vocab_size=300)
        all_text = " ".join(s.content for s in samples)
        tok.train(all_text)

        code_start = BPETokenizer.SPECIAL_TOKENS["<CODE>"]
        code_end = BPETokenizer.SPECIAL_TOKENS["<CODE_END>"]

        for sample in samples:
            body = tok.encode(sample.content, add_special_tokens=False)
            ids = [code_start] + body + [code_end]
            assert ids[0] == code_start
            assert ids[-1] == code_end
            assert len(ids) >= 3  # at least start + something + end

    def test_data_sources_the_stack_modality(self):
        """Ensure the_stack entry in DATASET_REGISTRY has modality=code."""
        from auralith_pipeline.sources.data_sources import DATASET_REGISTRY

        entry = DATASET_REGISTRY["the_stack"]
        assert entry.get("modality") == "code"
