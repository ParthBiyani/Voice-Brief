"""Document and repository parsing tests."""

from __future__ import annotations

import io
import zipfile

import pytest

from voicebrief.documents.parse import (
    MAX_CHUNK_CHARS,
    build_repo_document,
    import_centrality,
    parse_repo_zip,
    parse_text,
    parse_upload,
    split_by_structure,
)


class TestStructureSplitting:
    def test_splits_on_markdown_headings(self):
        text = "# Intro\n\n" + "a " * 80 + "\n\n## Method\n\n" + "b " * 80
        sections = [c.section for c in split_by_structure(text)]
        assert "Intro" in sections and "Method" in sections

    def test_splits_on_paper_sections_without_markdown(self):
        text = "Abstract\n\n" + "x " * 90 + "\n\nMethodology\n\n" + "y " * 90
        sections = [c.section.lower() for c in split_by_structure(text)]
        assert any("abstract" in s for s in sections)
        assert any("method" in s for s in sections)

    def test_oversized_sections_are_bounded(self):
        text = "# Big\n\n" + "\n\n".join(["word " * 200] * 12)
        assert all(c.char_count <= MAX_CHUNK_CHARS * 1.2 for c in split_by_structure(text))

    def test_splits_on_paragraph_boundaries_not_mid_sentence(self):
        text = "# H\n\n" + "\n\n".join([f"Paragraph {i}. " + "filler " * 150 for i in range(8)])
        for chunk in split_by_structure(text):
            assert not chunk.text.startswith("filler")

    def test_unstructured_text_still_chunks(self):
        chunks = split_by_structure("just some prose " * 60)
        assert chunks and chunks[0].section == "body"

    def test_empty_input(self):
        assert split_by_structure("") == []

    def test_tiny_sections_are_dropped(self):
        assert split_by_structure("# A\n\nshort\n\n# B\n\nalso short") == []


class TestImportCentrality:
    def test_counts_python_imports(self):
        files = {
            "core.py": "x = 1",
            "a.py": "from core import x",
            "b.py": "import core",
        }
        assert import_centrality(files)["core.py"] == 2

    def test_counts_js_imports(self):
        files = {
            "utils.ts": "export const x = 1;",
            "app.ts": "import { x } from './utils';",
        }
        assert import_centrality(files)["utils.ts"] == 1

    def test_self_import_is_not_centrality(self):
        assert import_centrality({"core.py": "import core"})["core.py"] == 0

    def test_unimported_file_scores_zero(self):
        assert import_centrality({"lonely.py": "x = 1"})["lonely.py"] == 0


class TestRepoDocument:
    def test_readme_comes_first(self):
        """The README states what the project is; ordering the outline behind
        alphabetical file names would bury that."""
        doc = build_repo_document(
            {
                "README.md": "# Project\n\n" + "It does a thing. " * 30,
                "zzz.py": "x = 1\n" + "# filler\n" * 40,
            },
            "demo",
        )
        assert doc.chunks[0].kind == "readme"

    def test_manifest_is_included(self):
        doc = build_repo_document(
            {"pyproject.toml": "[project]\nname='x'\n" + "# pad\n" * 30, "a.py": "x=1\n" * 40},
            "demo",
        )
        assert any(c.kind == "manifest" for c in doc.chunks)

    def test_central_modules_are_ranked_before_leaves(self):
        files = {
            "leaf.py": "# leaf module\n" * 30,
            "core.py": "# core module\n" * 30,
            "a.py": "from core import x\n" + "# pad\n" * 30,
            "b.py": "from core import y\n" + "# pad\n" * 30,
        }
        doc = build_repo_document(files, "demo")
        paths = [c.meta.get("path") for c in doc.chunks if c.kind == "file"]
        assert paths.index("core.py") < paths.index("leaf.py")

    def test_empty_repo_raises(self):
        with pytest.raises(ValueError, match="no readable source"):
            build_repo_document({}, "empty")

    def test_metadata_reports_shape(self):
        doc = build_repo_document(
            {"README.md": "# X\n\n" + "text " * 60, "core.py": "x=1\n" * 40}, "demo"
        )
        assert doc.meta["has_readme"] is True
        assert doc.meta["source_files"] == 1


class TestZipIngestion:
    def _zip(self, files: dict[str, str]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for path, content in files.items():
                archive.writestr(path, content)
        return buffer.getvalue()

    def test_strips_the_github_top_level_directory(self):
        data = self._zip({"repo-main/README.md": "# Repo\n\n" + "words " * 60})
        doc = parse_repo_zip(data, "repo")
        assert doc.chunks[0].meta["path"] == "README.md"

    def test_skips_vendor_directories(self):
        data = self._zip(
            {
                "repo-main/node_modules/dep/index.js": "junk " * 200,
                "repo-main/src.py": "x = 1\n" * 40,
            }
        )
        doc = parse_repo_zip(data, "repo")
        paths = [c.meta.get("path", "") for c in doc.chunks]
        assert not any("node_modules" in p for p in paths)

    def test_skips_unknown_extensions(self):
        data = self._zip(
            {"repo-main/image.png": "binary" * 100, "repo-main/code.py": "x = 1\n" * 40}
        )
        doc = parse_repo_zip(data, "repo")
        assert all(not c.meta.get("path", "").endswith(".png") for c in doc.chunks)


class TestDispatch:
    def test_markdown(self):
        doc = parse_upload("notes.md", b"# Title\n\n" + b"content " * 60)
        assert doc.kind == "markdown"

    def test_plain_text(self):
        assert parse_upload("notes.txt", b"content " * 80).kind == "text"

    def test_unsupported_extension_is_rejected_clearly(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_upload("photo.png", b"\x89PNG")


def test_parse_text_sets_title():
    assert parse_text("body " * 60, "My Paper").title == "My Paper"
