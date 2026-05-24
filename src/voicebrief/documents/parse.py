"""Document and repository parsing for Mode 2.

Chunking is structure-aware rather than fixed-window, because the structure is the
part that carries meaning. A paper's "Method" section is a unit; so is a repo's
`auth.py`. Splitting either at an arbitrary 800 characters produces chunks that
retrieve badly and summarize worse.

Repository ingestion is the differentiated piece (PRD §5): the walker reads the README
and the dependency manifest, then ranks modules by import centrality, so the outline
can explain the *architecture* instead of narrating files alphabetically.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from voicebrief.logging import get_logger

log = get_logger(__name__)

MAX_CHUNK_CHARS = 3000
MIN_CHUNK_CHARS = 120

# Headings in papers and markdown alike.
_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_PAPER_SECTION = re.compile(
    r"^\s*(?:\d+\.?\s+)?("
    r"abstract|introduction|background|related work|method(?:s|ology)?|approach|"
    r"experiments?|evaluation|results?|discussion|limitations?|conclusions?|"
    r"future work|references|appendix"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".dart", ".c", ".cpp", ".h", ".sql",
}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "__pycache__", ".venv", "venv",
    "target", "vendor", ".next", "coverage", ".pytest_cache", "migrations",
}


@dataclass(slots=True)
class Chunk:
    """One retrievable unit with enough context to be cited."""

    index: int
    text: str
    section: str
    kind: str  # section | file | readme | manifest
    meta: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    kind: str  # pdf | markdown | text | docx | repo
    chunks: list[Chunk] = field(default_factory=list)
    page_count: int | None = None
    meta: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Text and markdown
# ─────────────────────────────────────────────────────────────────────────────
def split_by_structure(text: str, *, fallback_section: str = "body") -> list[Chunk]:
    """Split on headings, then bound oversized sections at paragraph edges."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    boundaries: list[tuple[int, str]] = []
    for match in _MD_HEADING.finditer(text):
        boundaries.append((match.start(), match.group(2).strip()))
    if not boundaries:
        for match in _PAPER_SECTION.finditer(text):
            boundaries.append((match.start(), match.group(1).strip().title()))

    chunks: list[Chunk] = []
    if not boundaries:
        for piece in _bound(text):
            chunks.append(
                Chunk(index=len(chunks), text=piece, section=fallback_section, kind="section")
            )
        return chunks

    if boundaries[0][0] > MIN_CHUNK_CHARS:
        for piece in _bound(text[: boundaries[0][0]]):
            chunks.append(Chunk(index=len(chunks), text=piece, section="preamble", kind="section"))

    for position, (start, heading) in enumerate(boundaries):
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        for piece in _bound(body):
            chunks.append(Chunk(index=len(chunks), text=piece, section=heading, kind="section"))

    return chunks


def _bound(text: str) -> list[str]:
    """Cut oversized text at paragraph boundaries, never mid-sentence."""
    text = text.strip()
    if len(text) <= MAX_CHUNK_CHARS:
        return [text] if text else []

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if size + len(paragraph) > MAX_CHUNK_CHARS and current:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def parse_text(content: str, title: str, kind: str = "text") -> ParsedDocument:
    return ParsedDocument(title=title, kind=kind, chunks=split_by_structure(content))


def parse_pdf(data: bytes, title: str) -> ParsedDocument:
    """Extract text from a PDF. Raises if the file yields nothing readable."""
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("PDF support needs `pip install 'voicebrief[docs]'`") from exc

    with pymupdf.open(stream=data, filetype="pdf") as document:
        pages = [page.get_text() for page in document]
        page_count = document.page_count

    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        # A scanned PDF has pages but no text layer. Saying so beats producing an
        # empty episode and leaving the user to guess why.
        raise ValueError(
            f"{title} has no extractable text ({page_count} pages). "
            f"It is probably a scan; OCR is not supported in v1."
        )

    parsed = parse_text(text, title, kind="pdf")
    parsed.page_count = page_count
    return parsed


def parse_docx(data: bytes, title: str) -> ParsedDocument:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("DOCX support needs `pip install 'voicebrief[docs]'`") from exc

    document = docx.Document(io.BytesIO(data))
    lines = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Re-emit Word headings as markdown so one splitter handles every format.
        if paragraph.style.name.startswith("Heading"):
            level = paragraph.style.name.removeprefix("Heading ").strip()
            hashes = "#" * (int(level) if level.isdigit() else 2)
            lines.append(f"{hashes} {text}")
        else:
            lines.append(text)
    return parse_text("\n\n".join(lines), title, kind="docx")


# ─────────────────────────────────────────────────────────────────────────────
# Repositories
# ─────────────────────────────────────────────────────────────────────────────
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|"
    r"(?:import|export).*?from\s+['\"]([^'\"]+)['\"])",
    re.MULTILINE,
)


def parse_repo_zip(data: bytes, title: str) -> ParsedDocument:
    """Walk a repository archive and chunk it by file, ranked by centrality."""
    files: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.file_size > 400_000:
                continue
            path = Path(info.filename)
            # GitHub archives nest everything under one top directory.
            parts = path.parts[1:] if len(path.parts) > 1 else path.parts
            if any(part in SKIP_DIRS for part in parts):
                continue
            if path.suffix.lower() not in SOURCE_EXTENSIONS | DOC_EXTENSIONS:
                continue
            try:
                files["/".join(parts)] = archive.read(info).decode("utf-8", errors="replace")
            except (KeyError, OSError):
                continue

    return build_repo_document(files, title)


def build_repo_document(files: dict[str, str], title: str) -> ParsedDocument:
    """Turn a path->content map into an architecture-first document.

    Ordering is the differentiator. README and manifest come first because they state
    what the project is; modules then follow in order of how often the rest of the
    codebase imports them, so the most structurally important code is what the
    outline and script see first.
    """
    if not files:
        raise ValueError(f"{title} contains no readable source or documentation files")

    chunks: list[Chunk] = []

    readme_path = next(
        (p for p in files if p.lower() in ("readme.md", "readme.rst", "readme.txt")), None
    )
    if readme_path:
        for chunk in split_by_structure(files[readme_path]):
            chunk.index = len(chunks)
            chunk.kind = "readme"
            chunk.meta["path"] = readme_path
            chunks.append(chunk)

    for manifest in ("pyproject.toml", "package.json", "go.mod", "Cargo.toml", "pubspec.yaml"):
        if manifest in files:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=files[manifest][:MAX_CHUNK_CHARS],
                    section=manifest,
                    kind="manifest",
                    meta={"path": manifest},
                )
            )

    centrality = import_centrality(files)
    source_files = [
        p for p in files if Path(p).suffix.lower() in SOURCE_EXTENSIONS
    ]
    source_files.sort(key=lambda p: (-centrality.get(p, 0), len(p)))

    for path in source_files[:60]:
        content = files[path].strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        for piece in _bound(content):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=piece,
                    section=path,
                    kind="file",
                    meta={"path": path, "imported_by": centrality.get(path, 0)},
                )
            )

    return ParsedDocument(
        title=title,
        kind="repo",
        chunks=chunks,
        meta={
            "file_count": len(files),
            "source_files": len(source_files),
            "entry_points": source_files[:5],
            "has_readme": readme_path is not None,
        },
    )


def import_centrality(files: dict[str, str]) -> dict[str, int]:
    """Count how many other files import each module.

    A rough proxy for architectural importance, and a good one: the module everything
    imports is almost always the one worth explaining first.
    """
    stems = {Path(p).stem: p for p in files if Path(p).suffix.lower() in SOURCE_EXTENSIONS}
    counts: dict[str, int] = dict.fromkeys(stems.values(), 0)

    for path, content in files.items():
        if Path(path).suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        for match in _IMPORT_RE.finditer(content):
            target = next((g for g in match.groups() if g), "")
            name = Path(target.replace(".", "/")).name or target
            imported = stems.get(name)
            # A file importing itself is not evidence of centrality.
            if imported and imported != path:
                counts[imported] += 1
    return counts


def parse_upload(filename: str, data: bytes) -> ParsedDocument:
    """Dispatch on extension."""
    suffix = Path(filename).suffix.lower()
    title = Path(filename).stem

    if suffix == ".pdf":
        return parse_pdf(data, title)
    if suffix == ".docx":
        return parse_docx(data, title)
    if suffix == ".zip":
        return parse_repo_zip(data, title)
    if suffix in DOC_EXTENSIONS:
        return parse_text(data.decode("utf-8", errors="replace"), title,
                          kind="markdown" if suffix == ".md" else "text")

    raise ValueError(
        f"Unsupported file type {suffix!r}. Supported: .pdf, .docx, .md, .txt, .rst, .zip"
    )
