"""ProjectMind as a source of personal context.

ProjectMind is a separate project: a cross-project memory of how its owner works (what
they reach for, what they abandoned) plus an episodic history read from their git
repositories. VoiceBrief borrows from it so a brief can say "this is the tradeoff you
already made in another project", not only "you depend on this library".

The integration is deliberately thin and defensive, because ProjectMind is developed
independently and may be mid-change:

* **Public front door only.** The one thing used is `projectmind context <prompt>
  --project <path>`, whose markdown output is ProjectMind's documented contract. Its
  database and internals are never opened.
* **Fail-soft.** Missing, broken, slow or surprising output yields no personal context
  and a warning, never a failed brief.
* **Pinned by fixtures.** The parser is exercised against recorded samples of the
  documented format, so a format change fails a test instead of silently degrading the
  7am run.
* **Read-only and offline.** ProjectMind's own privacy rules apply; what comes back is
  used to rank and to brief the writer, and is stored only as the derived context in the
  history of the episode that used it.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)

# ProjectMind only answers a prompt that names something it knows: a generic "what do I
# use?" gets "nothing injected". So the questions name concrete technologies, taken from
# the listener's own repositories when a GitHub profile exists.
DEFAULT_SEEDS: tuple[str, ...] = (
    "FastAPI", "Postgres", "Qdrant", "Flutter", "Supabase", "LangGraph", "Claude", "Docker",
)  # fmt: skip
SEEDS_PER_QUERY = 4
MAX_QUERIES = 3


def profile_queries(seeds: list[str] | None = None) -> tuple[str, ...]:
    """Questions that name technologies, so ProjectMind's relevance gate lets them through."""
    # The listener's own technologies first, then the defaults ProjectMind is known to
    # recognise, so a profile full of obscure packages still gets useful questions.
    seen: set[str] = set()
    names: list[str] = []
    for name in [*(seeds or []), *DEFAULT_SEEDS]:
        if len(name) >= 3 and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    names = names[: SEEDS_PER_QUERY * MAX_QUERIES]
    groups = [names[i : i + SEEDS_PER_QUERY] for i in range(0, len(names), SEEDS_PER_QUERY)]
    queries = []
    for group in groups:
        listed = ", ".join(group[:-1]) + f" or {group[-1]}" if len(group) > 1 else group[0]
        queries.append(
            f"Should I build this with {listed}, or something else? "
            "Which of these do I usually pick, and which have I moved away from?"
        )
    return tuple(queries)


PROFILE_QUERIES: tuple[str, ...] = profile_queries()

QUERY_TIMEOUT_SECONDS = 90
MAX_STATEMENTS = 40
MAX_HISTORY = 15
CONTEXT_TTL_HOURS = 24

_SECTION_RE = re.compile(r"^###\s+(.*)$")
_ENTITIES_RE = re.compile(r"entities matched:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SCOPE_RE = re.compile(r"\s+_\((.*?)\)_\s*$")
# "- **[decision] Project** (2026-08) — what" in the documented form; current ProjectMind
# renders "- ** project** (2026-08) — what" with no type tag, so the tag is optional.
_HISTORY_HEAD_RE = re.compile(
    r"^-\s+\*\*\s*(?:\[(?P<type>[^\]]+)\]\s+)?(?P<project>[^*]+?)\s*\*\*\s+\((?P<when>[^)]*)\)"
    r"\s+[—-]\s+(?P<what>.+)$"
)
_ABANDON_RE = re.compile(
    r"\b(abandon\w*|moved? (?:away )?from|stopp?ed using|dropp?ed|replaced|migrated (?:off|away)|"
    r"gave up on|no longer uses?|avoids?)\b",
    re.IGNORECASE,
)
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9+#.]{2,}\b")
# Capitalised words that open a sentence about someone's habits, not tools.
_NOT_TOOLS = frozenset(
    {
        "uses", "reaches", "prefers", "abandoned", "avoids", "chose", "picked", "moved",
        "stopped", "dropped", "replaced", "migrated", "tried", "started", "builds",
        "wrote", "writes", "relies", "favours", "favors", "treats", "keeps", "does",
        "the", "for", "and", "with", "when", "this", "that", "then", "also", "after",
        "before", "because", "anything", "everything", "mostly", "usually", "always",
        "never", "projects", "project", "developer", "client", "server", "backend",
        "frontend", "mobile", "relational", "reads", "auth", "storage",
    }
)  # fmt: skip


@dataclass(slots=True)
class HistoryItem:
    kind: str
    project: str
    when: str
    what: str


@dataclass(slots=True)
class PersonalContext:
    """What ProjectMind told us about the listener, reduced to what ranking can use."""

    source: str = "projectmind"
    statements: list[str] = field(default_factory=list)
    history: list[HistoryItem] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    fetched_at: datetime | None = None
    queries_answered: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.statements or self.history or self.interests)

    def to_json(self) -> dict:
        data = asdict(self)
        data["fetched_at"] = self.fetched_at.isoformat() if self.fetched_at else None
        return data

    @classmethod
    def from_json(cls, data: dict) -> PersonalContext:
        fetched = data.get("fetched_at")
        return cls(
            source=data.get("source", "projectmind"),
            statements=list(data.get("statements", [])),
            history=[HistoryItem(**h) for h in data.get("history", [])],
            interests=list(data.get("interests", [])),
            avoid=list(data.get("avoid", [])),
            projects=list(data.get("projects", [])),
            fetched_at=datetime.fromisoformat(fetched) if fetched else None,
            queries_answered=int(data.get("queries_answered", 0)),
        )


@dataclass(slots=True)
class ParsedBundle:
    statements: list[str] = field(default_factory=list)
    history: list[HistoryItem] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.statements or self.history)


def parse_bundle(markdown: str) -> ParsedBundle:
    """Read ProjectMind's rendered context bundle.

    Tolerant by design: unknown sections are ignored, and anything unrecognised yields
    an empty result rather than an exception.
    """
    parsed = ParsedBundle()
    if not markdown or "ProjectMind context" not in markdown:
        return parsed

    section = ""
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if heading := _SECTION_RE.match(line):
            section = heading.group(1).strip().lower()
            continue
        if not line.startswith("- "):
            continue

        if section.startswith("how this developer works"):
            statement = _SCOPE_RE.sub("", line[2:]).strip()
            if statement:
                parsed.statements.append(statement)
        elif section.startswith("relevant history"):
            if match := _HISTORY_HEAD_RE.match(line):
                parsed.history.append(
                    HistoryItem(
                        kind=(match["type"] or "history").strip(),
                        project=match["project"].strip(),
                        when=match["when"].strip(),
                        what=match["what"].strip(),
                    )
                )

    if entities := _ENTITIES_RE.search(markdown):
        parsed.entities = [
            e.strip().lower() for e in re.split(r"[,;]", entities.group(1)) if e.strip()
        ]
    return parsed


def _tools_in(statement: str) -> list[str]:
    """Tool-like names in a statement: capitalised words that are not sentence openers."""
    found: list[str] = []
    for match in _PROPER_RE.finditer(statement):
        word = match.group(0)
        if word.lower() in _NOT_TOOLS or match.start() == 0:
            continue
        found.append(word.rstrip(".,"))
    return found


def derive_terms(bundles: list[ParsedBundle]) -> tuple[list[str], list[str], list[str]]:
    """Interests, things moved away from, and recent project names."""
    interests: list[str] = []
    avoid: list[str] = []
    projects: list[str] = []

    for bundle in bundles:
        for entity in bundle.entities:
            if entity not in interests:
                interests.append(entity)
        for statement in bundle.statements:
            abandoned = bool(_ABANDON_RE.search(statement))
            for tool in _tools_in(statement):
                lowered = tool.lower()
                if abandoned:
                    if lowered not in avoid:
                        avoid.append(lowered)
                elif lowered not in interests:
                    interests.append(lowered)
        for item in bundle.history:
            if item.project not in projects:
                projects.append(item.project)

    # A tool cannot be both an interest and something moved away from; the explicit
    # "abandoned" statement wins, since a person usually mentions a tool in the other
    # sentence only to say what they replaced it with.
    interests = [i for i in interests if i not in avoid]
    return interests[:60], avoid[:30], projects[:12]


class ProjectMindClient:
    """Calls the ProjectMind CLI. The only code that touches the process boundary."""

    def __init__(self, command: list[str], project_path: str, timeout: int = QUERY_TIMEOUT_SECONDS):
        self.command = command
        self.project_path = project_path
        self.timeout = timeout

    def query(self, prompt: str) -> str:
        completed = subprocess.run(
            [*self.command, "context", prompt, "--project", self.project_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"projectmind exited {completed.returncode}: {completed.stderr.strip()[:200]}"
            )
        return completed.stdout


def configured_command() -> list[str] | None:
    """The ProjectMind command, from config or PATH. None when it is not available."""
    settings = get_settings()
    if settings.projectmind_enabled is False:
        return None
    if settings.projectmind_cmd:
        return [part.strip('"') for part in shlex.split(settings.projectmind_cmd, posix=False)]
    found = shutil.which("projectmind")
    return [found] if found else None


def fetch_personal_context(
    client: ProjectMindClient | None = None,
    *,
    seeds: list[str] | None = None,
    now: datetime | None = None,
) -> PersonalContext | None:
    """Ask ProjectMind about the technologies in `seeds` and reduce the answers.

    Only its profile and history sections are used. The "From your own files and notes"
    section is raw text from unrelated documents (other projects, other people's work):
    it is not news-relevant and is never read here, stored, or sent to a model.

    Returns None when ProjectMind is absent or yields nothing usable.
    """
    if client is None:
        command = configured_command()
        if command is None:
            return None
        settings = get_settings()
        client = ProjectMindClient(command, settings.projectmind_project_path or ".")

    bundles: list[ParsedBundle] = []
    for prompt in profile_queries(seeds):
        try:
            bundles.append(parse_bundle(client.query(prompt)))
        except Exception as exc:
            log.warning("projectmind.query_failed", error=f"{type(exc).__name__}: {exc}"[:200])
            # The first failure usually means the tool is unavailable; stop asking.
            break

    usable = [b for b in bundles if not b.is_empty or b.entities]
    if not usable:
        log.info("projectmind.nothing_returned", asked=len(bundles))
        return None

    statements: list[str] = []
    history: list[HistoryItem] = []
    for bundle in usable:
        statements.extend(s for s in bundle.statements if s not in statements)
        history.extend(h for h in bundle.history if h not in history)

    interests, avoid, projects = derive_terms(usable)
    context = PersonalContext(
        statements=statements[:MAX_STATEMENTS],
        history=history[:MAX_HISTORY],
        interests=interests,
        avoid=avoid,
        projects=projects,
        fetched_at=now or datetime.now(UTC),
        queries_answered=len(usable),
    )
    log.info(
        "projectmind.context",
        statements=len(context.statements),
        interests=len(context.interests),
        avoid=len(context.avoid),
        projects=len(context.projects),
    )
    return None if context.is_empty else context
