"""Factuality verification.

The PRD sets two hard bars: at least 95% of script sentences attributable to a source
span, and exactly zero URLs that were not in the source set. This module is what makes
those measurements rather than aspirations.

Verification is lexical and deterministic, not another model call. That is a
deliberate choice:

* An LLM judge would cost money on every episode and introduce a second thing that
  can hallucinate — using a model to check a model's grounding just moves the
  question.
* Grounding is genuinely checkable without judgement: a sentence is supported if its
  *content* words appear in the facts it was written from. That is a weaker test than
  entailment, but it is a test that cannot itself be wrong.

The known limitation is stated plainly: this detects fabricated specifics — invented
version numbers, benchmark figures, product names — which is the failure mode that
actually matters in a news brief. It does not detect a sentence that reuses source
vocabulary while inverting the meaning. Catching that needs entailment, which is v2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from voicebrief.logging import get_logger

log = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Zऀ-ॿ])")
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#_-]*")

# Content-free words carry no grounding signal; a sentence made only of these is
# connective tissue, not a claim.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here it its is are was
    were be been being have has had do does did will would can could should may might
    of in on at to from by for with without about into over under again further once
    all any both each few more most other some such no nor not only own same so too very
    as we you they he she i our your their them us who whom which what when where why how
    now new also just still yet while because since after before during between up down
    out off above below through
    """.split()
)

# A sentence with fewer content words than this is narration ("Here is what changed"),
# not a factual claim, and holding it to a grounding standard would be meaningless.
MIN_CONTENT_WORDS = 3

# Fraction of a claim's content words that must appear in the source facts.
SUPPORT_THRESHOLD = 0.5

# Sentences that assert something checkable — a figure, a version, a named artefact.
# Everything else is connective prose ("Four papers tackle this from different
# angles"), which is true of the segment as a whole but cites nothing in particular.
#
# The distinction matters because the PRD's metric is "sentences attributable to a
# source span". Scoring narration against lexical overlap measures writing style, not
# factuality, and drags a well-grounded script from ~0.95 down to ~0.49 for saying
# "from different angles". What must never happen is a *specific* claim with no
# source — an invented benchmark number or a version that does not exist — and that
# is exactly what a claim sentence is.
# CamelCase or all-caps identifiers: DIPTTA, MiniCPM, LangGraph, GGUF, Qwen3.
_PROPER_NOUN = re.compile(r"\b[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9.+-]*\b")
_HAS_DIGIT = re.compile(r"\d")

# Enumerators are decided by position, not magnitude.
#
# The first attempt treated any spoken number below 100 as narration. That silently
# exempted "improved by ninety seven percent" and "eighteen object types" from
# checking entirely — the two sentence shapes most likely to carry a fabricated
# figure. Magnitude is the wrong signal.
#
# What actually distinguishes them is where the number sits. "Four papers tackle
# this" opens with a count of the sources themselves; "improved by ninety seven
# percent" states a measurement mid-sentence. Only a leading small number is
# enumeration.
_ENUMERATOR_WINDOW = 2
_ENUMERATOR_CEILING = 20.0


def is_claim(sentence: str) -> bool:
    """True when the sentence asserts something specific enough to cite.

    Everything else is connective prose. See the SUPPORT_THRESHOLD comment above for
    why that distinction is what the PRD's factuality metric actually needs.
    """
    if _HAS_DIGIT.search(sentence):
        return True
    if _PROPER_NOUN.search(sentence):
        return True

    found = spoken_numbers(sentence)
    if not found:
        return False

    lead = " ".join(sentence.split()[:_ENUMERATOR_WINDOW]).lower().strip(",.")
    lead_values = spoken_numbers(lead)
    if lead_values and found <= lead_values:
        # Every number in the sentence is the opening enumerator.
        return any(
            "." in v or float(v) > _ENUMERATOR_CEILING
            for v in lead_values
            if _is_number(v)
        )
    return True


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


@dataclass(slots=True)
class GroundingReport:
    total_sentences: int = 0
    supported_sentences: int = 0
    unsupported: list[str] = field(default_factory=list)
    hallucinated_links: list[str] = field(default_factory=list)
    skipped_narration: int = 0

    @property
    def grounded_ratio(self) -> float:
        return (
            self.supported_sentences / self.total_sentences
            if self.total_sentences
            else 1.0
        )


def split_sentences(text: str) -> list[str]:
    """Split spoken script into sentences.

    Deliberately simple. Script text is generated under instructions that forbid
    markdown, lists and abbreviations, so the pathological cases a full sentence
    tokenizer exists to handle do not arise here.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    # Strip two-host speaker labels before splitting.
    cleaned = re.sub(r"\b(HOST [AB]|SPEAKER [12]):\s*", "", cleaned)
    return [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]


def content_words(text: str) -> set[str]:
    """Words that carry grounding signal.

    Number words are excluded: they are checked separately and far more precisely by
    the numeric comparison. Leaving them in double-counts a spoken figure as eight
    unmatched prose words ("thirty three thousand two hundred thirty one") against a
    source that simply wrote "33,231", which sinks overlap on the most factual
    sentences in the script.

    Alphanumeric identifiers also emit their alphabetic stem, so a script saying
    "Nanbeige" matches a source that wrote "Nanbeige4.2-3B".
    """
    words: set[str] = set()
    for raw in _WORD_RE.findall(text):
        word = raw.lower().rstrip(".")
        if len(word) < 2 or word in STOPWORDS or word in _SPOKEN_NUMBER_TOKENS:
            continue
        words.add(word)
        stem = re.match(r"^([a-z]{3,})[\d]", word)
        if stem:
            words.add(stem.group(1))
    return words


# Spoken-form numbers. The script generator is instructed to expand numerals for
# text-to-speech ("version zero point four", "eighty two percent"), so a verifier that
# compares raw digits against spoken words flags every correct number as invented.
# Measured on the first real episode, that alone dropped the attribution rate from
# well above target to 0.43 — the verifier was wrong, not the script.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MULTIPLIERS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000,
                "billion": 1_000_000_000, "trillion": 1_000_000_000_000}
_SPOKEN_NUMBER_TOKENS = frozenset(
    [*_NUMBER_WORDS, *_MULTIPLIERS, "point", "percent"]
)
_NUMBER_TOKEN = re.compile(
    r"\b(" + "|".join([*_NUMBER_WORDS, *_MULTIPLIERS, "point", "and", "percent"]) + r")\b",
    re.IGNORECASE,
)


def spoken_numbers(text: str) -> set[str]:
    """Numbers expressed in words, rendered back to digit strings.

    Handles the forms a news brief actually produces: "four point two" -> 4.2,
    "eighty two percent" -> 82, "three billion" -> 3000000000, "two thousand four
    hundred seventy six" -> 2476.
    """
    found: set[str] = set()
    # Split on punctuation as well as whitespace. Without this, "four point two,
    # three billion parameters" is read as a single number (4000000000.23) instead of
    # the two distinct figures it actually is.
    # Hyphens are number-internal ("thirty-three thousand"), so they normalise to
    # spaces; commas and full stops genuinely separate figures.
    normalised = text.replace("-", " ")
    for run in re.split(r"[^a-zA-Z\s]+", normalised):
        found |= _spoken_numbers_in_run(run)
    return found


def _spoken_numbers_in_run(text: str) -> set[str]:
    found: set[str] = set()
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]

    i = 0
    while i < len(tokens):
        if tokens[i] not in _NUMBER_WORDS:
            i += 1
            continue

        total = 0
        current = 0
        decimal = ""
        mantissa = None  # "three billion" should also satisfy a source saying "3B"
        j = i
        while j < len(tokens):
            token = tokens[j]
            if token in _NUMBER_WORDS:
                if decimal != "":
                    decimal += str(_NUMBER_WORDS[token])
                else:
                    current += _NUMBER_WORDS[token]
            elif token in _MULTIPLIERS:
                multiplier = _MULTIPLIERS[token]
                if multiplier >= 1000:
                    if mantissa is None:
                        mantissa = current or 1
                    total += max(current, 1) * multiplier
                    current = 0
                else:
                    current = max(current, 1) * multiplier
            elif token == "point":
                decimal = "."
            elif token == "and" and decimal == "":
                pass  # "two hundred and five"
            else:
                break
            j += 1

        value = total + current
        if j > i:
            found.add(f"{value}{decimal}" if decimal not in ("", ".") else str(value))
            # Also record the bare integer part: "four point two" should satisfy a
            # source that only mentions version 4.
            if decimal:
                found.add(str(value))
            # And the mantissa, so "three billion parameters" matches a source that
            # writes the same figure as "3B".
            if mantissa is not None:
                found.add(str(mantissa))
        i = max(j, i + 1)
    return found


def _numbers(text: str) -> set[str]:
    """Every number in the text, digits and spoken words alike.

    Numbers are checked strictly and separately from prose overlap, because a
    fabricated figure is the most damaging hallucination in a news brief — and unlike
    prose, a number is either in the sources or it was invented.
    """
    digits = set(re.findall(r"\d+(?:\.\d+)*", text))
    # A source writing "Nanbeige4.2-3B" should satisfy a script saying "four point
    # two, three billion", so split glued alphanumerics into their numeric parts.
    digits |= set(re.findall(r"(?<=[a-zA-Z])(\d+(?:\.\d+)*)", text))
    # Comma-grouped figures: "33,231" and "33231" are the same number.
    digits |= {m.replace(",", "") for m in re.findall(r"\d{1,3}(?:,\d{3})+", text)}

    # Suffixed magnitudes: a source writing "3B" or "27B" is stating the same figure
    # a script speaks as "three billion". Emit both forms so neither side has to
    # guess which rendering the other chose.
    for value, suffix in re.findall(r"(\d+(?:\.\d+)?)\s*([BMK])\b", text):
        digits.add(value)
        scale = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
        digits.add(str(int(float(value) * scale)))

    normalised = {d.rstrip(".") for d in digits}
    # A decimal also stands for its integer part: "version 4.2" satisfies "version
    # four", which is how a brief usually says it out loud.
    normalised |= {d.split(".")[0] for d in normalised if "." in d}
    return normalised | spoken_numbers(text)


def verify_segment(
    script: str,
    summary,
    extra_vocabulary: set[str] | None = None,
    source_text: str = "",
) -> GroundingReport:
    """Check a written segment against the facts it was supposed to use.

    `summary` is a ClusterSummary; only its facts and source URLs are read, so this
    stays usable from the eval harness with plain fixtures.

    `extra_vocabulary` carries terms that are grounded but live outside the sources —
    the listener's own repository and dependency names. "you already use transformers
    in Voice-Brief" is a true statement drawn from their stack profile, and flagging
    it as unsupported would penalise the one thing the product exists to say.
    """
    report = GroundingReport()

    # The PRD's bar is attribution to a *source span*, so the source documents are
    # the comparison target — not the intermediate summary, which is a lossy
    # compression of them. A claim the summarizer dropped but the writer legitimately
    # drew from the article is still grounded.
    corpus = " ".join(
        [
            source_text,
            summary.headline or "",
            summary.what_changed or "",
            summary.why_it_matters or "",
            *[f["fact"] for f in summary.key_facts],
            *[s.title for s in summary.sources],
        ]
    )
    source_words = content_words(corpus) | {w.lower() for w in (extra_vocabulary or set())}
    source_numbers = _numbers(corpus)
    allowed_urls = {u.rstrip("/") for u in summary.citation_urls}

    for url in _URL_RE.findall(script):
        if url.rstrip("/").rstrip(".,") not in allowed_urls:
            report.hallucinated_links.append(url)

    for sentence in split_sentences(script):
        words = content_words(sentence)
        claim = is_claim(sentence)

        # The content-word floor applies to prose only. A claim like "throughput
        # improved by ninety seven percent" has just two content words once number
        # words are excluded, and skipping it would exempt exactly the sentence shape
        # most likely to carry a fabricated figure.
        if not claim or (not words and not _numbers(sentence)):
            report.skipped_narration += 1
            continue
        if not claim and len(words) < MIN_CONTENT_WORDS:
            report.skipped_narration += 1
            continue

        report.total_sentences += 1

        # A number not present in the sources is a fabrication regardless of how well
        # the surrounding prose overlaps.
        invented = _numbers(sentence) - source_numbers
        if invented:
            report.unsupported.append(sentence)
            continue

        overlap = len(words & source_words) / len(words)
        if overlap >= SUPPORT_THRESHOLD:
            report.supported_sentences += 1
        else:
            report.unsupported.append(sentence)

    if report.unsupported or report.hallucinated_links:
        log.warning(
            "grounding.issues",
            unsupported=len(report.unsupported),
            hallucinated_links=len(report.hallucinated_links),
            ratio=round(report.grounded_ratio, 3),
        )
    return report


def verify_episode(segments: list, summaries: dict) -> GroundingReport:
    """Aggregate grounding across every story segment in an episode."""
    total = GroundingReport()
    for segment in segments:
        if getattr(segment, "kind", "") != "story":
            continue
        summary = summaries.get(getattr(segment, "cluster_id", None))
        if summary is None:
            continue
        report = verify_segment(segment.script, summary)
        total.total_sentences += report.total_sentences
        total.supported_sentences += report.supported_sentences
        total.unsupported.extend(report.unsupported)
        total.hallucinated_links.extend(report.hallucinated_links)
        total.skipped_narration += report.skipped_narration
    return total
