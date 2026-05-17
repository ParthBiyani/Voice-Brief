"""Evaluation harness.

`make eval` runs this. It executes the real pipeline components against the frozen
labelled sets, prints the metrics table, and diffs against the last recorded run so
every prompt or threshold change has a visible before/after.

Design constraints, all of them deliberate:

* **No network, no LLM.** The components measured here are the deterministic ones.
  A harness that costs money or needs an API key is a harness nobody runs.
* **Frozen inputs.** The corpus is committed alongside the labels. Re-running six
  weeks from now must produce the same numbers unless the code changed.
* **Diff, not just report.** A metric table with no baseline tells you nothing about
  whether the change you just made helped.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from voicebrief.evalkit.annotation import PERSONAS_BY_KEY, is_relevant
from voicebrief.evalkit.metrics import (
    MetricResult,
    adjusted_rand_index,
    clustered_fraction,
    homogeneity,
    ndcg_at_k,
    pairwise_prf,
    precision_at_k,
)
from voicebrief.pipeline.clustering import cluster_items, labels_from_clusters
from voicebrief.pipeline.dedup import DedupCandidate, find_duplicates, pairs_from_groups
from voicebrief.pipeline.embedding import get_embedding_service, item_text

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATASETS = ROOT / "eval" / "datasets"
RUNS = ROOT / "eval" / "runs"
BASELINE = ROOT / "eval" / "baseline.json"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(slots=True)
class Corpus:
    rows: list[dict]
    vectors: np.ndarray

    @property
    def by_id(self) -> dict[str, dict]:
        return {r["id"]: r for r in self.rows}


def load_corpus() -> Corpus:
    rows = _read_jsonl(DATASETS / "corpus.jsonl")
    service = get_embedding_service()
    vectors = service.encode([item_text(r["title"], r.get("summary", "")) for r in rows])
    return Corpus(rows=rows, vectors=vectors)


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────
def eval_dedup(corpus: Corpus, *, threshold: float = 0.92) -> list[MetricResult]:
    labels = _read_jsonl(DATASETS / "dedup_pairs.jsonl")
    index = {r["id"]: i for i, r in enumerate(corpus.rows)}

    # Only score pairs whose members are both in the frozen corpus.
    truth = {
        frozenset((r["a_id"], r["b_id"]))
        for r in labels
        if r["duplicate"] and r["a_id"] in index and r["b_id"] in index
    }
    judged = {
        frozenset((r["a_id"], r["b_id"]))
        for r in labels
        if r["a_id"] in index and r["b_id"] in index
    }

    candidates = [
        DedupCandidate(
            id=r["id"],
            title=r["title"],
            url=r["url"],
            trust_weight=0.5,
            engagement=0.0,
            published_ts=0.0,
        )
        for r in corpus.rows
    ]
    groups = find_duplicates(candidates, corpus.vectors, threshold=threshold)

    # Restrict predictions to the judged pairs. Scoring predictions on unjudged pairs
    # would count every unlabelled correct grouping as a false positive.
    predicted = pairs_from_groups(groups) & judged
    result = pairwise_prf(predicted, truth)

    return [
        MetricResult(
            "Deduplication", "pairwise precision", result.precision, target=0.90,
            detail=result.as_dict(),
        ),
        MetricResult(
            "Deduplication", "pairwise recall", result.recall, target=0.80,
            detail={"judged_pairs": len(judged), "true_pairs": len(truth)},
        ),
    ]


def eval_clustering(corpus: Corpus) -> list[MetricResult]:
    labels = _read_jsonl(DATASETS / "cluster_labels.jsonl")
    truth_by_id = {r["id"]: r["topic"] for r in labels if r["topic"] != "other"}

    index = {r["id"]: i for i, r in enumerate(corpus.rows)}
    scored_ids = [i for i in truth_by_id if i in index]

    import uuid as _uuid

    fake_ids = [_uuid.UUID(int=i) for i in range(len(scored_ids))]
    vectors = np.stack([corpus.vectors[index[i]] for i in scored_ids])

    clusters = cluster_items(fake_ids, vectors)
    predicted_labels = labels_from_clusters(clusters)

    topic_ids = {topic: n for n, topic in enumerate(sorted(set(truth_by_id.values())))}
    predicted = {sid: predicted_labels[fid] for sid, fid in zip(scored_ids, fake_ids, strict=True)}
    truth = {sid: topic_ids[truth_by_id[sid]] for sid in scored_ids}

    detail = {
        "scored_items": len(scored_ids),
        "true_topics": len(topic_ids),
        "predicted_clusters": len(clusters),
        "multi_item_clusters": sum(1 for c in clusters if c.size > 1),
    }

    # Primary metric. See metrics.homogeneity for why ARI alone is misleading here:
    # story-level clusters against subject-area labels score near zero on ARI even
    # when every cluster is topically correct.
    return [
        MetricResult(
            "Clustering", "homogeneity", homogeneity(predicted, truth), target=0.75,
            detail=detail,
        ),
        MetricResult(
            "Clustering", "clustered fraction", clustered_fraction(predicted), target=None,
            detail={"note": "guards against purity gamed by refusing to cluster"},
        ),
        MetricResult(
            "Clustering", "adjusted rand index", adjusted_rand_index(predicted, truth),
            target=None,
            detail={"note": "granularity mismatch: stories vs subject areas, see metrics.py"},
        ),
    ]


def eval_ranking(corpus: Corpus, *, k: int = 10) -> list[MetricResult]:
    """Rank each persona's candidate pool with the deterministic scorer.

    This measures the non-LLM ranking path. The LLM re-rank sits on top of it and is
    evaluated separately once a provider is configured; keeping this path measurable
    on its own is what makes the personalization ablation possible.
    """
    labels = _read_jsonl(DATASETS / "relevance.jsonl")
    by_id = corpus.by_id

    precisions, ndcgs, per_persona = [], [], {}
    for persona_key, persona in PERSONAS_BY_KEY.items():
        rows = [r for r in labels if r["persona"] == persona_key and r["item_id"] in by_id]
        if not rows:
            continue

        relevant = {r["item_id"] for r in rows if r["relevant"]}
        scored = []
        for row in rows:
            item = by_id[row["item_id"]]
            scored.append((row["item_id"], _persona_score(persona, item)))
        ranked = [i for i, _ in sorted(scored, key=lambda p: p[1], reverse=True)]

        p_at_k = precision_at_k(ranked, relevant, k)
        ndcg = ndcg_at_k(ranked, relevant, k)
        precisions.append(p_at_k)
        ndcgs.append(ndcg)
        per_persona[persona_key] = {
            "p_at_k": round(p_at_k, 4),
            "ndcg": round(ndcg, 4),
            "pool": len(rows),
            "relevant": len(relevant),
        }

    return [
        MetricResult(
            "Ranking", f"precision@{k}", float(np.mean(precisions)) if precisions else 0.0,
            target=0.70, detail=per_persona,
        ),
        MetricResult(
            "Ranking", f"nDCG@{k}", float(np.mean(ndcgs)) if ndcgs else 0.0,
            target=None, detail=None,
        ),
    ]


def _persona_score(persona, item: dict) -> float:
    """Deterministic relevance scorer: topic overlap plus dependency mentions.

    Intentionally the same shape as the production pre-LLM ranking signal, so the
    number this harness reports is the number the system actually achieves without a
    model in the loop.
    """
    text = f"{item['title']} {item.get('summary', '')}".lower()
    topics = {t.lower() for t in item.get("topics", [])}

    score = 0.0
    score += 0.5 * len(topics & {t.replace("-", "") for t in persona.topics})
    score += 0.5 * len(topics & persona.topics)
    score += 1.5 * sum(1 for dep in persona.dependencies if dep in text)
    score += 0.4 * sum(1 for lang in persona.languages if lang in text)
    score += 0.3 * float(item.get("engagement", 0.0) or 0.0)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 — a missing git is not a harness failure
        return "unknown"


def run_all(*, threshold: float = 0.92) -> dict:
    started = time.time()
    corpus = load_corpus()

    results: list[MetricResult] = []
    results += eval_dedup(corpus, threshold=threshold)
    results += eval_clustering(corpus)
    results += eval_ranking(corpus)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "corpus_size": len(corpus.rows),
        "dedup_threshold": threshold,
        "elapsed_seconds": round(time.time() - started, 2),
        "metrics": [r.as_dict() for r in results],
    }


def _format_table(report: dict, baseline: dict | None) -> str:
    base = {}
    if baseline:
        base = {(m["component"], m["metric"]): m["value"] for m in baseline["metrics"]}

    # ASCII only: Windows consoles default to cp1252 and a harness that raises
    # UnicodeEncodeError on its own output table is a harness nobody runs.
    header = f"{'Component':<16} {'Metric':<22} {'Value':>8} {'Target':>8} {'Change':>9}  Status"
    lines = [header, "-" * len(header)]

    for metric in report["metrics"]:
        key = (metric["component"], metric["metric"])
        target = f"{metric['target']:.2f}" if metric["target"] is not None else "n/a"

        delta = "n/a"
        if key in base:
            change = metric["value"] - base[key]
            delta = "  same" if abs(change) < 5e-4 else f"{change:+.4f}"

        status = "n/a"
        if metric["passed"] is True:
            status = "PASS"
        elif metric["passed"] is False:
            status = "FAIL"

        lines.append(
            f"{metric['component']:<16} {metric['metric']:<22} "
            f"{metric['value']:>8.4f} {target:>8} {delta:>9}  {status}"
        )

    lines.append("")
    lines.append(
        f"corpus={report['corpus_size']} items  sha={report['git_sha']}  "
        f"{report['elapsed_seconds']}s"
    )
    if baseline:
        lines.append(f"baseline: {baseline['generated_at']} (sha {baseline['git_sha']})")
    else:
        lines.append("baseline: none recorded — run with --record to set one")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VoiceBrief evaluation harness")
    parser.add_argument("--compare", action="store_true", help="Diff against the baseline")
    parser.add_argument("--record", action="store_true", help="Write this run as the baseline")
    parser.add_argument("--threshold", type=float, default=0.92, help="Dedup cosine threshold")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args()

    report = run_all(threshold=args.threshold)

    baseline = None
    if (args.compare or args.record) and BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_table(report, baseline))

    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RUNS / f"{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.record:
        BASELINE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nbaseline updated -> {BASELINE.relative_to(ROOT)}")

    failed = [m for m in report["metrics"] if m["passed"] is False]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
