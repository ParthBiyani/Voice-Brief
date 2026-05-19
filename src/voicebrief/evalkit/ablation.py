"""Ablations.

The PRD asks one question that a metrics table alone cannot answer: *does
personalization actually help, or does it just feel like it does?* An ablation is the
only honest way to answer that — run the same ranker on the same data with the stack
profile removed, and compare.

Run with `python -m voicebrief.evalkit.ablation`. Deterministic, offline, no spend.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import uuid

from voicebrief.evalkit.annotation import PERSONAS, topic_for
from voicebrief.evalkit.metrics import ndcg_at_k, precision_at_k
from voicebrief.personalization.github_profile import StackProfileData
from voicebrief.pipeline.ranking import RankCandidate, prerank

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATASETS = ROOT / "eval" / "datasets"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def persona_profile(persona) -> StackProfileData:
    """Turn a synthetic persona into the same shape the GitHub builder produces, so
    the ablation exercises the real ranking code rather than a parallel path."""
    return StackProfileData(
        login=persona.key,
        languages=dict.fromkeys(persona.languages, 1),
        dependencies={
            dep: {"repos": [f"{persona.key}/project"], "ecosystem": "pypi"}
            for dep in persona.dependencies
        },
        repos=[f"{persona.key}/project"],
        starred_topics=sorted(persona.declared_topics),
    )


def run_ablation(k: int = 10) -> dict:
    corpus = {r["id"]: r for r in _read_jsonl(DATASETS / "corpus.jsonl")}
    labels = _read_jsonl(DATASETS / "relevance.jsonl")

    results = {}
    for persona in PERSONAS:
        rows = [r for r in labels if r["persona"] == persona.key and r["item_id"] in corpus]
        if not rows:
            continue

        relevant = {r["item_id"] for r in rows if r["relevant"]}
        candidates = []
        id_by_cluster = {}
        for row in rows:
            item = corpus[row["item_id"]]
            cluster_id = uuid.uuid4()
            id_by_cluster[cluster_id] = row["item_id"]
            candidates.append(
                RankCandidate(
                    cluster_id=cluster_id,
                    title=item["title"],
                    summary=item.get("summary", ""),
                    url=item["url"],
                    topics=item.get("topics", []),
                    engagement=float(item.get("engagement", 0.0) or 0.0),
                    trust_weight=0.6,
                    cluster_size=1,
                    source_slug=item.get("source", ""),
                )
            )

        profile = persona_profile(persona)
        arms = {
            # Full system: stack profile plus declared topics.
            "with_profile": (profile, persona.declared_topics),
            # Topics only — what a system without GitHub access can do. Uses the same
            # source-vocabulary topics so the control arm is genuinely competitive.
            "topics_only": (None, persona.declared_topics),
            # Neither: recency, trust and popularity alone.
            "no_personalization": (None, set()),
        }

        persona_result = {}
        for arm, (arm_profile, arm_topics) in arms.items():
            ranked = prerank(candidates, profile=arm_profile, declared_topics=set(arm_topics))
            order = [id_by_cluster[s.candidate.cluster_id] for s in ranked]
            persona_result[arm] = {
                "p_at_k": round(precision_at_k(order, relevant, k), 4),
                "ndcg": round(ndcg_at_k(order, relevant, k), 4),
            }
        persona_result["pool"] = len(rows)
        persona_result["relevant"] = len(relevant)
        results[persona.key] = persona_result

    arms = ["with_profile", "topics_only", "no_personalization"]
    summary = {
        arm: {
            metric: round(
                sum(r[arm][metric] for r in results.values()) / max(len(results), 1), 4
            )
            for metric in ("p_at_k", "ndcg")
        }
        for arm in arms
    }
    return {"k": k, "per_persona": results, "mean": summary}


def format_table(report: dict) -> str:
    k = report["k"]
    lines = [
        f"Ablation: does the stack profile help? (k={k})",
        "",
        f"{'Arm':<22} {'P@' + str(k):>8} {'nDCG@' + str(k):>10} {'vs none':>10}",
        "-" * 54,
    ]
    baseline = report["mean"]["no_personalization"]["p_at_k"]
    for arm in ("with_profile", "topics_only", "no_personalization"):
        row = report["mean"][arm]
        delta = row["p_at_k"] - baseline
        change = "  baseline" if arm == "no_personalization" else f"{delta:+.4f}"
        lines.append(f"{arm:<22} {row['p_at_k']:>8.4f} {row['ndcg']:>10.4f} {change:>10}")

    lines += ["", "Per persona (P@%d):" % k]
    for key, row in report["per_persona"].items():
        lines.append(
            f"  {key:<16} profile={row['with_profile']['p_at_k']:.3f}  "
            f"topics={row['topics_only']['p_at_k']:.3f}  "
            f"none={row['no_personalization']['p_at_k']:.3f}  "
            f"({row['relevant']}/{row['pool']} relevant)"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the personalization ablation")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_ablation(k=args.k)
    print(json.dumps(report, indent=2) if args.json else format_table(report))

    out = ROOT / "eval" / "ablation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
