"""Measure the full ranking path (heuristic, then the model re-rank) offline.

`make eval` measures only the free heuristic stage, because the re-rank needs an API
key. This runs the production `prerank` then `rerank` on each persona's pool and scores
the final order against both label sets. Costs about ₹0.5 a run (three Haiku calls,
recorded in the cost ledger like any other call).

The judge labels and the re-rank are both model judgements, so their agreement is not
independent evidence in the way the rubric is; the two numbers are read together.

Usage:
    python scripts/eval_rerank.py
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import numpy as np

from voicebrief.db import session_scope
from voicebrief.evalkit.annotation import PERSONAS
from voicebrief.evalkit.metrics import ndcg_at_k, precision_at_k
from voicebrief.evalkit.run import DATASETS, _read_jsonl, load_corpus, persona_profile
from voicebrief.llm.client import LLMClient
from voicebrief.pipeline.ranking import RankCandidate, prerank, rerank

K = 10


def main() -> None:
    corpus = load_corpus()
    label_sets = {
        name: _read_jsonl(DATASETS / f)
        for name, f in (("rubric", "relevance.jsonl"), ("judge", "relevance_judge.jsonl"))
    }
    orders: dict[str, tuple[list[str], list[str]]] = {}
    with session_scope() as session:
        client = LLMClient(session)
        for persona in PERSONAS:
            rows = [r for r in label_sets["rubric"] if r["persona"] == persona.key]
            candidates, ids = [], {}
            for row in rows:
                item = corpus.by_id[row["item_id"]]
                cid = uuid.uuid4()
                ids[cid] = row["item_id"]
                candidates.append(
                    RankCandidate(
                        cluster_id=cid,
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
            first = prerank(
                candidates, profile=profile, declared_topics=set(persona.declared_topics)
            )
            heuristic = [ids[s.candidate.cluster_id] for s in first]
            final = rerank(first, client, profile=profile)
            orders[persona.key] = (heuristic, [ids[s.candidate.cluster_id] for s in final])
        cost = client.run_cost_inr

    result: dict = {"at": datetime.now(UTC).isoformat(), "cost_inr": round(cost, 3)}
    for name, labels in label_sets.items():
        for stage, which in (("heuristic", 0), ("with re-rank", 1)):
            ps, ns = [], []
            for persona in PERSONAS:
                relevant = {
                    r["item_id"] for r in labels if r["persona"] == persona.key and r["relevant"]
                }
                order = orders[persona.key][which]
                ps.append(precision_at_k(order, relevant, K))
                ns.append(ndcg_at_k(order, relevant, K))
            result[f"{name} / {stage}"] = {
                "p_at_10": round(float(np.mean(ps)), 4),
                "ndcg_at_10": round(float(np.mean(ns)), 4),
                "per_persona": [round(p, 2) for p in ps],
            }
            print(
                f"{name:7} {stage:13} P@10={np.mean(ps):.3f} nDCG@10={np.mean(ns):.3f} {[round(p, 2) for p in ps]}"
            )
    print(f"cost: ₹{cost:.2f}")
    out = DATASETS.parent / "runs" / f"rerank-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
