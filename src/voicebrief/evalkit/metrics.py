"""Evaluation metrics.

Implemented directly rather than pulled from a library, because the definitions here
have to match what the PRD's target table means:

- Dedup is scored on **pairs**, not on groups. Two systems can produce different
  groupings that imply the same pairwise decisions, and pairwise P/R is the measure
  that doesn't punish a harmless difference in how a group was assembled.
- Clustering is scored primarily by **homogeneity**, not ARI. ARI is reported too,
  but it compares partitions at a fixed granularity and the system deliberately
  clusters at story level while the labels are subject-area level. See `homogeneity`
  for the full argument — this distinction was found by the metric reading 0.019 on
  data whose clusters were, on inspection, correct.
- Ranking uses nDCG@k with binary gains, since the relevance labels are binary.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class PRF:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }


def pairwise_prf(predicted: set[frozenset], truth: set[frozenset]) -> PRF:
    """Precision/recall/F1 over unordered duplicate pairs."""
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def adjusted_rand_index(
    predicted: dict[Hashable, int], truth: dict[Hashable, int]
) -> float:
    """ARI over the items present in both labelings.

    Chance-corrected: 0.0 is what random assignment scores, 1.0 is a perfect match.
    Negative values mean worse than random.
    """
    shared = sorted(set(predicted) & set(truth), key=repr)
    n = len(shared)
    if n < 2:
        return 1.0 if n else 0.0

    contingency: dict[tuple[int, int], int] = {}
    row_totals: dict[int, int] = {}
    col_totals: dict[int, int] = {}
    for key in shared:
        p, t = predicted[key], truth[key]
        contingency[(p, t)] = contingency.get((p, t), 0) + 1
        row_totals[p] = row_totals.get(p, 0) + 1
        col_totals[t] = col_totals.get(t, 0) + 1

    def choose2(x: int) -> float:
        return x * (x - 1) / 2.0

    sum_cells = sum(choose2(v) for v in contingency.values())
    sum_rows = sum(choose2(v) for v in row_totals.values())
    sum_cols = sum(choose2(v) for v in col_totals.values())
    total = choose2(n)

    expected = sum_rows * sum_cols / total
    maximum = (sum_rows + sum_cols) / 2.0
    if maximum == expected:
        return 1.0 if sum_cells == expected else 0.0
    return (sum_cells - expected) / (maximum - expected)


def precision_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(ranked)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def recall_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(list(ranked)[:k])
    return len(top & relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Normalized DCG with binary gains and log2 discount.

    The ideal ranking puts every relevant item first, so IDCG is the DCG of
    min(len(relevant), k) items at ranks 1..k.
    """
    if not relevant or k <= 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(list(ranked)[:k])
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def attribution_rate(supported: int, total: int) -> float:
    """Fraction of script sentences traceable to a source span."""
    return supported / total if total else 1.0


@dataclass(slots=True)
class MetricResult:
    """One row of the eval table."""

    component: str
    metric: str
    value: float
    target: float | None = None
    higher_is_better: bool = True
    detail: dict | None = None

    @property
    def passed(self) -> bool | None:
        if self.target is None:
            return None
        return self.value >= self.target if self.higher_is_better else self.value <= self.target

    def as_dict(self) -> dict:
        return {
            "component": self.component,
            "metric": self.metric,
            "value": round(self.value, 4),
            "target": self.target,
            "higher_is_better": self.higher_is_better,
            "passed": self.passed,
            "detail": self.detail or {},
        }


def homogeneity(predicted: dict[Hashable, int], truth: dict[Hashable, int]) -> float:
    """Fraction of items whose predicted cluster is topically pure.

    Why this exists alongside ARI: the two answer different questions, and using ARI
    alone produced a misleading 0.019 on real data.

    Clustering here groups *stories* — a paper, the release implementing it, and the
    thread discussing it — which yields many small clusters. The annotation rubric
    labels *subject areas*, which are large. ARI compares partitions directly and so
    scores correct story-level granularity as catastrophic over-segmentation.

    Homogeneity asks the question that actually matters for a brief: when the system
    puts items together, do they belong together? A cluster is pure when every member
    shares one rubric topic. Singletons are trivially pure and are excluded, so this
    cannot be gamed by refusing to cluster.
    """
    groups: dict[int, list[Hashable]] = {}
    for key, label in predicted.items():
        if key in truth:
            groups.setdefault(label, []).append(key)

    multi = [members for members in groups.values() if len(members) > 1]
    if not multi:
        return 0.0

    pure_items = 0
    total_items = 0
    for members in multi:
        topics = [truth[m] for m in members]
        majority = max(set(topics), key=topics.count)
        pure_items += sum(1 for t in topics if t == majority)
        total_items += len(topics)
    return pure_items / total_items if total_items else 0.0


def clustered_fraction(predicted: dict[Hashable, int]) -> float:
    """Share of items that landed in a multi-item cluster.

    Reported next to homogeneity because the two trade off: a system can reach perfect
    purity by clustering almost nothing. Together they are honest; separately, either
    can be gamed.
    """
    if not predicted:
        return 0.0
    counts: dict[int, int] = {}
    for label in predicted.values():
        counts[label] = counts.get(label, 0) + 1
    return sum(c for c in counts.values() if c > 1) / len(predicted)
