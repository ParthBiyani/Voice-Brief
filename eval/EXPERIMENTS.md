# Ranking experiments

What was tried against the offline ranking harness (`make eval`), including what did not
work, so it is not tried again for the same reason. The harness is the free heuristic stage
on 3 personas, so one ranked position is worth 0.033 of precision@10 and differences of a
single position are noise.

## Baseline

precision@10 **0.533**, nDCG@10 **0.585** (target 0.70).

## 2026-10-31: semantic similarity to the listener's profile

**Idea.** Many candidates tie (the same generic topic tags give 3.8 or 2.2), and nothing
in the heuristic reads the story's content. Add a signal: cosine similarity between the
story (title and summary, embedded with the existing bge-small model) and a text built from
the listener's libraries, languages, starred topics and declared topics, min-max normalised
over the pool and weighted 1 to 4. Tried with and without BGE's query instruction prefix.

| Variant | Weight | P@10 | nDCG@10 |
|---|---:|---:|---:|
| none (baseline) | 0 | **0.533** | **0.585** |
| plain query | 1 to 4 | 0.467 | 0.537 to 0.555 |
| with query prefix | 1 to 4 | 0.467 | 0.562 to 0.575 |

**Result: rejected.** It lowered precision at every weight and nDCG at every weight, and
is not shipped.

**Why, as far as can be told.** The relevance labels come from a documented rubric over
topic tags and dependencies (`scripts/build_eval_sets.py`), so a tag-based heuristic is
structurally favoured and a content-based signal can only disagree with it. That does not
show the signal is bad for a real reader. It shows this harness cannot tell. Judging it
needs labels that do not come from the same tags: the route is `history export` plus real
thumbs (see `personalization/feedback.py`), after which this experiment is worth repeating.

**Not claimed.** That semantic similarity helps or hurts the daily brief. Only that it did
not improve this measurement, and that the measurement has a known bias.
