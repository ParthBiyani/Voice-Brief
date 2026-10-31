# Ranking experiments

What was tried against the offline ranking harness (`make eval`), including what did not
work, so it is not tried again for the same reason. The harness is the free heuristic stage
on 3 personas, so one ranked position is worth 0.033 of precision@10 and differences of a
single position are noise.

## Where it stands (2026-10-31)

| Ranking path | Labels | P@10 | nDCG@10 |
|---|---|---:|---:|
| Heuristic (`make eval`) | rubric | 0.500 | 0.599 |
| Heuristic (`make eval`) | judge | 0.533 | 0.660 |
| Heuristic + model re-rank of the top 30 (`scripts/eval_rerank.py`) | rubric | 0.533 | 0.628 |
| Heuristic + model re-rank of the top 30 | judge | **0.600** | **0.730 to 0.756** |

Target 0.70, not met on either label set. The previous baseline was heuristic-only on the
rubric: P@10 0.533, nDCG@10 0.585, with no judge labels and no way to measure the re-rank.

## 2026-10-31: content topics, a second label set, and measuring the re-rank

**What was wrong.** The topic signal matched a story's *source tags* against the
listener's declared topics. Every arXiv cs.AI paper carries `agentic-ai, machine-learning,
research`, so a whole day of papers tied at exactly 3.80 or 2.20, and "I follow computer
vision" could not tell a vision paper from an agent paper. Separately, the dependency
`transformers` matched every paper that mentions the transformer architecture ("looped
transformers"). It was the top-ranked reason in 4 of the CV persona's top 8, and in the real
brief it produces a false "you use this".

**Changes.**
- `pipeline/topics.py` tags a story by its own words, in the product's topic vocabulary.
  The terms were written from that taxonomy, not copied from the labelling rubric. A
  content match counts fully (1.2); a declared topic the source is tagged with but the
  text does not mention counts 0.4. The same treatment applies to starred GitHub topics.
- `transformers`, `datasets`, `accelerate`, `evaluate` and `tokenizers` join the names
  that need package-style evidence (`huggingface/transformers`).
- The model re-rank scores the heuristic's top 30 instead of 15, in the same single call.
- The ranker and re-rank see the fuller of summary and fetched article. A Hacker News link
  used to be judged on its title alone.

**A second, independent label set.** `relevance.jsonl` is a keyword rubric (topic keywords
plus dependency substrings), so it favours a ranker that thinks in the same keywords and
it rewards the `transformers` false positive. `relevance_judge.jsonl`
(`scripts/build_judge_labels.py`) has a model (Sonnet 5.5) read each persona's
plain-language description and the 76 items, with no rubric, no tags and no ranker output.
It is stricter: 8 to 11 relevant per persona, against 10 to 21. A change is trusted only if
it holds on both.

| Heuristic | rubric P@10 | rubric nDCG | judge P@10 | judge nDCG |
|---|---:|---:|---:|---:|
| Before (source tags only) | 0.533 | 0.585 | 0.367 | 0.453 |
| + content topics | 0.600 | 0.690 | 0.400 | 0.526 |
| + `transformers` needs package evidence | 0.467 | 0.556 | 0.500 | 0.615 |
| + starred topics read from content (shipped) | 0.500 | 0.599 | 0.533 | 0.660 |

The rubric drop in row three is the rubric rewarding the false positive. Its CV labels
count any mention of "transformers" as relevant. The fix stays, and the rubric was not edited
to recover the points.

**Re-rank depth** (full path, judge labels, one run each, ±0.03 to 0.07 run-to-run):

| Re-ranked | P@10 | nDCG@10 | Cost per 3 personas |
|---:|---:|---:|---:|
| 15 (old) | 0.567 | 0.732 | ₹0.18 |
| 25 | 0.600 | 0.759 | ₹0.26 |
| 30 (shipped) | 0.667 | 0.814 | ₹0.30 |
| 40 | 0.633 | 0.766 | ₹0.39 |

These were measured before the starred-topic change. Re-measured after it, twice, at 30:
P@10 0.600, nDCG 0.730 and 0.756.

**Personalization ablation** (heuristic, P@10 / nDCG):

| Arm | rubric | judge |
|---|---|---|
| Stack profile + topics | 0.500 / 0.599 | 0.533 / 0.660 |
| Topics only | 0.500 / 0.548 | 0.533 / 0.606 |
| None | 0.133 / 0.188 | 0.033 / 0.084 |

**Not claimed.** The judge and the re-rank are both model judgements, so their agreement is
weaker evidence than agreement with people. With 3 personas and 28 judge positives, one
position is 0.033 of P@10. Tuning stopped here deliberately, because further gains on this
set would be fitting the set. The route to a real number is still real thumbs via `history
export`. The frozen corpus also predates article fetching, so Hacker News items here are
titles only, and part of the remaining miss is ones the shipped system now sees in full.

## Earlier baseline

precision@10 **0.533**, nDCG@10 **0.585** (heuristic, rubric labels).

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
