/**
 * Sentence and word timing for click-to-play transcripts.
 *
 * Sentence times are measured: the server records where each rendered sentence starts and
 * ends. A speech engine reports no times for the words inside a sentence, so those are
 * shared out by length. That is close enough to land a click on the right word or the one
 * beside it, which is why the player lets the listener choose between jumping by sentence
 * (exact) and by word (approximate).
 */

import type { Cue } from "./api";

export interface TimedWord {
  text: string;
  start: number;
  end: number;
}

export interface Turn {
  speaker: string;
  cues: Cue[];
}

/** Words of a sentence with times spread by length, so longer words take longer. */
export function wordTimes(cue: Cue): TimedWord[] {
  const words = cue.text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  const weights = words.map((word) => word.length + 1);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const span = Math.max(cue.end - cue.start, 0);
  let cursor = cue.start;
  return words.map((text, index) => {
    const start = cursor;
    cursor += (span * weights[index]) / total;
    return { text, start, end: index === words.length - 1 ? cue.end : cursor };
  });
}

/** Consecutive sentences by one speaker, as one paragraph. */
export function groupTurns(cues: Cue[]): Turn[] {
  const turns: Turn[] = [];
  for (const cue of cues) {
    const last = turns[turns.length - 1];
    if (last && last.speaker === cue.speaker) last.cues.push(cue);
    else turns.push({ speaker: cue.speaker, cues: [cue] });
  }
  return turns;
}

/** The sentence being spoken at `position`, or -1 between sentences and outside the segment. */
export function activeCue(cues: Cue[], position: number): number {
  return cues.findIndex((cue) => position >= cue.start && position < cue.end);
}

/** Where a click on a word should start the audio. A hair early, so the word is not clipped. */
export function seekTarget(start: number): number {
  return Math.max(start - 0.05, 0);
}
