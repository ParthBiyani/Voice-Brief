import { describe, expect, it } from "vitest";
import { activeCue, groupTurns, seekTarget, wordTimes } from "./cues";

const cue = (speaker: string, text: string, start: number, end: number) => ({
  speaker,
  text,
  start,
  end,
});

describe("wordTimes", () => {
  it("covers the sentence exactly, in order", () => {
    const words = wordTimes(cue("", "Streaming is now the default", 10, 12.5));
    expect(words.map((w) => w.text)).toEqual(["Streaming", "is", "now", "the", "default"]);
    expect(words[0].start).toBe(10);
    expect(words[words.length - 1].end).toBe(12.5);
    for (let i = 1; i < words.length; i++) {
      expect(words[i].start).toBeCloseTo(words[i - 1].end, 6);
    }
  });

  it("gives a longer word more time", () => {
    const [short, long] = wordTimes(cue("", "a extraordinarily", 0, 3));
    expect(long.end - long.start).toBeGreaterThan(short.end - short.start);
  });

  it("handles an empty sentence", () => {
    expect(wordTimes(cue("", "   ", 0, 1))).toEqual([]);
  });
});

describe("groupTurns", () => {
  it("joins consecutive sentences by one speaker and splits on a change", () => {
    const turns = groupTurns([
      cue("A", "What changed?", 0, 1),
      cue("B", "Streaming.", 1.4, 2),
      cue("B", "It is the default.", 2.2, 3),
      cue("A", "Okay.", 3.2, 3.5),
    ]);
    expect(turns.map((t) => [t.speaker, t.cues.length])).toEqual([
      ["A", 1],
      ["B", 2],
      ["A", 1],
    ]);
  });
});

describe("activeCue", () => {
  const cues = [cue("", "One.", 0, 1), cue("", "Two.", 1.5, 2.5)];
  it("finds the sentence being spoken", () => {
    expect(activeCue(cues, 0.5)).toBe(0);
    expect(activeCue(cues, 2)).toBe(1);
  });
  it("is -1 in the pause between sentences", () => {
    expect(activeCue(cues, 1.2)).toBe(-1);
  });
});

describe("seekTarget", () => {
  it("starts a hair early and never before zero", () => {
    expect(seekTarget(10)).toBeCloseTo(9.95, 6);
    expect(seekTarget(0.01)).toBe(0);
  });
});
