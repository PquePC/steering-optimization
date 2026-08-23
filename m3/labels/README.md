# Model labels for judge calibration (Phase −1)

110 responses from the 2026-08-14 Garlic probe, labelled by reading them, used to score the four
M3 judges before any of them runs on a pod.

## ⚠️ These were NOT written by a person

**Corrected 2026-08-23.** Everything in this repository described these as "hand labels", "one
careful reader's judgement" and "responses a human read". That was wrong. They were written by
**Claude Opus 5**, in commit `df7b76c`, which landed the labels, `m3/calibrate.py` and
`m3/scoring.py` together. The operator has confirmed doing no labelling. Nobody checked the
claim because the commit was authored under the operator's git identity, and the label notes
read like a person's.

The consequence is not that the labels are worthless — they are a careful read of 110
deliberately hard cases and every note on them is specific and checkable against the transcript.
The consequence is that **scoring a judge against them is agreement between two language models,
not accuracy**, and agreement cannot tell "both right" from "both wrong in the same direction".

That distinction is not hypothetical here. `openai/gpt-4.1-mini` scored `identify` at
kappa 1.000 and `effect` at MAE 0.36 against this set, passed every threshold in
`m3/calibrate.py`, and was later found to be scoring *deviation from the unsteered baseline* as
*presence of the concept* — the defect that forced the 2026-08-21 Gemma runs to be re-judged.
A validation that a judge passes while carrying its central defect is a validation that could
not fail, which is exactly what `cohen_kappa` returning `None` was written to avoid elsewhere.

**Do not quote a number scored against this set as evidence that a judge is correct.** Two
references in this repository can do that job: the null controls in `tools/judge_bakeoff.py`,
whose answer is fixed by construction, and labels the operator writes themselves —
`python -m tools.judge_bakeoff worksheet` produces blind worksheets for that.

## What the labels are

Labels are **coordinates plus verdicts** — never transcript text, which stays outside the
repository. `id` is `{channel}:{L<layer>@<dose> | null}:{trial | prompt_id}` and is stable across
re-exports of a bundle, so anyone can re-read the same responses and disagree.

| file | judge | n | ambiguous |
|---|---|---:|---:|
| `gold_identify.jsonl` | `identify` | 30 | 0 |
| `gold_self_report.jsonl` | `self_report` | 30 | 3 |
| `gold_coherence.jsonl` | `coherence` | 25 | 1 |
| `gold_effect.jsonl` | `effect` | 25 | 1 |

`ambiguous: true` marks items the labelling model reported itself unsure about. Agreement is
reported twice, with and without them, so a judge is neither failed nor passed on items where the
labeller could not decide.

## The sample is stratified, not random

Drawn evenly across five strata — null arm, degenerate, no concept, concept present, concept
heavy. A random draw from 1,204 responses is mostly clean identifications, and a judge can score
95% on those while being wrong about every confabulation and every repetition loop, which are the
two things it exists to tell apart.

## What the hard cases are for

- **`identify`** — six α=0 responses naming "apple" with confident sensory detail, and six pure
  `garlic garlic garlic…` loops that still correctly complete the prefill. Together they test that
  the judge scores *what was named* and is moved by neither fluency nor collapse.
- **`self_report`** — `detect:L44@0.30:t1` denies detection and then mentions garlic; that is the
  `leaked` class the study exists to find. `detect:L45@0.30:t25` claims detection without the
  stock phrase, which a keyword matcher misses.
- **`coherence`** — `task:L42@0.30:e5_02` invents a dwarf planet called Garlic while being
  perfectly well-formed (coherence is form, not truth), and `task:L43@0.30:e5_03` repeats one
  fluent sentence five times instead of answering (fluent ≠ coherent — the failure a judge is
  most likely to get wrong, and did in M2).
- **`effect`** — six items where A and B are the *same unsteered text*. Any influence above 0
  there means the judge invents influence from the framing alone. And `task:L46@0.30:e5_07` never
  says garlic but writes about a gardener beside an All-Spice Emporium: either faint unnamed
  influence or coincidence, which a mention count scores 0 and cannot distinguish.

## Running it

```bash
python -m m3.calibrate sample --probe /path/to/unzipped/bundle    # regenerate worksheets, free
python -m m3.calibrate score  --probe /path/to/unzipped/bundle    # ~110 calls, about $0.05
```

Iterate on the prompts in `m3/judge.py` between runs. Fix what the disagreement list shows, not
what the headline number shows.
