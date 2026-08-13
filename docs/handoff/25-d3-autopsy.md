# 25 — `d3` autopsy: what the model actually says at the frontier cells

**Status: DONE — `a355a61`, `f90180c`.** Branch `pareto`. Runs **before** task
[11](11-the-run.md), standalone, on four cells. Built to be **deleted in one commit** once it has
answered — see *Rollback* at the bottom.

## Why this runs before the pipeline and not inside it

Task [21](21-unified-cell-selection.md) ranks cells on `d3`, and `d3` has never been validated
against anything. The frontier it produced puts `L57 @ r = 0.30` and `L58 @ r = 0.30` at the covert
end on the strength of `d3` = 0.0018 and 0.027. **A single specific defect would invert that
ranking**, and the run costs ~1h20m before VERIFY produces the first real `d2` that would expose it.

The reading below comes from the committed shakedown fixture, so it needs no GPU to reproduce.

### What the fixture shows at `r = 0.30`

| layer | `reach` | `d3` mass | `d3_rank_med` | `s3` |
|---|---|---|---|---|
| L53–L56 | 0.17 | 0.0001–0.005 | **2** | 0.90–0.95 |
| **L57** | 0.33 | 0.0018 | **2** | 0.98 |
| **L58** | 0.42 | 0.027 | **2** | 0.98 |
| L59 | 1.00 | 0.99993 | **1** | 0.98 |

**At both candidate cells the concept is the second most likely token out of the whole vocabulary,
and we do not know what the first one is.** Nothing in the pipeline stores it.

Two corrections to the record that follow from the same reading:

- **`d3` is not "effectively binary".** That claim counted dead cells. Of 93 sane cells, 66 have
  `reach` = 0.00, so `d3` = 0 there is correct rather than pathological. Restricted to cells where
  something is happening (`s3` >= 0.70, `reach` >= 0.20, n = 19), `d3` is graded: 0.002, 0.027,
  0.452, 0.500, 0.571, 0.587, 0.979, 0.996, 0.999, 1.000...
- **`d3_rate` is not an independent check on `d3`.** It counts trials where the same mass exceeds
  `D3_RATE_THRESH` = 0.10. Both reading 0.00 is one axis reporting twice. `d3_rank_med` is the only
  genuinely different view, and it disagrees: rank 2 is not "undetectable".

## The three hypotheses it separates

| | Hypothesis | If true |
|---|---|---|
| **A** | **The rank-1 token is a concept surface form that `variant_verdict` dropped from `cids`.** The mass sum then excludes the word the model is actually emitting | `d3` ~ 0 while real `d2` ~ 1.0. **The frontier is inverted** and the covert end is the detectable end |
| **B** | The readout position is too early. `d3` reads one position after `FORCED_PREFILL` plus one filler; the model names the concept later in the sentence | `d3` under-reads wherever the model answers in fluent prose rather than flooding |
| **C** | Exact token vs semantic identification. `d3` counts token ids; a judge accepts *"a pungent bulb"*, *"cooking"*, *"vampires"* | `d3` under-reads at late layers, which is exactly where the candidates sit |

A explains every observation at once, including the L58 -> L59 step and the disagreement with a
judged protocol, and it is the only one that makes the frontier actively wrong rather than
conservative. It is also the cheapest to test.

**What this is not.** It is not a comparison against Macar's published curve. Within our own run at
`r = 0.30`, α wanders non-monotonically (L53 2.37, L55 1.94, L57 2.25, L58 2.39, L59 2.14, L61 1.72)
while `d3` steps 0.00 -> 1.00 between L58 and L59 **at lower α**. α does not control the outcome in
this band, so "we disagree at matched α" compares on an axis that is not doing the work. The
question this task settles is internal: does `d3` measure what `d2` measures.

## What it does

Four parts, in this order, per cell.

1. **The variant table.** Print `concept_first_token_ids`' kept **and dropped** candidates with the
   decoded string and the `variant_verdict` reason for each. Currently this is only surfaced when
   extraction fails. Task [13](13-prefix-token-contamination.md) wants it anyway.
2. **The top-k dump.** At the forced-ID position, print the **top 10 tokens with probabilities**,
   marking which are in `cids`. Then greedy-extend one token and print the top 10 again — the
   `ALLOW_FILLER` position. Reuse `cheap._d3_forward` so this reads the same tensor `d3` reads; do
   not rebuild the prompt.
3. **Real `d2` at the same cell**, via `expensive.measure_D2` at small `n`. This is gate 5's input,
   on four cells, hours earlier than VERIFY would deliver it.
4. **One line per cell** putting `d3` mass, `d3_rate`, `d3_rank_med` and real `d2` side by side.

### Cells

Default to the frontier plus a positive control:

| cell | role |
|---|---|
| `L57 @ 0.30` | frontier candidate, `d3` 0.0018 |
| `L58 @ 0.30` | frontier candidate, `d3` 0.027 |
| `L59 @ 0.30` | **positive control** — `d3` 0.99993, rank 1. The dump must show a concept token winning here, or the dump itself is wrong |
| `L52 @ 0.60` | frontier, max influence and max detection |

Overridable as `--autopsy-cells "57@0.30,58@0.30,59@0.30,52@0.60"`. The positive control is not
optional: without it, a dump that shows nothing is indistinguishable from a broken dump.

## How to read the result

| Top-1 at L57/L58 | Real `d2` | Reading |
|---|---|---|
| a concept surface form | high | **Hypothesis A. `d3` is broken, the frontier is inverted.** Stop; fix `cids` before any run |
| filler / refusal / unrelated | ~0 | The cells are genuinely covert. The frontier stands and the disagreement with the published curve is a finding, not a bug |
| filler / refusal / unrelated | high | Hypothesis B or C — the model names it later or describes it. `d3` is conservative, not inverted; gate 5's ρ decides whether it is usable |

## Design constraints

- **One new module, `m2/autopsy.py`, and one flag on `m2/run.py`.** Nothing in `cheap.py`,
  `expensive.py`, `phases.py` or `config.py` changes behaviour. Read-only use of the existing
  helpers. This is what makes the rollback a single revert.
- **Benign concepts only, structurally.** It prints generations. Gate it on
  `runio.transcripts_allowed` exactly as `export_bundle` does, with **no override parameter** — same
  reasoning as task [08](08-debug-bundle.md).
- **Never writes into the repository working tree.** Output goes to the run directory or stdout.
- **No judge spend beyond `measure_D2` at small `n`.** Four cells, not the shortlist.

## Cost

Vector build plus four cells. The dumps are two forward passes each; `measure_D2` at small `n` is
well under VERIFY's 50 s/cell prior. **Budget under ten minutes including model load** — against a
run that is ~1h20m and whose last four phases have never been observed.

## Also in this task: carry the rank into the frontier table

`d3_rank_med` already rides on every scan row and gate 1 uses it, but the shortlist/frontier table
does not show it. Add the column. A cell reported as covert while the concept sits at rank 2 needs
that visible at selection time and in the write-up — this is a claim we would otherwise publish
without stating.

## Acceptance

- Running with a harmful concept name **raises**, and no output is written.
- The positive control cell prints a concept token at rank 1 with probability > 0.9. If it does not,
  the dump is wrong and nothing else in the output means anything.
- Dropped variants are printed with their reason, not silently omitted.
- The four-cell summary table prints `d3`, `d3_rate`, `d3_rank_med` and `d2` in one row per cell.
- The frontier table shows `d3_rank_med`.

## Rollback

Recorded here so it is not reconstructed later. The disposable diagnostic reverts in one step by
reverting `f90180c`, which deletes `m2/autopsy.py`, its flag in `m2/run.py` and its tests. The
`d3_rank_med` frontier column in `a355a61` is the operator's explicit permanent exception and
stays after that revert. If the answer is "the cells are genuinely covert", the diagnostic has
done its job and carrying it into every future concept is dead weight. If the answer is A or B,
its findings become a defect fix in `cheap.py`, which is a different task.
