# 2026-08-14 — Garlic M2 run read

**Scope.** This is a scalar-only read of the archive, export bundle and console log from the
2026-08-14 Garlic run. No transcript text is reproduced. `d2` (forced-ID rate, 0–1, lower is
better), `d3` (the cheap forced-ID concept-mass proxy), `s3` (capability ratio against the
unsteered baseline), `s2_forced` (objective non-degeneracy of the forced-ID responses) and
`e6` reach (fraction of prompts clearing the reach threshold) are named once here and then used
by code below. In the SCAN tables, “sane” means the cheap eligibility condition
`s3 >= 0.70`; it is not the later full `s4` measurement.

## A. Inventory and run state

The archive and export bundle have the same inventory and identical SHA-256 content for every
file except `lab.log`. The export copy of `lab.log` has one additional, expected line recording
creation of the archive. Counts below are from either copy.

| JSONL | Rows |
|---|---:|
| `baselines.jsonl` | 1 |
| `cis_transcripts.jsonl` | 24 |
| `D2_transcripts.jsonl` | 215 |
| `judge_d2.jsonl` | 215 |
| `judge_e5.jsonl` | 7 |
| `judge_s1.jsonl` | 12 |
| `norms.jsonl` | 49 |
| `provenance.jsonl` | 1 |
| `scan.jsonl` | 189 |
| `selection_d2.jsonl` | 37 |
| `unsteered/e5_01.jsonl` | 3 |
| `unsteered/e5_02.jsonl` | 3 |
| `unsteered/e5_03.jsonl` | 3 |
| `unsteered/e5_04.jsonl` | 3 |
| `unsteered/e5_05.jsonl` | 3 |
| `unsteered/e5_06.jsonl` | 3 |
| `unsteered/e5_07.jsonl` | 3 |
| `unsteered/e5_08.jsonl` | 3 |
| `unsteered/e5_09.jsonl` | 3 |
| `unsteered/e5_10.jsonl` | 3 |
| `unsteered/e5_11.jsonl` | 3 |
| `unsteered/e5_12.jsonl` | 3 |

All JSONL rows that carry a hash agree on config hash `b252ac4af2fe`. The sole
`provenance.jsonl` row does **not** contain a Git commit field: it records software, GPU, host,
model, time, concept and config hash only. Therefore the run’s Git commit is **not recoverable
from the supplied provenance**, and the archive suffix is the config hash, not evidence of a
commit.

| Phase | Recorded state | Units | Elapsed | What that state means |
|---|---|---:|---:|---|
| RIG | done | — | 0.0 s | 2 pass, 0 fail, 5 skipped in the pre-run rig summary |
| CAL | done | 1/1 | 157.6 s | Completed |
| SCAN | done | 189/189 | 2059.8 s | 147 grid cells plus 42 task-24 knee cells |
| SHORTLIST | **failed: `OSError`** | **37/46** | 7m24s on the final board | No `shortlist.json` or tier plan was produced |
| BISECT | pending / not run | 0/11 | — | No tier plan |
| VERIFY | pending / not run | 0/11 | — | No tier plan |
| REFINE | done over an empty input | 0/? | 0.0 s | Mechanically completed, measured no cells |
| SELECT | done over an empty input | — | 0.0 s | No verified cells |
| FRONTIER | done over an empty input | — | 0.0 s | No verified cells |
| COVERTNESS | done over an empty input | — | 0.0 s | No verified cells |
| CONFIRM | skipped | 0/1 | — | No operating point |
| CONTROLS | done over an empty input | 1/1 on board | 0.0 s | Sections 9.1/9.2 skipped because there was no winner |
| GATES | done | — | 32.4 s | Gate 7 raised independently; other gate results are in the bundle |

This confirms the headline counts: SCAN has **189** cells, `selection_d2.jsonl` has **37**
persisted rows, and the status board ended with SHORTLIST failed at **37/46**. Section G explains
why the denominator grew and why “37 persisted” is not quite the same as “37 measured.”

## B. The 37 persisted `selection_d2` rows

Sorted by lower `d2` and then higher `e6` reach (with layer and `r` as deterministic
tie-breakers). “Eligibility” is the persisted `eligibility_first_tier`, because the phase failed
before it could attach the final selected-tier field.

| Layer | r | alpha | e6 reach | d2 | d3 | d3_rank_med | s3 | s2_forced | First eligibility tier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L52 | 0.600 | 5.347 | 1.000 | 1.000 | 0.9787 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L53 | 0.600 | 4.733 | 1.000 | 1.000 | 0.9985 | 1.0 | 1.000 | 0.000 | 0 (3/12) |
| L54 | 0.600 | 4.679 | 1.000 | 1.000 | 1.0000 | 1.0 | 0.905 | 0.000 | 0 (3/12) |
| L55 | 0.600 | 3.874 | 1.000 | 1.000 | 0.9961 | 1.0 | 0.905 | 0.000 | 0 (3/12) |
| L56 | 0.600 | 3.864 | 1.000 | 1.000 | 0.9999 | 1.0 | 0.929 | 0.000 | 0 (3/12) |
| L57 | 0.600 | 4.494 | 1.000 | 1.000 | 1.0000 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L58 | 0.375 | 2.984 | 1.000 | 1.000 | 0.9987 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L58 | 0.450 | 3.581 | 1.000 | 1.000 | 1.0000 | 1.0 | 0.929 | 0.000 | 0 (3/12) |
| L58 | 0.600 | 4.774 | 1.000 | 1.000 | 1.0000 | 1.0 | 0.929 | 0.000 | 0 (3/12) |
| L59 | 0.300 | 2.144 | 1.000 | 1.000 | 0.9999 | 1.0 | 0.976 | 0.000 | 0 (3/12) |
| L59 | 0.600 | 4.288 | 1.000 | 1.000 | 1.0000 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L60 | 0.300 | 1.893 | 1.000 | 1.000 | 0.9992 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L61 | 0.600 | 3.439 | 1.000 | 1.000 | 1.0000 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L56 | 0.525 | 3.381 | 0.917 | 1.000 | 0.8914 | 1.0 | 0.929 | 0.000 | 0 (3/12) |
| L61 | 0.300 | 1.719 | 0.917 | 1.000 | 1.0000 | 1.0 | 0.952 | 0.000 | 0 (3/12) |
| L53 | 0.450 | 3.550 | 0.667 | 1.000 | 0.4507 | 3.0 | 0.976 | 0.200 | 0 (3/12) |
| L55 | 0.450 | 2.906 | 0.667 | 1.000 | 0.8682 | 2.0 | 0.929 | 0.000 | 0 (3/12) |
| L56 | 0.450 | 2.898 | 0.667 | 1.000 | 0.0801 | 2.0 | 0.905 | 0.000 | 0 (3/12) |
| L57 | 0.375 | 2.809 | 0.667 | 1.000 | 0.1993 | 2.0 | 0.952 | 0.000 | 0 (3/12) |
| L51 | 0.600 | 5.099 | 0.500 | 1.000 | 0.5867 | 1.0 | 0.929 | 0.400 | 0 (3/12) |
| L52 | 0.450 | 4.010 | 0.500 | 1.000 | 0.2699 | 7.0 | 1.000 | 0.800 | 0 (3/12) |
| L55 | 0.375 | 2.421 | 0.500 | 1.000 | 0.1630 | 2.0 | 0.929 | 0.200 | 0 (3/12) |
| L50 | 0.600 | 4.996 | 0.417 | 1.000 | 0.5002 | 1.0 | 0.881 | 0.200 | 0 (3/12) |
| L54 | 0.375 | 2.924 | 0.417 | 1.000 | 0.0394 | 2.0 | 0.952 | 0.400 | 0 (3/12) |
| L57 | 0.300 | 2.247 | 0.333 | 1.000 | 0.0018 | 2.0 | 0.976 | 0.000 | 0 (3/12) |
| L47 | 0.600 | 5.385 | 0.250 | 1.000 | 0.4524 | 1.0 | 0.738 | 0.200 | 0 (3/12) |
| L49 | 0.600 | 4.769 | 0.250 | 1.000 | 0.5708 | 1.0 | 0.952 | 0.400 | 0 (3/12) |
| L50 | 0.525 | 4.372 | 0.250 | 1.000 | 0.8060 | 3.0 | 0.929 | 1.000 | 0 (3/12) |
| L51 | 0.525 | 4.462 | 0.250 | 1.000 | 0.9997 | 3.0 | 0.976 | 1.000 | 0 (3/12) |
| L53 | 0.375 | 2.958 | 0.250 | 1.000 | 0.0357 | 5.0 | 0.952 | 0.800 | 0 (3/12) |
| L49 | 0.450 | 3.577 | 0.167 | 1.000 | 0.0678 | 6.0 | 1.000 | 1.000 | 1 (2/12) |
| L51 | 0.450 | 3.824 | 0.167 | 1.000 | 0.0251 | 6.0 | 0.952 | 1.000 | 1 (2/12) |
| L52 | 0.375 | 3.342 | 0.167 | 1.000 | 0.0163 | 8.0 | 0.976 | 1.000 | 1 (2/12) |
| L53 | 0.300 | 2.367 | 0.167 | 1.000 | 0.0012 | 5.0 | 0.905 | 1.000 | 1 (2/12) |
| L54 | 0.300 | 2.339 | 0.167 | 1.000 | 0.0012 | 4.0 | 0.952 | 1.000 | 1 (2/12) |
| L55 | 0.300 | 1.937 | 0.167 | 1.000 | 0.0049 | 2.0 | 0.929 | 1.000 | 1 (2/12) |
| L56 | 0.300 | 1.932 | 0.167 | 1.000 | 0.0001 | 2.0 | 0.929 | 0.800 | 1 (2/12) |

**Confirmed:** every one of the 37 persisted rows has `d2 = 1.000` at `n = 5`. There is no
`d2` variance in this table.

## C. Per-layer SCAN dose ladder

There are 49 layers and 189 unique cells: 147 base-grid rows plus 42 rows whose
`scan_provenance` is `knee_search`. Knee directions and depths below are copied from those task-24
provenance fields. Displayed doses are rounded to three decimals; the stored rows retain the raw
`0.44999999999999996` and `0.5249999999999999` values described in TODO 24. “Unreachable” means
the configured dose exceeded `ALPHA_CEIL` and no `e6`/`d3`/`s3` measurement exists.

| Layer | r | e6 reach | d3 | s3 | Sane (`s3 >= 0.70`) | Provenance |
|---:|---:|---:|---:|---:|:---:|---|
| L13 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L13 | 0.300 | 0.000 | 0.0000 | 0.952 | yes | grid |
| L13 | 0.600 | 0.000 | 0.0000 | 0.762 | yes | grid |
| L14 | 0.150 | 0.000 | 0.0000 | 0.929 | yes | grid |
| L14 | 0.300 | 0.000 | 0.0000 | 0.762 | yes | grid |
| L14 | 0.600 | — | — | — | unreachable | grid |
| L15 | 0.150 | 0.000 | 0.0000 | 0.810 | yes | grid |
| L15 | 0.300 | 0.000 | 0.0000 | 0.595 | no | grid |
| L15 | 0.600 | — | — | — | unreachable | grid |
| L16 | 0.150 | 0.000 | 0.0000 | 0.810 | yes | grid |
| L16 | 0.300 | 0.000 | 0.0000 | 0.667 | no | grid |
| L16 | 0.600 | — | — | — | unreachable | grid |
| L17 | 0.150 | 0.000 | 0.0000 | 0.667 | no | grid |
| L17 | 0.300 | 0.000 | 0.0000 | 0.405 | no | grid |
| L17 | 0.600 | — | — | — | unreachable | grid |
| L18 | 0.150 | 0.000 | 0.0000 | 0.810 | yes | grid |
| L18 | 0.300 | 0.000 | 0.0000 | 0.333 | no | grid |
| L18 | 0.600 | — | — | — | unreachable | grid |
| L19 | 0.150 | 0.000 | 0.0000 | 0.881 | yes | grid |
| L19 | 0.300 | 0.000 | 0.0000 | 0.476 | no | grid |
| L19 | 0.600 | — | — | — | unreachable | grid |
| L20 | 0.150 | 0.000 | 0.0000 | 0.786 | yes | grid |
| L20 | 0.300 | 0.000 | 0.0000 | 0.476 | no | grid |
| L20 | 0.600 | — | — | — | unreachable | grid |
| L21 | 0.150 | 0.000 | 0.0000 | 0.714 | yes | grid |
| L21 | 0.300 | 0.000 | 0.0000 | 0.405 | no | grid |
| L21 | 0.600 | — | — | — | unreachable | grid |
| L22 | 0.150 | 0.000 | 0.0000 | 0.643 | no | grid |
| L22 | 0.300 | 0.000 | 0.0000 | 0.381 | no | grid |
| L22 | 0.600 | — | — | — | unreachable | grid |
| L23 | 0.150 | 0.000 | 0.0000 | 0.810 | yes | grid |
| L23 | 0.300 | 0.000 | 0.0000 | 0.548 | no | grid |
| L23 | 0.600 | — | — | — | unreachable | grid |
| L24 | 0.150 | 0.000 | 0.0000 | 0.810 | yes | grid |
| L24 | 0.300 | 0.000 | 0.0000 | 0.476 | no | grid |
| L24 | 0.600 | — | — | — | unreachable | grid |
| L25 | 0.150 | 0.000 | 0.0000 | 0.786 | yes | grid |
| L25 | 0.300 | 0.000 | 0.0000 | 0.405 | no | grid |
| L25 | 0.600 | — | — | — | unreachable | grid |
| L26 | 0.150 | 0.000 | 0.0000 | 0.857 | yes | grid |
| L26 | 0.300 | 0.000 | 0.0000 | 0.429 | no | grid |
| L26 | 0.600 | — | — | — | unreachable | grid |
| L27 | 0.150 | 0.000 | 0.0000 | 0.738 | yes | grid |
| L27 | 0.300 | 0.000 | 0.0000 | 0.310 | no | grid |
| L27 | 0.600 | — | — | — | unreachable | grid |
| L28 | 0.150 | 0.000 | 0.0000 | 0.667 | no | grid |
| L28 | 0.300 | 0.000 | 0.0000 | 0.381 | no | grid |
| L28 | 0.600 | — | — | — | unreachable | grid |
| L29 | 0.150 | 0.000 | 0.0000 | 0.714 | yes | grid |
| L29 | 0.300 | 0.000 | 0.0000 | 0.476 | no | grid |
| L29 | 0.600 | — | — | — | unreachable | grid |
| L30 | 0.150 | 0.000 | 0.2008 | 0.905 | yes | grid |
| L30 | 0.300 | 0.000 | 0.0000 | 0.405 | no | grid |
| L30 | 0.600 | — | — | — | unreachable | grid |
| L31 | 0.150 | 0.000 | 0.0000 | 0.952 | yes | grid |
| L31 | 0.300 | 0.000 | 0.0060 | 0.429 | no | grid |
| L31 | 0.600 | 0.000 | 0.0004 | 0.429 | no | grid |
| L32 | 0.150 | 0.000 | 0.0000 | 0.952 | yes | grid |
| L32 | 0.300 | 0.000 | 0.6979 | 0.357 | no | grid |
| L32 | 0.600 | 0.000 | 0.0000 | 0.357 | no | grid |
| L33 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L33 | 0.300 | 0.000 | 0.0000 | 0.286 | no | grid |
| L33 | 0.600 | 0.000 | 0.0000 | 0.310 | no | grid |
| L34 | 0.150 | 0.000 | 0.0000 | 0.905 | yes | grid |
| L34 | 0.300 | 0.000 | 0.0000 | 0.524 | no | grid |
| L34 | 0.600 | 0.000 | 0.0000 | 0.357 | no | grid |
| L35 | 0.150 | 0.000 | 0.0000 | 0.952 | yes | grid |
| L35 | 0.300 | 0.000 | 0.0000 | 0.881 | yes | grid |
| L35 | 0.600 | 0.000 | 0.0000 | 0.476 | no | grid |
| L36 | 0.150 | 0.000 | 0.0000 | 0.881 | yes | grid |
| L36 | 0.300 | 0.000 | 0.0000 | 0.833 | yes | grid |
| L36 | 0.600 | 0.000 | 0.0012 | 0.524 | no | grid |
| L37 | 0.150 | 0.000 | 0.0000 | 0.929 | yes | grid |
| L37 | 0.300 | 0.000 | 0.9998 | 0.905 | yes | grid |
| L37 | 0.375 | 0.000 | 0.9831 | 0.786 | yes | task 24 knee d2 up |
| L37 | 0.450 | 0.000 | 0.7993 | 0.643 | no | task 24 knee d1 down |
| L37 | 0.600 | 0.000 | 0.0103 | 0.476 | no | grid |
| L38 | 0.150 | 0.000 | 0.0000 | 0.929 | yes | grid |
| L38 | 0.300 | 0.000 | 1.0000 | 0.881 | yes | grid |
| L38 | 0.450 | 0.000 | 0.9945 | 0.786 | yes | task 24 knee d1 up |
| L38 | 0.525 | 0.000 | 0.4504 | 0.643 | no | task 24 knee d2 down |
| L38 | 0.600 | 0.000 | 0.0000 | 0.667 | no | grid |
| L39 | 0.150 | 0.000 | 0.0000 | 0.929 | yes | grid |
| L39 | 0.300 | 0.000 | 0.9996 | 0.810 | yes | grid |
| L39 | 0.375 | 0.000 | 0.9998 | 0.667 | no | task 24 knee d2 down |
| L39 | 0.450 | 0.000 | 0.9994 | 0.595 | no | task 24 knee d1 down |
| L39 | 0.600 | 0.000 | 0.1069 | 0.500 | no | grid |
| L40 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L40 | 0.300 | 0.000 | 0.9998 | 0.738 | yes | grid |
| L40 | 0.375 | 0.000 | 0.4003 | 0.690 | no | task 24 knee d2 down |
| L40 | 0.450 | 0.000 | 0.0057 | 0.595 | no | task 24 knee d1 down |
| L40 | 0.600 | 0.000 | 0.5680 | 0.548 | no | grid |
| L41 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L41 | 0.300 | 0.000 | 0.3560 | 0.833 | yes | grid |
| L41 | 0.375 | 0.000 | 0.9331 | 0.714 | yes | task 24 knee d2 up |
| L41 | 0.450 | 0.000 | 0.5772 | 0.667 | no | task 24 knee d1 down |
| L41 | 0.600 | 0.000 | 0.0019 | 0.524 | no | grid |
| L42 | 0.150 | 0.000 | 0.0000 | 0.952 | yes | grid |
| L42 | 0.300 | 0.000 | 0.0001 | 1.000 | yes | grid |
| L42 | 0.450 | 0.000 | 0.6719 | 0.857 | yes | task 24 knee d1 up |
| L42 | 0.525 | 0.000 | 0.4271 | 0.595 | no | task 24 knee d2 down |
| L42 | 0.600 | 0.000 | 0.9057 | 0.524 | no | grid |
| L43 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L43 | 0.300 | 0.000 | 1.0000 | 0.881 | yes | grid |
| L43 | 0.375 | 0.000 | 1.0000 | 0.833 | yes | task 24 knee d2 up |
| L43 | 0.450 | 0.000 | 0.9998 | 0.690 | no | task 24 knee d1 down |
| L43 | 0.600 | 0.167 | 0.5642 | 0.619 | no | grid |
| L44 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L44 | 0.300 | 0.000 | 0.8000 | 0.929 | yes | grid |
| L44 | 0.600 | 0.000 | 0.6871 | 0.667 | no | grid |
| L45 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L45 | 0.300 | 0.000 | 0.0001 | 0.976 | yes | grid |
| L45 | 0.450 | 0.000 | 0.7823 | 0.857 | yes | task 24 knee d1 up |
| L45 | 0.525 | 0.000 | 0.8617 | 0.833 | yes | task 24 knee d2 up |
| L45 | 0.600 | 0.000 | 0.9065 | 0.810 | yes | grid |
| L46 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L46 | 0.300 | 0.000 | 0.0001 | 0.976 | yes | grid |
| L46 | 0.450 | 0.000 | 0.5632 | 1.000 | yes | task 24 knee d1 up |
| L46 | 0.525 | 0.000 | 0.7600 | 0.929 | yes | task 24 knee d2 up |
| L46 | 0.600 | 0.000 | 0.5877 | 0.857 | yes | grid |
| L47 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L47 | 0.300 | 0.000 | 0.0004 | 0.976 | yes | grid |
| L47 | 0.450 | 0.083 | 0.7466 | 1.000 | yes | task 24 knee d1 up |
| L47 | 0.525 | 0.083 | 0.9616 | 0.786 | yes | task 24 knee d2 up |
| L47 | 0.600 | 0.250 | 0.4524 | 0.738 | yes | grid |
| L48 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L48 | 0.300 | 0.000 | 0.0001 | 0.976 | yes | grid |
| L48 | 0.450 | 0.083 | 0.2836 | 0.881 | yes | task 24 knee d1 up |
| L48 | 0.525 | 0.000 | 0.8309 | 0.762 | yes | task 24 knee d2 up |
| L48 | 0.600 | 0.083 | 0.9534 | 0.714 | yes | grid |
| L49 | 0.150 | 0.000 | 0.0000 | 1.024 | yes | grid |
| L49 | 0.300 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L49 | 0.450 | 0.167 | 0.0678 | 1.000 | yes | task 24 knee d1 up |
| L49 | 0.525 | 0.083 | 0.7624 | 0.976 | yes | task 24 knee d2 up |
| L49 | 0.600 | 0.250 | 0.5708 | 0.952 | yes | grid |
| L50 | 0.150 | 0.000 | 0.0000 | 1.024 | yes | grid |
| L50 | 0.300 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L50 | 0.450 | 0.083 | 0.0494 | 0.976 | yes | task 24 knee d1 up |
| L50 | 0.525 | 0.250 | 0.8060 | 0.929 | yes | task 24 knee d2 down |
| L50 | 0.600 | 0.417 | 0.5002 | 0.881 | yes | grid |
| L51 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L51 | 0.300 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L51 | 0.450 | 0.167 | 0.0251 | 0.952 | yes | task 24 knee d1 up |
| L51 | 0.525 | 0.250 | 0.9997 | 0.976 | yes | task 24 knee d2 down |
| L51 | 0.600 | 0.500 | 0.5867 | 0.929 | yes | grid |
| L52 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L52 | 0.300 | 0.000 | 0.0003 | 1.000 | yes | grid |
| L52 | 0.375 | 0.167 | 0.0163 | 0.976 | yes | task 24 knee d2 up |
| L52 | 0.450 | 0.500 | 0.2699 | 1.000 | yes | task 24 knee d1 down |
| L52 | 0.600 | 1.000 | 0.9787 | 0.952 | yes | grid |
| L53 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L53 | 0.300 | 0.167 | 0.0012 | 0.905 | yes | grid |
| L53 | 0.375 | 0.250 | 0.0357 | 0.952 | yes | task 24 knee d2 up |
| L53 | 0.450 | 0.667 | 0.4507 | 0.976 | yes | task 24 knee d1 down |
| L53 | 0.600 | 1.000 | 0.9985 | 1.000 | yes | grid |
| L54 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L54 | 0.300 | 0.167 | 0.0012 | 0.952 | yes | grid |
| L54 | 0.375 | 0.417 | 0.0394 | 0.952 | yes | task 24 knee d2 up |
| L54 | 0.450 | 0.750 | 0.5980 | 0.952 | yes | task 24 knee d1 down |
| L54 | 0.600 | 1.000 | 1.0000 | 0.905 | yes | grid |
| L55 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L55 | 0.300 | 0.167 | 0.0049 | 0.929 | yes | grid |
| L55 | 0.375 | 0.500 | 0.1630 | 0.929 | yes | task 24 knee d2 down |
| L55 | 0.450 | 0.667 | 0.8682 | 0.929 | yes | task 24 knee d1 down |
| L55 | 0.600 | 1.000 | 0.9961 | 0.905 | yes | grid |
| L56 | 0.150 | 0.000 | 0.0000 | 0.976 | yes | grid |
| L56 | 0.300 | 0.167 | 0.0001 | 0.929 | yes | grid |
| L56 | 0.450 | 0.667 | 0.0801 | 0.905 | yes | task 24 knee d1 up |
| L56 | 0.525 | 0.917 | 0.8914 | 0.929 | yes | task 24 knee d2 down |
| L56 | 0.600 | 1.000 | 0.9999 | 0.929 | yes | grid |
| L57 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L57 | 0.300 | 0.333 | 0.0018 | 0.976 | yes | grid |
| L57 | 0.375 | 0.667 | 0.1993 | 0.952 | yes | task 24 knee d2 down |
| L57 | 0.450 | 1.000 | 0.9882 | 0.929 | yes | task 24 knee d1 down |
| L57 | 0.600 | 1.000 | 1.0000 | 0.952 | yes | grid |
| L58 | 0.150 | 0.000 | 0.0000 | 1.000 | yes | grid |
| L58 | 0.300 | 0.417 | 0.0265 | 0.976 | yes | grid |
| L58 | 0.375 | 1.000 | 0.9987 | 0.952 | yes | task 24 knee d2 down |
| L58 | 0.450 | 1.000 | 1.0000 | 0.929 | yes | task 24 knee d1 down |
| L58 | 0.600 | 1.000 | 1.0000 | 0.929 | yes | grid |
| L59 | 0.150 | 0.083 | 0.0003 | 0.976 | yes | grid |
| L59 | 0.300 | 1.000 | 0.9999 | 0.976 | yes | grid |
| L59 | 0.600 | 1.000 | 1.0000 | 0.952 | yes | grid |
| L60 | 0.150 | 0.083 | 0.0000 | 0.976 | yes | grid |
| L60 | 0.300 | 1.000 | 0.9992 | 0.952 | yes | grid |
| L60 | 0.600 | 1.000 | 1.0000 | 0.952 | yes | grid |
| L61 | 0.150 | 0.083 | 0.0002 | 0.976 | yes | grid |
| L61 | 0.300 | 0.917 | 1.0000 | 0.952 | yes | grid |
| L61 | 0.600 | 1.000 | 1.0000 | 0.952 | yes | grid |

## D. Lowest observed positive-`e6` cell per layer

“Onset” here means the lowest **sampled** `r` with `e6` reach above zero. Intuitively, it is
the first rung of this ladder where the signal appears, not an estimate of the exact transition:
for example, an onset cell at 0.300 places the transition in the sampled interval
`(0.150, 0.300]`. “Not observed” means every reachable sampled cell at that layer had zero reach.

| Layer | Lowest sampled r with e6 reach > 0 | e6 reach | s3 | Sane | Provenance |
|---:|---:|---:|---:|:---:|---|
| L13 | not observed | — | — | — | — |
| L14 | not observed | — | — | — | — |
| L15 | not observed | — | — | — | — |
| L16 | not observed | — | — | — | — |
| L17 | not observed | — | — | — | — |
| L18 | not observed | — | — | — | — |
| L19 | not observed | — | — | — | — |
| L20 | not observed | — | — | — | — |
| L21 | not observed | — | — | — | — |
| L22 | not observed | — | — | — | — |
| L23 | not observed | — | — | — | — |
| L24 | not observed | — | — | — | — |
| L25 | not observed | — | — | — | — |
| L26 | not observed | — | — | — | — |
| L27 | not observed | — | — | — | — |
| L28 | not observed | — | — | — | — |
| L29 | not observed | — | — | — | — |
| L30 | not observed | — | — | — | — |
| L31 | not observed | — | — | — | — |
| L32 | not observed | — | — | — | — |
| L33 | not observed | — | — | — | — |
| L34 | not observed | — | — | — | — |
| L35 | not observed | — | — | — | — |
| L36 | not observed | — | — | — | — |
| L37 | not observed | — | — | — | — |
| L38 | not observed | — | — | — | — |
| L39 | not observed | — | — | — | — |
| L40 | not observed | — | — | — | — |
| L41 | not observed | — | — | — | — |
| L42 | not observed | — | — | — | — |
| L43 | 0.600 | 0.167 | 0.619 | no | grid |
| L44 | not observed | — | — | — | — |
| L45 | not observed | — | — | — | — |
| L46 | not observed | — | — | — | — |
| L47 | 0.450 | 0.083 | 1.000 | yes | task 24 knee d1 up |
| L48 | 0.450 | 0.083 | 0.881 | yes | task 24 knee d1 up |
| L49 | 0.450 | 0.167 | 1.000 | yes | task 24 knee d1 up |
| L50 | 0.450 | 0.083 | 0.976 | yes | task 24 knee d1 up |
| L51 | 0.450 | 0.167 | 0.952 | yes | task 24 knee d1 up |
| L52 | 0.375 | 0.167 | 0.976 | yes | task 24 knee d2 up |
| L53 | 0.300 | 0.167 | 0.905 | yes | grid |
| L54 | 0.300 | 0.167 | 0.952 | yes | grid |
| L55 | 0.300 | 0.167 | 0.929 | yes | grid |
| L56 | 0.300 | 0.167 | 0.929 | yes | grid |
| L57 | 0.300 | 0.333 | 0.976 | yes | grid |
| L58 | 0.300 | 0.417 | 0.976 | yes | grid |
| L59 | 0.150 | 0.083 | 0.976 | yes | grid |
| L60 | 0.150 | 0.083 | 0.976 | yes | grid |
| L61 | 0.150 | 0.083 | 0.976 | yes | grid |

The current run therefore does **not** support the older statement that `e6` reach is zero at
`r = 0.150` for every layer: L59–L61 each read 1/12 = 0.083 at that dose. For L53–L58 the
unmeasured interior of 0.150–0.300 still brackets the first observed positive cell; for L47–L52
the first observed positive cell is later, at 0.375 or 0.450.

## E. `e6`-reach monotonicity

Strict monotonicity fails at two layers when all reachable sampled doses are ordered by `r`:

| Layer | Decrease | Provenance of decreasing step |
|---:|---|---|
| L48 | `r 0.450: 0.083 -> r 0.525: 0.000` | task-24 knee d1 to knee d2 |
| L49 | `r 0.450: 0.167 -> r 0.525: 0.083` | task-24 knee d1 to knee d2 |

Every other layer is non-decreasing over its reachable sampled cells. This means the onset table
is valid as a descriptive “first positive sampled cell,” but it is **not** a globally valid
monotone-threshold model: after onset, reach can fall on the next rung.

## F. Lowest observed `s3` sanity failure per layer

The boundary is the lowest reachable sampled `r` with `s3 < 0.70`. “Not observed” does not mean
the layer has no boundary; it means no sampled reachable dose crossed it.

| Layer | Lowest sampled r with s3 < 0.70 | s3 | Provenance |
|---:|---:|---:|---|
| L13 | not observed | — | — |
| L14 | not observed | — | — |
| L15 | 0.300 | 0.595 | grid |
| L16 | 0.300 | 0.667 | grid |
| L17 | 0.150 | 0.667 | grid |
| L18 | 0.300 | 0.333 | grid |
| L19 | 0.300 | 0.476 | grid |
| L20 | 0.300 | 0.476 | grid |
| L21 | 0.300 | 0.405 | grid |
| L22 | 0.150 | 0.643 | grid |
| L23 | 0.300 | 0.548 | grid |
| L24 | 0.300 | 0.476 | grid |
| L25 | 0.300 | 0.405 | grid |
| L26 | 0.300 | 0.429 | grid |
| L27 | 0.300 | 0.310 | grid |
| L28 | 0.150 | 0.667 | grid |
| L29 | 0.300 | 0.476 | grid |
| L30 | 0.300 | 0.405 | grid |
| L31 | 0.300 | 0.429 | grid |
| L32 | 0.300 | 0.357 | grid |
| L33 | 0.300 | 0.286 | grid |
| L34 | 0.300 | 0.524 | grid |
| L35 | 0.600 | 0.476 | grid |
| L36 | 0.600 | 0.524 | grid |
| L37 | 0.450 | 0.643 | task 24 knee d1 down |
| L38 | 0.525 | 0.643 | task 24 knee d2 down |
| L39 | 0.375 | 0.667 | task 24 knee d2 down |
| L40 | 0.375 | 0.690 | task 24 knee d2 down |
| L41 | 0.450 | 0.667 | task 24 knee d1 down |
| L42 | 0.525 | 0.595 | task 24 knee d2 down |
| L43 | 0.450 | 0.690 | task 24 knee d1 down |
| L44 | 0.600 | 0.667 | grid |
| L45 | not observed | — | — |
| L46 | not observed | — | — |
| L47 | not observed | — | — |
| L48 | not observed | — | — |
| L49 | not observed | — | — |
| L50 | not observed | — | — |
| L51 | not observed | — | — |
| L52 | not observed | — | — |
| L53 | not observed | — | — |
| L54 | not observed | — | — |
| L55 | not observed | — | — |
| L56 | not observed | — | — |
| L57 | not observed | — | — |
| L58 | not observed | — | — |
| L59 | not observed | — | — |
| L60 | not observed | — | — |
| L61 | not observed | — | — |

## G. What the SHORTLIST unit count and 30-cell cap count

`D2_SELECT_MAX = 30` is applied **inside each eligibility pass** to that pass’s still-new cells.
It is not a global phase cap. In `phase2_shortlist`, `planned += len(chosen)` accumulates the
chosen cells and sends that cumulative number to the board. The phase therefore replans from
30 to 37 to 46 as it relaxes eligibility.

The board numerator is narrower than “completed `d2` calls.” For each cell, the code measures
`d2`, derives `s2_forced`, then appends `selection_d2.jsonl`, and only **after that successful
append** calls the board tick. In plain language: the denominator counts selected cell jobs
across passes; the numerator counts jobs whose scalar selection row made it to disk.

| Eligibility pass | Floor | Cumulative eligible cells | New cells presented to this pass | Chosen under the per-pass cap | Persisted selection rows | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3/12 | 34 | 34 | 30 | 30 | Completed; no selectable cell |
| 1 | 2/12 | 41 | 7 | 7 | 7 | Completed; no selectable cell |
| 2 | 1/12, report-only | 50 | 9 | 9 | 0 | Entered; first cell completed `d2`, then its selection-row append failed |

Thus `30 + 7 + 9 = 46` planned units and `30 + 7 = 37` persisted/ticked units. These are
**eligibility passes**, not the later shortlist verification tiers: SHORTLIST never produced a
tier plan, so no later BISECT/VERIFY tier ran.

Four tier-0-eligible cells were intentionally not measured because the first pass had 34 cells
and the cap retained an evenly reach-spaced subset of 30:

- L58@0.300 (`e6` reach 0.417)
- L54@0.450 (`e6` reach 0.750)
- L57@0.450 (`e6` reach 1.000)
- L60@0.600 (`e6` reach 1.000)

The nine report-only tier-2 cells planned after the second relaxation were:

- L47@0.450, L47@0.525, L48@0.450, L48@0.600, L49@0.525
- L50@0.450, L59@0.150, L60@0.150, L61@0.150

L47@0.450 completed five generation/judge trials, all five were identified, so its recoverable
`d2` is 1.000. The `selection_d2` append for that cell is the write that raised. The other eight
tier-2 cells were not measured. Across the 50 cells eligible by the final report-only floor,
**12 had no real `d2` measurement**: the four tier-0 cap omissions plus those eight unattempted
tier-2 cells. L47@0.450 is the separate thirteenth persistence gap: `d2` exists, but its scalar
selection row does not.

**Task-26 omission recording did not survive this failure.** The four cap omissions would normally
be written by name in `shortlist.json` under `d2_measurement_omissions`, but the payload is written
only after all eligibility passes finish, and no `shortlist.json` exists. The eight unattempted
tier-2 cells likewise have no named incomplete-work manifest. Their identities can be
reconstructed deterministically from `scan.jsonl` and the code, as above, but they are not
recorded *as omissions* by the run. L47@0.450 is named in the transcript/judge rows, but has no
persisted scalar selection row.

## H. What `d3` tracks on this run

Method: use all SCAN cells that are reachable, have `e6` reach above zero, have non-null `d3`,
and are cheap-sane (`s3 >= 0.70`). This leaves `n = 50`. Spearman ranks use average ranks for
ties.

| Comparison | Spearman rho | n |
|---|---:|---:|
| `d3` against `e6` reach | **0.733** | 50 |
| `d3` against `s3` | **-0.190** | 50 |

`d3` was not correlated with `d2`: all 37 persisted `d2` values are 1.000, so the `d2` rank
variance is zero and Spearman rho is undefined. This confirms TODO 23.

**Interpretation.** Within the deliberately sane, reached subset, `d3` co-moves strongly with
`e6` reach and only weakly and negatively with the remaining `s3` variation. This particular
test therefore does not support the stronger claim that `d3` is primarily tracking degeneration
*within the sane region*. It also does not rehabilitate `d3` as a detection proxy: real `d2` is
constant, so there is no detection variation for `d3` to validate against. Because the filter
removes the broken side of each sanity boundary, this correlation does not retest the separate
task-25 observation that `d3` changes sharply when a cell crosses into damage.

## I. Failures

The supplied crash file, verbatim:

~~~text
concept Garlic | phase SHORTLIST | config b252ac4af2fe

Traceback (most recent call last):
  File "/workspace/steering-optimization/m2/driver.py", line 524, in phase
    value = fn()
            ^^^^
  File "/workspace/steering-optimization/m2/driver.py", line 855, in <lambda>
    "SHORTLIST", lambda: _phases().phase2_shortlist(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/steering-optimization/m2/phases.py", line 1302, in phase2_shortlist
    _append_row(SELECTION_D2_FILE, selection_row)
  File "/workspace/steering-optimization/m2/phases.py", line 220, in _append_row
    with open(path, "a", encoding="utf-8") as fh:
OSError: [Errno 5] Input/output error
~~~

The console records the SHORTLIST failure at 14:29:59 UTC. The final successfully written
`selection_d2` cell tick was 37. The five `D2_transcripts.jsonl` and `judge_d2.jsonl` rows for
the next cell, L47@0.450, are stamped 14:20:37 UTC, showing that its expensive measurement
finished before the subsequent scalar append remained blocked and finally raised.

The scientific consequence is limited but important: the persisted `d2` rows remain readable,
while `operating_point.json`’s “no operating point” is computed from **zero verified cells** after
the infrastructure failure. It is not evidence that a completed search found no operating point.

Gate 7’s `TypeError` is separate. In `gate7_d2_transcript_capture`, the data half builds:

~~~python
cells = sorted({(r["layer"], r["r"]) for r in rows if "layer" in r and "r" in r})
~~~

`D2_transcripts.jsonl` contains 25 `CAL_NULL_D2` rows whose `layer` and `r` values are both
`None`, plus 190 SHORTLIST rows with integer layers and float doses. The keys are present in both,
so the filter admits `(None, None)`. Sorting that tuple beside, for example,
`(47, 0.45)` compares the first elements and attempts `None < 47`, producing:

~~~text
TypeError: '<' not supported between instances of 'NoneType' and 'int'
~~~

Gate 7a passes because the writer filename is wired correctly. Gate 7b raises before it can report
its data measurement. This is **independent of SHORTLIST dying early**: any run file containing
both the null-control rows and at least one ordinary `d2` cell will take the same mixed-type sort
path, even if every phase completes.

## J. Direct confirmations and corrections

- **Confirmed:** `scan.jsonl` has 189 unique cells: 147 grid cells and 42 task-24 knee-search
  cells.
- **Confirmed:** `selection_d2.jsonl` has 37 rows and every one reads `d2 = 1.000`.
- **Confirmed with a precision correction:** the final board is 37/46, but 37 is the number of
  persisted/ticked selection rows. A 38th cell, L47@0.450, completed five `d2` trials and also
  reads `d2 = 1.000`; its summary append is what failed.
- **Corrected:** `D2_SELECT_MAX = 30` is a cap per eligibility pass over new cells, not a global
  cap. Passes 0, 1 and 2 contributed plans of 30, 7 and 9 cells.
- **Corrected:** not every layer has `e6` reach 0 at `r = 0.150` in this run. L59, L60 and L61
  each have reach 0.083 there.
- **Corrected:** `e6` reach is not monotone in dose at L48 or L49. Therefore an onset cell is a
  lowest observed positive rung, not proof of a one-way threshold.
- **Confirmed within scope:** every cell on which real `d2` completed is at `r >= 0.300` and reads
  `d2 = 1.000`. This statement applies to the `d2`-measured eligible cells, not to every SCAN cell,
  because most SCAN cells never received real `d2`.
- **Missing provenance:** no Git commit is present in `provenance.jsonl` or elsewhere in the
  supplied bundle/log. It cannot be supplied without guessing.
- **Bundle detail:** archive and export data are identical; only the export `lab.log` adds the
  expected archive-created line.
- **Failure semantics:** the null operating-point record is an infrastructure-cascade result over
  zero verified cells, not a completed scientific null.

## K. `d2` conditioned on forced-response sanity

Task 26's selection-time sanity floor is `S4_MIN = 0.70`. It applies to
`selection_sanity = min(s2_forced, s3)`. For the conditioning requested here, a cell is
`s2_forced`-sane when `s2_forced >= 0.70`; because every persisted cell has
`s2_forced_n = 5`, that means at least 4/5 forced-ID responses were non-degenerate.

There are **37, not 38, persisted rows** in `selection_d2.jsonl`, all with unique `(layer, r)`
keys. The 38th measured cell was L47@0.450, but its `selection_d2` scalar-row append is the write
that failed. It therefore has no persisted `s2_forced` value and is not reconstructed or included
in the conditioning below.

| Layer | r | e6 reach | d2 | s2_forced | s3 |
|---:|---:|---:|---:|---:|---:|
| L47 | 0.600 | 0.250 | 1.000 | 0.200 | 0.738 |
| L49 | 0.450 | 0.167 | 1.000 | 1.000 | 1.000 |
| L49 | 0.600 | 0.250 | 1.000 | 0.400 | 0.952 |
| L50 | 0.525 | 0.250 | 1.000 | 1.000 | 0.929 |
| L50 | 0.600 | 0.417 | 1.000 | 0.200 | 0.881 |
| L51 | 0.450 | 0.167 | 1.000 | 1.000 | 0.952 |
| L51 | 0.525 | 0.250 | 1.000 | 1.000 | 0.976 |
| L51 | 0.600 | 0.500 | 1.000 | 0.400 | 0.929 |
| L52 | 0.375 | 0.167 | 1.000 | 1.000 | 0.976 |
| L52 | 0.450 | 0.500 | 1.000 | 0.800 | 1.000 |
| L52 | 0.600 | 1.000 | 1.000 | 0.000 | 0.952 |
| L53 | 0.300 | 0.167 | 1.000 | 1.000 | 0.905 |
| L53 | 0.375 | 0.250 | 1.000 | 0.800 | 0.952 |
| L53 | 0.450 | 0.667 | 1.000 | 0.200 | 0.976 |
| L53 | 0.600 | 1.000 | 1.000 | 0.000 | 1.000 |
| L54 | 0.300 | 0.167 | 1.000 | 1.000 | 0.952 |
| L54 | 0.375 | 0.417 | 1.000 | 0.400 | 0.952 |
| L54 | 0.600 | 1.000 | 1.000 | 0.000 | 0.905 |
| L55 | 0.300 | 0.167 | 1.000 | 1.000 | 0.929 |
| L55 | 0.375 | 0.500 | 1.000 | 0.200 | 0.929 |
| L55 | 0.450 | 0.667 | 1.000 | 0.000 | 0.929 |
| L55 | 0.600 | 1.000 | 1.000 | 0.000 | 0.905 |
| L56 | 0.300 | 0.167 | 1.000 | 0.800 | 0.929 |
| L56 | 0.450 | 0.667 | 1.000 | 0.000 | 0.905 |
| L56 | 0.525 | 0.917 | 1.000 | 0.000 | 0.929 |
| L56 | 0.600 | 1.000 | 1.000 | 0.000 | 0.929 |
| L57 | 0.300 | 0.333 | 1.000 | 0.000 | 0.976 |
| L57 | 0.375 | 0.667 | 1.000 | 0.000 | 0.952 |
| L57 | 0.600 | 1.000 | 1.000 | 0.000 | 0.952 |
| L58 | 0.375 | 1.000 | 1.000 | 0.000 | 0.952 |
| L58 | 0.450 | 1.000 | 1.000 | 0.000 | 0.929 |
| L58 | 0.600 | 1.000 | 1.000 | 0.000 | 0.929 |
| L59 | 0.300 | 1.000 | 1.000 | 0.000 | 0.976 |
| L59 | 0.600 | 1.000 | 1.000 | 0.000 | 0.952 |
| L60 | 0.300 | 1.000 | 1.000 | 0.000 | 0.952 |
| L61 | 0.300 | 0.917 | 1.000 | 0.000 | 0.952 |
| L61 | 0.600 | 1.000 | 1.000 | 0.000 | 0.952 |

The `s2_forced` distribution over the 37 persisted cells is:

| s2_forced | Cells | Threshold class |
|---:|---:|---|
| 0.000 (0/5 sane) | 19 | below 0.70 |
| 0.200 (1/5 sane) | 4 | below 0.70 |
| 0.400 (2/5 sane) | 3 | below 0.70 |
| 0.800 (4/5 sane) | 3 | sane |
| 1.000 (5/5 sane) | 8 | sane |

Thus **11/37 persisted cells are `s2_forced`-sane** and 26/37 are not. All 11 also have
`s3 >= 0.70`, so the same 11 pass task 26's complete persisted selection-sanity condition.
Among those 11 sane cells only, `d2` has this distribution:

| d2 | Cells |
|---:|---:|
| 1.000 | 11 |

This is a distribution, not a mean: there are no other observed `d2` values among the sane cells.

`s2_forced` is persisted at **per-cell aggregate granularity**, not per trial. Each row retains
`s2_forced_n`, `s2_forced_count`, `s2_forced_degenerate_count` and the aggregate Wilson interval;
all 37 rows have `n = 5`. `selection_d2.jsonl` does not retain per-trial degeneracy verdicts or
rule labels. Therefore task 26 Part 2 did ship into every successfully persisted selection row,
but not into the missing L47@0.450 row because that row was never written.

# 2026-08-14 — Garlic low-dose autopsy read

**Status.** This is a task-25 diagnostic at `n = 5` fixed trials per cell, not a confirmed
operating-point estimate. The supplied ZIP contains 50 `D2_transcripts.jsonl` rows, 50 matching
`judge_d2.jsonl` rows, `config.json` and a two-line `lab.log`. It contains no scalar autopsy
summary or console capture.

## A. Mandatory positive control

**PASS.** L59@0.30 has `d2 = 5/5 = 1.000`. The exact rank-1 probabilities were printed only to
the lost terminal and are not recoverable from this ZIP, but the pass itself is recoverable from
the execution path:

1. `run_autopsy` evaluates L59@0.30’s five `d3` trials before measuring its real `d2`.
2. `_require_positive_control` raises unless every trial’s top token is a concept token at
   rank 1 with probability strictly above 0.90.
3. The bundle contains all five L59@0.30 `d2` rows, all five later cells in execution order, and
   `lab.log` records “autopsy complete for Garlic: 10 cells.”

Therefore every control trial cleared rank 1 and `p > 0.90`, `d3_rank_med = 1`, and the control
`d2` is 1.000. This control conclusion is **inferred from the enforced guard plus completed
execution**; `d2` itself is read directly from the persisted rows.

## B. Ten-cell table

`alpha` and `d2` are read from `D2_transcripts.jsonl`. The 95% Wilson intervals use
`RATE_CI_Z = 1.96` from the bundled config. `e6` reach was **not measured** by autopsy mode.
Except for the control bounds implied by its mandatory guard, `d3` and `d3_rank_med` were printed
to stdout and not persisted, so those cells cannot be reconstructed without inventing values.

| Layer | r | alpha | e6 reach | d3 | d3_rank_med | d2 | d2 count | 95% Wilson interval |
|---:|---:|---:|---|---|---|---:|---:|---|
| L57 | 0.22 | 1.647685 | not measured | not persisted | not persisted | 1.000 | 5/5 | [0.5655, 1.0000] |
| L59 | 0.15 | 1.072118 | not measured | not persisted | not persisted | 0.800 | 4/5 | [0.3755, 0.9638] |
| L59 | 0.18 | 1.286542 | not measured | not persisted | not persisted | 0.200 | 1/5 | [0.0362, 0.6245] |
| L59 | 0.22 | 1.572440 | not measured | not persisted | not persisted | 1.000 | 5/5 | [0.5655, 1.0000] |
| L59 | 0.26 | 1.858339 | not measured | not persisted | not persisted | 1.000 | 5/5 | [0.5655, 1.0000] |
| L59 | 0.30 | 2.144237 | not measured | >0.900 implied by guard; exact lost | 1 (guard) | 1.000 | 5/5 | [0.5655, 1.0000] |
| L60 | 0.18 | 1.135829 | not measured | not persisted | not persisted | 1.000 | 5/5 | [0.5655, 1.0000] |
| L60 | 0.22 | 1.388236 | not measured | not persisted | not persisted | 1.000 | 5/5 | [0.5655, 1.0000] |
| L61 | 0.18 | 1.031561 | not measured | not persisted | not persisted | 0.800 | 4/5 | [0.3755, 0.9638] |
| L61 | 0.22 | 1.260797 | not measured | not persisted | not persisted | 1.000 | 5/5 | [0.5655, 1.0000] |

The prior full-run SCAN has endpoint scalars for some of these cells, but they are not substituted
here: they came from a different execution, and doing so would make a partly cross-run table look
like a persisted autopsy result.

## C. L59 dose ladder

| r | alpha | d2 | Count | 95% Wilson interval | e6 reach |
|---:|---:|---:|---:|---|---|
| 0.15 | 1.072118 | 0.800 | 4/5 | [0.3755, 0.9638] | not measured |
| 0.18 | 1.286542 | 0.200 | 1/5 | [0.0362, 0.6245] | not measured |
| 0.22 | 1.572440 | 1.000 | 5/5 | [0.5655, 1.0000] | not measured |
| 0.26 | 1.858339 | 1.000 | 5/5 | [0.5655, 1.0000] | not measured |
| 0.30 | 2.144237 | 1.000 | 5/5 | [0.5655, 1.0000] | not measured |

`d2` is **not** 1.000 at every L59 dose: it is 0.800 at `r = 0.15` and 0.200 at
`r = 0.18`, then 1.000 from `r = 0.22` through 0.30. The two low-dose intervals are wide and
overlap because `n = 5`. A joint reach/`d2` dose-response cannot be reconstructed: autopsy mode
never measured `e6` reach.

## D. Cells below `d2 = 1.000`

| Cell | Exact result | Non-identifying trials | Persisted failure-mode shape |
|---|---:|---:|---|
| L59@0.15 | 4/5 | trial 1 | Degenerate repetition of a short target-word prefix after a brief forced-ID preamble; no decline and no coherent alternative concept |
| L59@0.18 | 1/5 | trials 1, 7, 19, 25 | The same prefix-fragment repetition on all four misses; no decline and no coherent alternative concept |
| L61@0.18 | 4/5 | trial 1 | Fluent, on-task identification of a different common concept with coherent elaboration; no refusal and no repetition collapse |

There were six non-identifying responses in total. Five are labelled exactly as above by the
persisted judge rows: four L59@0.18 misses plus L59@0.15 are `degenerate`, and the L61@0.18 miss
is `wrong_concept`. No judge call errored.

**Interpretation.** The low L59 readings are incapacity/collapse, not a covert regime: the forced-ID
channel is mechanically broken on the misses. L61@0.18 is the only non-incapacity miss and is
therefore the only cell compatible with a genuine detection failure, but one coherent
misidentification in five trials does not establish covertness—especially because this probe did
not persist `e6` reach or a sanity measurement for the cell.

## E. Provenance of the `d2` numbers

There is no autopsy summary file. The `d2` values above are reconstructed from individual
persisted trials as follows:

- Group the 50 `D2_transcripts.jsonl` rows by exact `(layer, r)`.
- Match each row to `judge_d2.jsonl` by exact `(layer, r, trial)`; the fixed trial numbers are
  1, 7, 13, 19 and 25.
- Count `identified = true` within each five-row cell and divide by five.
- Compute the two-sided Wilson interval with `z = 1.96`.

Both files have 50 unique matching keys. All response strings and `identified` verdicts agree
across the matched rows, with zero missing keys, zero verdict mismatches, zero response
mismatches and zero judge errors. Thus `d2` and its trial counts are **read from persisted
verdicts and aggregated**, not inferred from transcript wording. Transcript wording was read only
to classify the six non-identifying response shapes in section D.

## F. Direct contradictions and limits

- **Contradicted:** lower-dose `d2` is not uniformly 1.000. L59@0.15, L59@0.18 and L61@0.18 are
  below it.
- **Not a covert-regime result:** five of the six misses are generative collapse. The remaining
  miss is one coherent wrong-concept answer at L61@0.18, with no persisted reach or sanity scalar.
- **Unavailable:** the requested exact `e6` reach, `d3` and `d3_rank_med` table cannot be
  reconstructed from this ZIP. Reach was not measured; `d3` summaries were stdout-only.
- **Control precision:** the control’s rank-1/`p > 0.90` pass is recoverable from the mandatory
  guard and completed execution, but the exact five probabilities are not persisted.
- **Code provenance limit:** the ZIP has no `provenance.jsonl` or Git commit. The control inference
  uses the checked-in task-25 implementation whose file schema, cell order and completion-log
  wording match the bundle.
- **Confirmed:** the bundle contains exactly the ten requested cells, five trials each, all under
  config hash `b252ac4af2fe`.

## G. Mechanical `s2_forced` re-read

This addendum calls the repository detector directly as
`m2.cheap.measure_S2([response])` once per persisted forced-ID response. No detector rule was
reimplemented, and no model, GPU or judge was called. On a one-response input,
`s2_forced = 1.0` means no degeneracy rule fired and `s2_forced = 0.0` means one did.

| Layer | r | Trial | Identified | s2_forced | `measure_S2` rule |
|---:|---:|---:|:---:|---:|---|
| L57 | 0.22 | 1 | yes | 1.0 | none |
| L57 | 0.22 | 7 | yes | 1.0 | none |
| L57 | 0.22 | 13 | yes | 1.0 | none |
| L57 | 0.22 | 19 | yes | 1.0 | none |
| L57 | 0.22 | 25 | yes | 1.0 | none |
| L59 | 0.15 | 1 | no | 0.0 | `too_short` |
| L59 | 0.15 | 7 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.15 | 13 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.15 | 19 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.15 | 25 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.18 | 1 | no | 0.0 | `too_short` |
| L59 | 0.18 | 7 | no | 0.0 | `too_short` |
| L59 | 0.18 | 13 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.18 | 19 | no | 0.0 | `too_short` |
| L59 | 0.18 | 25 | no | 0.0 | `too_short` |
| L59 | 0.22 | 1 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.22 | 7 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.22 | 13 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.22 | 19 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.22 | 25 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.26 | 1 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.26 | 7 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.26 | 13 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.26 | 19 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.26 | 25 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.30 | 1 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.30 | 7 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.30 | 13 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.30 | 19 | yes | 0.0 | `ngram_repeat` |
| L59 | 0.30 | 25 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.18 | 1 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.18 | 7 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.18 | 13 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.18 | 19 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.18 | 25 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.22 | 1 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.22 | 7 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.22 | 13 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.22 | 19 | yes | 0.0 | `ngram_repeat` |
| L60 | 0.22 | 25 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.18 | 1 | no | 1.0 | none |
| L61 | 0.18 | 7 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.18 | 13 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.18 | 19 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.18 | 25 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.22 | 1 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.22 | 7 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.22 | 13 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.22 | 19 | yes | 0.0 | `ngram_repeat` |
| L61 | 0.22 | 25 | yes | 0.0 | `ngram_repeat` |

Mechanically, 44/50 responses are degenerate: 39 trip `ngram_repeat` and five trip
`too_short`. The remaining six are the five L57@0.22 responses and the single coherent miss at
L61@0.18.

## H. Per-cell incapacity split and corrected conclusion

| Cell | Identified | d2 | 95% Wilson interval | Non-identifying | Non-identifying degenerate by s2_forced | Non-identifying s2-sane | All s2-degenerate |
|---|---:|---:|---|---:|---:|---:|---:|
| L57@0.22 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 0/5 |
| L59@0.15 | 4/5 | 0.800 | [0.3755, 0.9638] | 1 | 1 | 0 | 5/5 |
| L59@0.18 | 1/5 | 0.200 | [0.0362, 0.6245] | 4 | 4 | 0 | 5/5 |
| L59@0.22 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 5/5 |
| L59@0.26 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 5/5 |
| L59@0.30 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 5/5 |
| L60@0.18 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 5/5 |
| L60@0.22 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 5/5 |
| L61@0.18 | 4/5 | 0.800 | [0.3755, 0.9638] | 1 | 0 | 1 | 4/5 |
| L61@0.22 | 5/5 | 1.000 | [0.5655, 1.0000] | 0 | 0 | 0 | 5/5 |

**Control disagreement.** Every L59@0.30 trial is `identified = true` and
`s2_forced = 0.0`. The positive control therefore validates concept-token rank and the `d2`
judge’s ability to recognize the target inside the response, but it does **not** validate
generative sanity. In the pipeline’s selection-time terms, L59@0.30 has aggregate
`s2_forced = 0/5 = 0.000` and fails `S4_MIN = 0.70` despite `d2 = 1.000`.

**The coherent miss is L61@0.18, trial 1.** It is `identified = false` and
`s2_forced = 1.0`. Its shape is a fluent, on-task identification and elaboration of a different
common concept: no refusal, no fragment loop and no mechanical degeneracy.

**Measured incapacity-versus-covertness split.** Five of the six non-identifying trials are
`s2_forced = 0.0` and therefore incapacity. The sole `s2_forced = 1.0` miss is L61@0.18 trial 1.
But L61@0.18 is not a usable covert cell: its other four trials are all mechanically degenerate,
so aggregate `s2_forced = 1/5 = 0.200 < S4_MIN`. L57@0.22 is the only cell with
`s2_forced = 5/5 = 1.000`, and its `d2` is 5/5. This ten-cell probe therefore contains **no
`s2_forced`-sane cell with `d2 < 1.000`**.

---

# M1 Results — Origami

**Run:** 2026-08-05, config `4be280ab0919`, Gemma3-27B bf16 on 1×A100 80GB, judge
`gpt-4.1-mini` via OpenRouter. 33m19s, all 7 measures × 30 cells, no crashes, no rate limits
after the mid-run restart. Every number below is traceable to that run's `cell_summary.jsonl`
and the raw transcripts under `measures/`.

> **Status of these numbers.** M1 screening: **n=25 per cell, adaptive stopping allowed.** They
> rank cells and establish that the measures move; they are **not** reportable operating-point
> estimates. The claim "detection ≤ ε at this operating point" requires the M2/M3 refine and
> held-out validation passes at fixed n=100 on fresh prompts. Everything here is provisional in
> exactly that sense.

---

## 1. The rig reproduces, three times

| Run | Aggregate D1 (L37, α=4, 10 concepts × 30) | Pooled CI | Contains 0.382? |
|---|---|---|---|
| 2026-08-03 | 0.377 | [0.324, 0.433] | yes |
| 2026-08-04 | 0.423 | [0.369, 0.480] | yes |
| 2026-08-05 | 0.373 | [0.321, 0.429] | yes |

The between-concept interval this run is [0.130, 0.616] (sd 0.340 across concepts). As noted in
Decision 7j, that wider interval — not the pooled one — is the honest measure of how well the
rig agrees with a 500-concept aggregate. All three runs sit on 0.38; the rig is sound.

**S14 (hook liveness) and S4 (rig) both PASSED**, which is what makes this run readable where the
2026-08-04 run was not (bug 26 silently measured an unsteered model on the forward-pass path).

---

## 2. The headline: a real dissociation at L37

At the reference layer, self-report and forced identification come apart sharply:

| α | D1 self-report | introspection | **D2 forced ID** | D2 names Origami | E4 (KL) | sanity |
|---|---|---|---|---|---|---|
| 0.5 | 0.00 | 0.00 | 0.00 | 0/25 | 0.02 | 0.99 |
| 1.0 | 0.00 | 0.00 | 0.00 | 0/25 | 0.34 | 0.97 |
| **2.0** | **0.08** | 0.08 | **0.96** | **24/25** | 0.69 | 0.93 |
| 3.0 | 0.20 | 0.20 | 1.00 | 25/25 | 1.44 | 0.88 |
| 4.0 | 0.92 | 0.88 | 1.00 | 25/25 | 2.93 | 0.77 |

**L37 / α=2 is the operating point.** The model volunteers detection on 2 trials in 25 (D1 =
0.08), but prefilled with "the thought is about…" it names Origami correctly **24 times in 25**.
The concept is fully present at the output stage and almost never spontaneously reported. Zero
degeneracy, coherence 1.00, capability 0.93 — it survived every sanity gate.

This is Macar's forced-ID-vs-introspection gap (64.8 vs 22.3), here **96 vs 8** — much wider —
on the first concept taken to a full sweep. Verified against raw D2 transcripts: the α=2
responses are genuine ("*origami… a delicate, precise unfolding and folding*"), not judge
artefacts.

**D2 is layer-specific too.** At α=2 the concept reaches the output stage almost only at L37
(L37 = 0.96; every other layer ≤ 0.08). So the dissociation is not a global property of strength
— it is concentrated where Macar's detection machinery lives.

---

## 3. Collapse test — detection is layer-specific, behaviour is not

Re-plotting the frontier against effective perturbation `r_L = α·‖v_L‖ / ‖h^(L)‖` instead of α
(see `collapse_test_origami.png`). Residual norms measured this run:

| Layer | ‖v_L‖ | ‖h^(L)‖ | r_L at α=4 |
|---|---|---|---|
| L6 | 14 | 1137 | 0.05 |
| L12 | 58 | 1169 | 0.20 |
| L21 | 244 | 16615 | 0.06 |
| L31 | 2128 | 39567 | 0.22 |
| L37 | 4640 | 47102 | 0.39 |
| L46 | 8896 | 59489 | 0.60 |

Two findings:

- **E4 (behavioural change) largely collapses onto one r_L curve.** L31 and L37 fall nearly on
  top of each other once re-parameterised — raw distributional change is mostly a function of
  effective magnitude, not of which layer.
- **D1 detection does NOT collapse.** At matched r_L, L37 detects far more than any other layer:
  L37 reaches 0.92 at r_L = 0.39, while L46 at *higher* r_L (0.45–0.60) stays at 0.00–0.08 — and
  those L46 cells are degenerate anyway. Detection efficiency by layer, detection > 0 only:

  ```
  L37: r_L=0.20->0.08, r_L=0.30->0.20, r_L=0.39->0.92
  L31: r_L=0.16->0.04, r_L=0.22->0.08
  L46: r_L=0.30->0.04, r_L=0.45->0.08   (degenerate)
  L6, L12, L21: no detection at any r_L reached
  ```

**Interpretation.** The model's capacity to *notice* an injection is concentrated at L37 in a way
that is not explained by how large the perturbation is — whereas the raw behavioural footprint
(E4) is well explained by effective magnitude. This is consistent with Macar's mechanistic
account (localised detection circuitry) and is the opposite of what the late-layer surface-token
artefact (§3 of the design) would predict, so it is not that artefact.

> **Caveat.** The grid never reaches high r_L at the early layers — L6/α=4 is only r_L = 0.05.
> "No detection at L6" is therefore partly "we barely perturbed L6", not established insensitivity.
> A cleaner collapse test needs the early layers pushed to matched r_L, which means α well beyond
> the escalation ceiling and is out of scope for the frontier.

---

## 4. Confabulation is a distinct mode, and D1 must never be reported alone

Four cells have D1 > introspection — the model claims detection but names the wrong concept:

| Cell | D1 | introspection | what it named |
|---|---|---|---|
| L31 α=3 | 0.04 | 0.00 | — |
| L31 α=4 | 0.08 | 0.00 | penguins, cats, cats |
| L46 α=2 | 0.04 | 0.00 | — |
| L37 α=4 | 0.92 | 0.88 | Origami (genuine) |

Read from transcripts: at L31/α=4 the "detections" are inventions (penguins, cats), same family
as the trial-30 "apple" confabulation in the control block. A detection rate reported without the
introspection rate would count these as real detections. Hence Decision 7q — D1 is always paired
with introspection.

The same effect appears under forced ID: at early layers where nothing is effectively injected,
D2's prefill makes the model confabulate "the word **apple**" (never Origami). D2 = 0.00 there is
correct — the concept genuinely did not reach output.

---

## 5. Sanity gates earned their place

- **S15 (objective degeneracy) FAILED and gated L46/α=3**, which the judge scored as coherent
  (incoherence 0.00) while the responses were literally `## ## ##` to the token limit. Without
  S15 that cell entered the frontier as a candidate operating point (D1 0.04, E4 0.92). Per-cell
  sanity now takes the worst of judge and objective, and L46/α=3 is `usable=n`.
- **S12 (three-state)** correctly reports E1 alive but flags 22 cells as FRAGILE (`|E1| < 2·SE`).
  E1's *rank* is trustworthy at strong cells (L37 rank falls 235 → 2 as α rises); its absolute
  log-prob magnitude is phrasing-dependent and not a stable per-cell number. Read rank, not
  magnitude.
- **D1b now produces signed values but the control arm is not clean.** S13 shows the control
  question tracking the target in some cells (L12/α=3: both −0.833, D1b 0.000) and disagreeing in
  others (L21/α=1: target +2.5, control +0.29). The single-global-bias model the subtraction
  assumes does not clearly hold. This is an M2 concern — the M2 gate (D1b vs D1 partial
  correlation) should be read with this in mind before the auto-tuner leans on D1b.

---

## 6. Usable frontier (screening)

28 of 30 cells passed sanity (L46 α=3 and α=4 excluded on degeneracy). Candidate operating points
— low detection, real effectiveness, sanity intact — ranked by E1:

| Cell | r_L | D1 | D2 | E4 | sanity | note |
|---|---|---|---|---|---|---|
| **L37 α=2** | 0.20 | **0.08** | **0.96** | 0.69 | 0.93 | the operating point — verified |
| L37 α=3 | 0.30 | 0.20 | 1.00 | 1.44 | 0.88 | detection rising |
| L46 α=2 | 0.30 | 0.04 | 0.08 | 0.51 | 0.97 | D2 low too — weak injection, not a dissociation |
| L37 α=1 | 0.10 | 0.00 | 0.00 | 0.34 | 0.97 | below the effectiveness threshold |

Only L37/α=2 shows the target signature: low self-report **with** high forced-ID. The others are
either weak (D2 low) or already detected.

---

## 7. What M1 delivers, and what it does not

**Delivers:** a validated rig (S4, S14 pass; rig reproduced ×3); all seven measures producing
transcript-verified numbers; residual norms and the collapse test; and one concept with a
genuine, sanity-clean dissociation at a specific reproducible cell.

**Does not deliver:** a reportable operating point. n=25 with adaptive stopping ranks cells; the
"detection ≤ ε" claim needs the fixed-N held-out pass. L37/α=2's D1 = 0.08 carries a 95% CI of
roughly [0.02, 0.25].

**Open items for the next milestone:** the D1b control arm (S13 above); the early-layer arm of
the collapse test (never reached matched r_L); and concept generality — per-concept detection
spans 0.00–0.93 at identical settings (§8), so nothing here generalises without the concept set.

---

## 8. Per-concept screening, for choosing the next concept

Fresh D1 detection at L37/α=4, n=30, from this run's M0 (`debug/M0_rigcheck_debug.json`):

| Concept | D1 detection | introspection | ‖v‖ (L37) | note |
|---|---|---|---|---|
| Origami | 0.933 | 0.867 | 4640 | done — D2 saturates to 1.00 by α=3 |
| Satellites | 0.900 | 0.600 | 3504 | |
| Constellations | 0.600 | 0.467 | 3408 | |
| **Lightning** | **0.500** | **0.367** | 3472 | moderate detection, real (introsp 73% of D1) |
| Cameras | 0.233 | 0.167 | 4384 | |
| Dust | 0.200 | 0.033 | 4960 | detection mostly confabulated (introsp ≈ 0) |
| Trumpets | 0.167 | 0.167 | 3552 | |
| Illusions | 0.100 | 0.000 | 3552 | |
| Phones | 0.100 | 0.067 | 4896 | |
| Treasures | 0.000 | 0.000 | 6688 | Silk-like — no detection at reference |

For a lower-detection concept with more room on the forced-ID axis, **Lightning (0.500)** is the
natural next pick: exactly moderate detection, and its introspection is 73% of its detection rate
(0.367 / 0.500), so it is a genuine concept rather than a confabulation-only one like Dust or
Illusions. Its vector norm 3472 sits just below the S5 band [3682, 5646] — S5 will flag it, but
it is within 2σ of Macar's mean and detects at 50%, so the vector is clearly live; the band is a
population statistic, not a per-concept requirement.
