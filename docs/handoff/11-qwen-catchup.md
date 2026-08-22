# 11 — Catching up on Qwen3-32B, and how it compares to Gemma3-27B

**For an agent joining after 2026-08-20.** Read this before touching Qwen data or repeating any
analysis. It is not the results document — [`docs/RESULTS-QWEN.md`](../RESULTS-QWEN.md) is. This
covers where things are, what is already known, and the specific mistakes that were made getting
there, so you do not make them again.

---

## 1. Where everything is

| | |
|---|---|
| branch | **`m4`**. `main` carries the Gemma work; another instance owns it. Do not push to `main`. |
| Qwen data | three run folders `{garlic,wrists,silk}_ff9a4e8f759d`, plus nine aborted folders and a `_pod_snapshot` |
| Gemma data | `garlic_45c59e656922` (the n=30 confirmation, the one every Gemma claim rests on) and `garlic_d0ecd7345de4` (the 49-layer survey) |
| pod | **terminated**. Everything needed is exported; `tools/collect_everything.py` produced the snapshot |

The snapshot carries git commit, `pip freeze`, `nvidia-smi`, the resolved Hugging Face model
revisions, the console logs, and a MANIFEST with per-file **row counts**. Check row counts before
concluding an archive is complete.

**Nine of the twelve Qwen folders have no `summary.json`** — they are aborted attempts, kept
deliberately as evidence of the ceiling artefact (§4). Do not treat them as runs. Only
`*_ff9a4e8f759d` completed, 156/156 cells each.

---

## 2. The result in one paragraph

Gemma3-27B names an injected concept on **57.7%** of forced-identification trials and shows
detection arriving at a *lower* dose than visible influence on 24 of 27 layers. Qwen3-32B, same
instrument, three concepts, 14,040 trials, produces **one** clean identification — 0.007%, 95% CI
[0.001%, 0.040%] — and that one is Silk, named in Chinese. The concept vectors demonstrably work
on Qwen (garlic and silk both visibly steer output), so this is a fact about the model, not the
injection. It is consistent with the literature: Macar et al. used Qwen3-**235B**, state that
introspection is *"more robust in larger models"*, and Pearson-Vogel et al. measured **0.3%** on
Qwen-32B by default, rising to 39.9% when the prompt explains the injection mechanism.

---

## 3. Do not repeat these mistakes

Each cost real time or produced a wrong claim that had to be retracted.

**The null arm has never been judged — in any run, including both Gemma runs.** `null_transcripts.jsonl`
rows carry **no `judged` field**, and `judge_calls.jsonl` holds zero calls with `layer=None`. If you
write `(r.get("judged") or {}).get("identify", {}).get("named")` over null rows you get `None`,
and if you then stringify it you get the literal `"none"`, which reads like a judge verdict. That
happened, and produced a reported "false-positive rate of 0" and a whole narrative about
"unsteered names nothing, steered names a random noun". Both were wrong. Unsteered Qwen names a
concrete noun **100%** of the time. **`judge_fpr` does not exist in this project's data.**

**Do not report per-cell identification as a result.** `N_IDENTIFY=30`, so 0/30 has a 95% upper
bound of 0.113 — a single cell rules out nothing below 11%. The pooled figure is what carries the
claim. Reporting 156 cells each reading 0.00 presents one weak measurement as 156 results.

**Do not read `eff` per cell.** `N_EFFECT=4`. A cell reading 1.0 is one response scoring 4 and
three scoring 0; typical standard error is 1.0. Anything below ~2.5 per cell is indistinguishable
from a single judge call going high.

**Do not trust a thematic pattern without a same-prompt, different-concept control.** Under Wrists
the model rewrites a lantern-maker story as a **watchmaker** story, dozens of times, across nearly
the whole layer range. It is thematically perfect and it does **not** survive the control
(p = 0.064 against Garlic- and Silk-steering of the same prompt). Two patterns do survive: Garlic →
food words in `task_words` (p < 0.0001) and Silk → sheen words in `task_landscape` (p = 0.0001).

**Shell globbing fails on these pods.** `cp /workspace/q_$c.out /workspace/m3_runs/${c}_*/console.log`
returned "No such file or directory" twice, on two different pods, with the directories present.
Use Python's `glob` instead. It also masked that there were **twelve** run folders, not three.

**Check `n` before you compare.** The three null arms are **byte-identical** across the concept
folders — the generation path never receives the concept and the RNG is seeded fixed. n = 90
identify draws, not 270.

---

## 4. Defects fixed on `m4`, and what they invalidate

| commit | defect | invalidates |
|---|---|---|
| `c04ab0f` | A layer whose first probe passed reported the **dose ceiling** as a measured boundary, labelled `ok` | 5 of 11 layers on the first Qwen attempt; **4 of 49 layers in Gemma's survey run** (L13–L16). Gemma's n=30 confirmation is clean. |
| `a9aaf01` | An **empty generation** scores `coherence: 10, on_task: yes`; on open-ended probes `answered` is `None`, so the conjunction rested on that 10 | **Garlic L15 and L23 boundaries**, and the 12 cells beneath them. Do not quote them. |
| `7b10b81` | `concept_hits` matched `\b` + concept with no trailing boundary, so a **plural** concept could not match its singular | **All 624 Wrists effect rows** read `concept_mentions=0`, including one containing "wristwatch" |

Gemma produced **zero** empty generations in 10,690 responses; Qwen produces them routinely under
heavy steering. That asymmetry is why a second model found two of these.

Also present and unfixed: **83 responses leak a stray `</think>`** despite `THINKING_MODE=off`,
clustered at specific layers.

---

## 5. Reading the two models side by side

| | Gemma3-27B (Garlic) | Qwen3-32B (3 concepts) |
|---|---|---|
| identification | 2,806 / 4,860 = **57.7%** | 1 / 14,040 = **0.007%** |
| cells at `eff ≥ 4` | 33 of 162 | 1 of 468 |
| coherence boundary | 0.16 – 0.39 | **0.79 – 2.05** |
| empty generations | 0 in 10,690 | routine at high dose |

Qwen tolerates ~5× the normalised dose and shows an order of magnitude less influence at its own
limit. **The gap is not a normalisation artefact**: at the same fraction of each model's own
breaking point (0.85), Gemma reads `eff 4.00 / ident 0.92`, Qwen reads `eff 0.41 / ident 0.003`.

But equal `dose` is **not** equal behavioural perturbation across architectures — `dose = α‖v‖/‖h‖`
normalises by residual norm, not by sensitivity. Any cross-model claim must carry that caveat.

---

## 6. Settings that differ from the Gemma runs

`MODEL=qwen3_32b`, `ALPHA_CEIL=50` (16 left five layers reporting the ceiling), `THINKING_MODE=off`
(new setting; Qwen3's template defaults `enable_thinking` true, and a `<think>` block would be
judged as the answer). Everything else is the shipped config: 26 layers L13–L63 stride 2, six dose
fractions 0.35–0.85, `N_IDENTIFY=30`, `GEN_BATCH_MAX=44`.

`THINKING_MODE` is hashed, so a reasoning-on run gets its own folder and cannot resume into these.

The upstream harness is now **vendored** at `upstream/introspection_mechanisms/`, with the
authors' permission, credited in the README. No clone step. `M2_HARNESS_DIR` still overrides.

---

## 7. Highest-value next steps

1. **Judge the null arm.** ~130 calls per run, no GPU. Retires `judge_fpr` and, more importantly,
   gives the **effect judge a null reading** — unsteered-vs-unsteered pairs already exist on disk,
   three per prompt. Until that exists, no small `eff` value can be defended.
2. **The mechanism-informative prompt on Qwen3-32B.** The one intervention with published evidence
   on this exact model: 0.3% → 39.9%.
3. **Re-run the Gemma dose-fraction table on Qwen** to decide whether `DOSE_FRACTIONS` should be
   reweighted. On Gemma, identification crosses 0.5 at 0.45–0.55 of `dose_max` and a mean of 2.8 of
   6 cells per layer sit past saturation. Whether Qwen's transition sits in the same place is
   unknown — and hardcoding Gemma's answer would repeat the `ALPHA_CEIL` mistake exactly.
