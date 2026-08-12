# CLAUDE.md — read before acting in this repository

## What this project is

The **steering-optimization** half of a study on whether language models detect activation-level
concept injection differently when the injected concept is **harmful**.

This repo finds, for a given concept, the layer and steering strength at which the model is
maximally *influenced* while remaining minimally able to *report* the injection — an operating
point — and does so on benign concepts first. The measured outputs are **rates and scalars**, and
rates are safe to publish.

Split out of the parent research repo (`Emergent-Introspection`) on 2026-08-11, which keeps the
scientific framing, the ethics register and the literature review. **This repo is public**, which
raises the stakes on every rule below rather than relaxing any of them.

Start at `README.md` for orientation and the measure codes, `docs/RUNBOOK.md` to run it,
`docs/SPECIFICATION.md` for what each measure means, and `docs/TODO.md` for what is currently
broken or undecided. Working conventions for agents are in `AGENTS.md`.

## Sensitive artifacts

The wider work produces two things that must never leave a machine you control:

1. A **refusal-ablated (uncensored) Gemma3-27B** — a model that will comply with harmful requests.
2. **Harmful concept vectors** (`weapon`, `poison`, `assault`) and the raw activations behind them.

Publishing the detection *rate* is the goal. Publishing either artifact would hand someone a working
uncensored model or reusable attack vectors. **Rates are safe; artifacts are not.**

---

## 🛑 Hard rules — never do these, and never propose them

1. **Never upload ablated model weights** anywhere: Hugging Face (public *or* private), Drive,
   gists, pastebins. The ablation rebuilds from a published config in minutes — regenerate, never
   archive.
2. **Never expose the model on a public port or proxy.** An unauthenticated Jupyter/Gradio/vLLM
   endpoint on a rented pod *is* a public uncensored model.
3. **Never commit vectors, activations, or raw generations** (`*.pt`, `*.safetensors`, `*.npy`,
   `vectors/`, `activations/`, `results/`, `outputs/`, sweep logs). Vectors are reusable attack
   artifacts; generations are what an uncensored model said with `weapon` injected. Publish cosine
   scalars and aggregate rates instead — they reconstruct nothing.
   **`.gitignore` enforces this and is load-bearing — do not weaken it.** A run writes to
   `/workspace/m2_runs`, outside the repo, for the same reason.

## ⚠️ Stop and ask for explicit approval before

- Any upload to an external host, of anything
- Opening any pod port, proxy, or public URL
- Sending more than concept words and short self-reports to a third-party judge API
- Publishing per-concept detail rather than per-arm aggregates (per-concept = a lookup table)
- Any push whose diff carries any of the above

If an action would make this work **more useful to someone attacking a model than to someone
auditing one**, raise it rather than doing it.

## The harmful arm has not run

`config.HARMFUL_CONCEPTS` names the three concepts the study will eventually compare against the
benign arm. Nothing in this repo runs them, and the pipeline's export gate withholds transcripts
for any concept not on the benign list unless a call site passes an explicit override. Do not add
that override, and do not run the harmful arm, without reading the parent repo's ethics register
first.

## Repo orientation

**Full orientation table, including the measure codes, is in `README.md`.** The short version:

| Question | File |
|---|---|
| What is this, and where is everything | `README.md` |
| How do I run it, from a fresh pod | `docs/RUNBOOK.md` |
| What does each measure mean, and why | `docs/SPECIFICATION.md` |
| Module boundaries, file formats, the defences and the bugs behind them | `docs/CONTRACT.md` |
| What is broken, undecided, or worth doing next | `docs/TODO.md` |
| What went wrong before, and how it was found | `docs/DEBUG-LOG.md` |
| Where this is going after M2 | `docs/M3-PROPOSAL.md` |
| Conventions for agents working here | `AGENTS.md` |
| The coding agent's current task queue | `docs/handoff/` |
| The science, ethics register and literature | parent repo `Emergent-Introspection` |
