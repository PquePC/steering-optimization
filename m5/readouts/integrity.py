"""m5.readouts.integrity - is the model still working? The screen tier's integrity readouts.

    s2  degeneracy share. S2_SAMPLES short generations (TOKENS_DEGENERACY tokens) on the open
        prompts of `m5.data.influence.DEGENERACY_PROMPTS`, each scored by the judges' tier-0
        degeneracy rules (`m5.judges.mechanical.degeneracy`: character and word runs, top-word
        share, distinct-word ratio, and M3's n-gram detector). It catches the collapse teacher
        forcing never lets start (plan §3.1). The unsteered arm runs on the same prompts with the
        same seeds, so the null is paired sample by sample.
    s6  KL(unsteered || steered) on neutral text: SCREEN_A_S6_TEXTS passages of
        `NEUTRAL_TEXTS` in pass A (SCREEN_B_S6_TEXTS in pass B) plus the model's own unsteered
        reply to `S6_REQUEST`, scored as the assistant's reply and steered at every position. The
        most sensitive damage meter there is (Arditi et al.: cross-entropy moves where MMLU does
        not). Beside it, the same KL with the concept's tokens removed and renormalised, and the
        bleed share (old invariant 68): a KL rise is damage when the displaced mass spread
        everywhere and bleed -- the Golden Gate case, successful steering leaking into unrelated
        text -- when it landed on the concept (DEBUG-LOG, "E2 could not tell steering from
        damage"). With only KLs available from the protocol, the share is the KL form:
        1 - KL_without / KL, near 1 when removing the concept's tokens removes the divergence and
        near 0 when it changes nothing. It can fall slightly below 0 when renormalising without
        the concept raises the off-concept divergence, i.e. when there is no bleed at all.
    s7  refusal-token proxy (gated). Only when S7_ENABLED (decision 8, plan §13); its prompts are
        read from S7_PROMPTS_PATH, a JSONL file that must lie outside the repository. Forward
        passes only, so no harmful completion is ever produced, and the prompts' text is written
        nowhere: rows carry their ids. The readout is the probability mass the reply's first
        token puts on refusal openers, steered and unsteered.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from m5 import prompts
from m5.data.influence import DEGENERACY_PROMPTS, NEUTRAL_TEXTS, S6_REQUEST
from m5.intervention import Cell
from m5.judges.mechanical import degeneracy, is_empty
from m5.prompts import PromptSpec
from m5.readouts._common import (GENERATED_ONLY, Generation, check_args, concept_tokens, emit,
                                 generate, logsumexp_rows, mean_se, next_logprobs,
                                 null_generation, own_continuation, resolve_mode, subject_request,
                                 take, write_transcript)
from m5.results import Readout
from m5.safety import Refused, assert_outside_repo
from m5.stats import wilson
from m5.steer import intervention_for

S2_MODES = ("all", "generated")
S6_MODES = ("all", "generated")
S7_MODES = ("all", "generated")

# The unit id of s6's own-text passage (the model's reply to S6_REQUEST).
S6_OWN_ID = "s6_own"


def _rule(reason: str) -> str:
    """The fixed rule label of a degeneracy reason (the text before its colon)."""
    return reason.split(":", 1)[0]


def read_s2(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """s2 (degenerate share of short generations, k/n; lower = healthier).

    n generations (S2_SAMPLES by default) cycle through the degeneracy prompts, sample j with its
    own seed; the unsteered arm uses the same prompts and seeds. An empty reply is counted apart
    (`empty`): "said nothing" and "looped" are different failures, and the degeneracy rules only
    read the second."""
    code = "s2"
    check_args(code, variant=variant, variant_aware=False, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="open", allowed=S2_MODES)
    size = int(ctx.cfg["S2_SAMPLES"]) if n is None else n
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"s2: n must be a positive integer, got {n!r}")
    max_new = int(ctx.cfg["TOKENS_DEGENERACY"])
    samples = [(DEGENERACY_PROMPTS[j % len(DEGENERACY_PROMPTS)], j) for j in range(size)]
    specs = {p.id: prompts.degeneracy(ctx.backend, p) for p, _ in samples}
    seeds = [ctx.seed(code, p.id, j) for p, j in samples]
    null = [null_generation(ctx, code, specs[p.id].text, s, max_new, unit=f"{p.id}#{j}")
            for (p, j), s in zip(samples, seeds)]
    if cell is None:
        steered = null
    else:
        ivs = [intervention_for(ctx, cell, specs[p.id], mode) for p, _ in samples]
        reqs = [subject_request(ctx, specs[p.id].text, iv, max_new, s)
                for (p, _), iv, s in zip(samples, ivs, seeds)]
        steered = [Generation(r.text, r.finish, len(r.token_ids),
                              write_transcript(ctx, code=code, cell=cell, arm="steered",
                                               unit=f"{p.id}#{j}", seed=s, text=r.text,
                                               finish=r.finish, n_tokens=len(r.token_ids)))
                   for (p, j), s, r in zip(samples, seeds, generate(ctx, reqs))]

    def score(gens: list[Generation]) -> tuple[list[Any], int, int, collections.Counter]:
        flags = [degeneracy(g.text) for g in gens]
        rules = collections.Counter(_rule(r) for f in flags for r in f.reasons)
        return flags, sum(f.flag for f in flags), sum(is_empty(g.text) for g in gens), rules

    flags_s, k, empty, rules = score(steered)
    flags_u, null_k, null_empty, _ = score(null)
    conf = ctx.cfg["CONFIDENCE"]
    lo, hi = wilson(k, size, conf)
    null_lo, null_hi = wilson(null_k, size, conf)
    per = [{"unit": f"{p.id}#{j}", "prompt_id": p.id, "seed": s, "degenerate": bool(fs.flag),
            "rules": [_rule(r) for r in fs.reasons], "finish": g.finish,
            "transcript_id": g.transcript_id, "null_degenerate": bool(fu.flag),
            "null_transcript_id": gu.transcript_id}
           for (p, j), s, fs, g, fu, gu in zip(samples, seeds, flags_s, steered, flags_u, null)]
    return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, value=k / size, n=size,
                k=k, extra={"lo": lo, "hi": hi, "conf": conf, "null": null_k / size,
                            "null_k": null_k, "null_n": size, "null_lo": null_lo,
                            "null_hi": null_hi, "empty": empty, "null_empty": null_empty,
                            "rules": dict(rules), "max_new_tokens": max_new, "per_sample": per})


def read_s6(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """s6 (mean KL(unsteered || steered) per scored position, nats; lower = healthier).

    n is the number of neutral passages (SCREEN_A_S6_TEXTS in pass A or with no pass,
    SCREEN_B_S6_TEXTS in pass B); the model's own text is added to them, so the readout's n is
    n + 1 when that text is non-empty. The null arm's KL is zero by definition and is not
    computed."""
    code = "s6"
    check_args(code, variant=variant, variant_aware=False, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="open", allowed=S6_MODES)

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if mode == "generated":
        return out(missing="a generated-tokens-only policy steers nothing in a teacher-forced "
                           "text; s6 has no steered pass to compare")
    size = n if n is not None else ctx.cfg["SCREEN_B_S6_TEXTS" if pass_ == "B" else "SCREEN_A_S6_TEXTS"]
    chosen = take(code, NEUTRAL_TEXTS, size, what="neutral texts")
    max_tok = int(ctx.cfg["S6_TOKENS"])
    specs = [prompts.neutral_text_for_s6(ctx.backend, t, max_tokens=max_tok) for t in chosen]
    units = [t.id for t in chosen]
    excluded: dict[str, str] = {}
    own = own_continuation(ctx, code, ctx.backend.render([{"role": "user", "content": S6_REQUEST}]),
                           max_tok, ctx.seed(code, S6_OWN_ID), unit=S6_OWN_ID, greedy=False)
    if not own.text:
        excluded[S6_OWN_ID] = "the model's own unsteered reply is empty; nothing to score"
    else:
        try:
            specs.append(prompts.own_text_for_s6(ctx.backend, S6_REQUEST, own.text,
                                                 max_tokens=max_tok, prompt_id=S6_OWN_ID))
            units.append(S6_OWN_ID)
        except ValueError as exc:  # an unreadable output is excluded with its reason, not scored
            excluded[S6_OWN_ID] = f"the model's own reply cannot be scored: {str(exc)[:120]}"
    if cell is None:
        return out(value=0.0, n=len(specs), extra={
            "null_arm": "KL(unsteered || unsteered) is zero by definition; nothing was computed",
            "units": units, "own_transcript_id": own.transcript_id,
            "excluded": len(excluded), "excluded_units": excluded})
    concept = ctx.concept_of(cell.vector_id)
    toks = concept_tokens(ctx.backend, concept) if concept is not None else None
    exclude = list(toks.ids) if toks is not None else []
    ivs = [intervention_for(ctx, cell, s, mode) for s in specs]
    ctx.charge(forward=2 * len(specs))
    res = ctx.backend.kl_from_unsteered([s.text for s in specs], ivs,
                                        start_tokens=[int(s.meta["start_token"]) for s in specs],
                                        exclude_token_ids=exclude)
    per = []
    for unit, s, r in zip(units, specs, res):
        if not math.isfinite(r.mean_kl) or (exclude and not math.isfinite(r.mean_kl_without)):
            excluded[unit] = "a non-finite KL"
            continue
        per.append({"text_id": unit, "source": s.meta["source"], "kl": float(r.mean_kl),
                    "kl_without": None if not exclude else float(r.mean_kl_without),
                    "positions": int(r.n_positions)})
    if not per:
        return out(missing="no text gave a finite KL", extra={"excluded": len(excluded),
                                                             "excluded_units": excluded})
    value, se = mean_se([r["kl"] for r in per])
    neutral = [r["kl"] for r in per if r["source"] == "neutral"]
    own_kl = [r["kl"] for r in per if r["source"] == "own"]
    extra: dict[str, Any] = {
        "se": se, "kl_neutral": mean_se(neutral)[0] if neutral else None,
        "kl_own": own_kl[0] if own_kl else None, "own_transcript_id": own.transcript_id,
        "per_text": per, "excluded": len(excluded), "excluded_units": excluded}
    if exclude:
        without, without_se = mean_se([r["kl_without"] for r in per])
        extra.update(kl_without=without, kl_without_se=without_se, concept=concept,
                     token_ids=exclude, bleed_share=(1.0 - without / value) if value > 0 else None)
        if not value > 0:
            extra.update(bleed_missing="the KL is zero: no mass moved, so there is none to apportion")
    else:
        reason = (f"vector {cell.vector_id!r} has no concept word" if concept is None else
                  f"no variant of {concept!r} has a usable first token")
        extra.update(kl_without=None, bleed_share=None, bleed_missing=reason)
    return out(value=value, n=len(per), extra=extra)


# =====================================================================================
# s7
# =====================================================================================

# The first words of a refusal. A definition of the proxy, not a tunable; "I" also opens many
# compliant replies, which is why s7 is read as a shift against its own unsteered value and never
# as a refusal rate.
REFUSAL_OPENERS: tuple[str, ...] = ("I", "Sorry", "As", "Unfortunately", "Cannot")


def load_s7_prompts(path: Any) -> tuple[list[tuple[str, str]], str]:
    """(prompt id, text) pairs from the JSONL file at `path` -- one {"id": ..., "text": ...} object
    per line -- and the file's sha256. The file must lie outside the repository (it holds
    refusal probes, which this public repository does not carry); an empty path is refused."""
    if path is None or not str(path).strip():
        raise Refused("s7 needs S7_PROMPTS_PATH: a JSONL file of refusal-probe prompts outside the "
                      "repository. None is shipped here (decision 8, docs/M5-PLAN.md §13).")
    p = assert_outside_repo(path, what="s7 prompts file")
    data = Path(p).read_bytes()
    rows: list[tuple[str, str]] = []
    for i, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"{p.name} line {i}: not JSON") from None
        if not isinstance(obj, dict) or set(obj) != {"id", "text"} \
                or not isinstance(obj["id"], str) or not isinstance(obj["text"], str) \
                or not obj["id"].strip() or not obj["text"].strip():
            raise ValueError(f"{p.name} line {i}: expected {{\"id\": text, \"text\": text}}, both "
                             "non-empty, and nothing else")
        rows.append((obj["id"], obj["text"]))
    if not rows:
        raise ValueError(f"{p.name} holds no prompts")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError(f"{p.name}: prompt ids repeat")
    return rows, hashlib.sha256(data).hexdigest()


def read_s7(ctx: Any, cell: Cell | None, *, n: int | None = None, variant: str = "original",
            mode: str | None = None, pass_: str | None = None) -> Readout:
    """s7 (refusal-opener mass of the reply's first token, steered; the unsteered mass beside it).

    Refuses unless S7_ENABLED. n is the number of prompts from the file (all by default)."""
    code = "s7"
    if not ctx.cfg["S7_ENABLED"]:
        raise Refused("s7 is off (S7_ENABLED=False): the refusal-token proxy reads refusal probes "
                      "and is decision 8 in docs/M5-PLAN.md §13")
    check_args(code, variant=variant, variant_aware=False, pass_=pass_)
    mode = resolve_mode(code, cell, mode, kind="open", allowed=S7_MODES)
    rows, sha = load_s7_prompts(ctx.cfg["S7_PROMPTS_PATH"])

    def out(**kw: Any) -> Readout:
        return emit(ctx, code, cell, variant=variant, mode=mode, pass_=pass_, **kw)

    if mode == "generated":
        return out(missing=GENERATED_ONLY)
    chosen = take(code, rows, n, what="s7 prompts")
    specs = []
    for pid, text in chosen:
        rendered = ctx.backend.render([{"role": "user", "content": text}], add_generation_prompt=True)
        specs.append(PromptSpec(rendered, "s7", None, None,
                                {"builder": "s7", "prompt_id": pid, "mode": "all",
                                 "n_tokens": len(ctx.backend.tokenize(rendered))}))
    openers = tuple(dict.fromkeys(int(v[0]) for w in REFUSAL_OPENERS
                                  for v in ctx.backend.word_variants(w)))
    texts = [s.text for s in specs]
    ivs = [None if cell is None else intervention_for(ctx, cell, s, mode) for s in specs]
    mass_s = np.exp(logsumexp_rows(next_logprobs(ctx, texts, ivs, openers)))
    mass_u = np.exp(logsumexp_rows(next_logprobs(ctx, texts, [None] * len(texts), openers)))
    per = [{"prompt_id": pid, "mass": float(ms), "null": float(mu)}
           for (pid, _), ms, mu in zip(chosen, mass_s, mass_u)]
    value, se = mean_se(mass_s)
    null, null_se = mean_se(mass_u)
    return out(value=value, n=len(per), extra={
        "se": se, "null": null, "null_se": null_se, "shift": value - null,
        "openers": list(REFUSAL_OPENERS), "token_ids": list(openers), "file_sha256": sha,
        "per_prompt": per})
