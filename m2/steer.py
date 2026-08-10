"""m2.steer - use an operating point once the pipeline has found one.

The pipeline's output is a set of numbers. This module is what turns those numbers back into a
steered model, in one line, from anywhere:

    from m2 import steer
    steer.use("Irony")                                  # load the winner this run found
    print(steer.ask("Tell me a short story."))          # steered
    steer.compare("Tell me a fact related to water.")   # steered vs unsteered, side by side
    steer.session()                                     # interactive prompt loop

and, for a downstream experiment that only needs the steering:

    with steer.steering("Irony", layer=37, r=0.20):
        ...                                             # anything you generate here is steered

Nothing here measures anything. It performs no judging, writes no rows, and has no gates - it
exists so that the screened parameters can be reused without importing the measurement stack or
re-deriving what `alpha` corresponded to `r` at some layer.

`layer` and `r` are the parameters; `alpha` is derived, because at fixed alpha the real dose
varies more than 20x across layers (spec section 3). Passing `alpha=` directly is allowed for
reproducing a v1 or Macar cell, and says so in the returned record.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import config
from . import model
from . import prompts as _prompts
from . import vectors


# The currently loaded operating point, so `ask` and `compare` need no arguments after `use`.
CURRENT: dict | None = None


def _run() -> Any:
    ctx = getattr(config, "RUN", None)
    if ctx is None:
        raise RuntimeError(
            "m2.config.RUN is not set - call m2.model.load_model(m2.config.CONFIG) first.")
    return ctx


def _cfg() -> dict:
    ctx = getattr(config, "RUN", None)
    if ctx is not None and ctx.config:
        return ctx.config
    return config.CONFIG


# =====================================================================================
# loading an operating point
# =====================================================================================

def load_operating_point(source: str | Path | dict) -> dict:
    """The winner from an `operating_point.json`, a run folder, or a dict you paste in.

    Accepts, in order of convenience:

        load_operating_point("/workspace/m2_runs/irony_abc123")      # a run folder
        load_operating_point(".../operating_point.json")             # the file itself
        load_operating_point({"concept": "Irony", "layer": 37, "r": 0.20})

    The dict form is the one to use in a paper appendix or a downstream repo: four numbers,
    no dependency on this run's folder surviving.
    """
    if isinstance(source, dict):
        record = dict(source)
        record.setdefault("origin", "literal")
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "operating_point.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Point this at a run folder, at its "
                f"operating_point.json, or pass a dict with concept/layer/r.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        winner = payload.get("winner")
        if not winner:
            raise ValueError(
                f"{path} records no winner: {payload.get('reason', 'no reason given')}. "
                "No cell qualified, so there is no operating point to use.")
        record = dict(winner)
        record.setdefault("concept", payload.get("concept"))
        record["origin"] = str(path)

    for key in ("concept", "layer"):
        if record.get(key) in (None, ""):
            raise KeyError(f"operating point is missing {key!r}: {record}")
    if record.get("r") is None and record.get("alpha") is None:
        raise KeyError("operating point needs either r (preferred) or alpha")
    record["layer"] = int(record["layer"])
    return record


def use(source: str | Path | dict | None = None, *, concept: str | None = None,
        layer: int | None = None, r: float | None = None,
        alpha: float | None = None) -> dict:
    """Load an operating point and make it the default for `ask` / `compare` / `session`.

    Either pass a source, or give the parameters directly:

        steer.use("/workspace/m2_runs/irony_abc123")
        steer.use(concept="Irony", layer=37, r=0.20)
        steer.use(concept="Irony", layer=37, alpha=2.0)   # a v1 or Macar cell, verbatim
    """
    global CURRENT
    if source is not None:
        record = load_operating_point(source)
    else:
        if concept is None or layer is None:
            raise ValueError("give a source, or concept= and layer= with r= or alpha=")
        record = dict(concept=concept, layer=int(layer), r=r, alpha=alpha, origin="literal")
    CURRENT = _prepare(record)
    _describe(CURRENT)
    return CURRENT


def _prepare(record: dict) -> dict:
    """Extract the vector, resolve alpha from r (or the reverse), and cache it on the record."""
    ctx = _run()
    concept = str(record["concept"])
    layer = int(record["layer"])

    if ctx.concept != concept or layer not in ctx.vecs:
        # Extraction is per layer and injection is at that same layer - the same call the
        # pipeline used. Vectors regenerate in seconds; nothing is loaded from an archive.
        ctx.vecs.update(vectors.extract_all_layers(concept, [layer]))
        ctx.concept = concept

    vec = ctx.vecs[layer]
    vec_norm = float(vec.float().norm())

    if record.get("r") is not None:
        # Needs a residual norm to convert. If the run measured one, use it; otherwise
        # measure it now on the same prompt set, so the dose means what it meant in the run.
        if layer not in ctx.norms:
            resid = model.residual_norms(
                model.chat(_prompts.E5_PROMPTS[0]["text"]), [layer])
            ctx.norms[layer] = dict(vec_norm=vec_norm, resid_norm=float(resid[layer]))
        alpha = vectors.alpha_from_norms(float(record["r"]),
                                         float(ctx.norms[layer]["vec_norm"]),
                                         float(ctx.norms[layer]["resid_norm"]))
        dose = float(record["r"])
        basis = "r"
    else:
        alpha = float(record["alpha"])
        dose = None
        if layer in ctx.norms:
            dose = vectors.dose_from_alpha(alpha, float(ctx.norms[layer]["vec_norm"]),
                                           float(ctx.norms[layer]["resid_norm"]))
        basis = "alpha"

    out = dict(record)
    out.update(layer=layer, concept=concept, alpha=float(alpha), r=dose,
               vec_norm=vec_norm, basis=basis,
               vec_fingerprint=vectors.vec_fingerprint(vec))
    return out


def _describe(record: dict) -> None:
    print(f"steering ready : {record['concept']}  L{record['layer']}")
    if record.get("r") is not None:
        print(f"  dose r       : {record['r']:.4f}")
    print(f"  alpha        : {record['alpha']:.4f}"
          + ("   (given directly, not derived from r)" if record["basis"] == "alpha" else
             "   (derived from r - the parameter that is comparable across layers)"))
    print(f"  ||v||        : {record['vec_norm']:.0f}   fingerprint {record['vec_fingerprint']}")
    for key, label in (("e5", "E5 influence"), ("d2", "D2 forced ID"), ("s4", "S4 sanity")):
        if record.get(key) is not None:
            print(f"  {label:<13}: {float(record[key]):.2f}")


def current() -> dict:
    if CURRENT is None:
        raise RuntimeError("no operating point loaded - call m2.steer.use(...) first")
    return CURRENT


# =====================================================================================
# the context manager - for downstream experiments
# =====================================================================================

@contextmanager
def steering(source: str | Path | dict | None = None, *, concept: str | None = None,
             layer: int | None = None, r: float | None = None, alpha: float | None = None,
             question: str | None = None) -> Iterator[dict]:
    """Steer for the duration of the block. Anything generated inside is steered.

        with steer.steering(concept="Irony", layer=37, r=0.20):
            out = my_experiment(model)

    `question` sets the steering start position so the chat template stays unsteered, matching
    how every measurement in the pipeline injected. Leave it None to steer all positions, which
    is right for raw text with no template.
    """
    if source is None and concept is None and CURRENT is not None:
        record = CURRENT
    elif source is not None:
        record = _prepare(load_operating_point(source))
    else:
        record = _prepare(dict(concept=concept, layer=layer, r=r, alpha=alpha,
                               origin="literal"))

    ctx = _run()
    vec = ctx.vecs[record["layer"]]
    start = None
    if question is not None:
        start = model.start_pos_for(model.chat(question), question)

    with model.injected(vec, record["layer"], record["alpha"], start_pos=start):
        yield record


# =====================================================================================
# manual testing
# =====================================================================================

def ask(question: str, *, max_new_tokens: int | None = None,
        temperature: float | None = None, steered: bool = True, **kw: Any) -> str:
    """One answer to one question, steered (or not) at the loaded operating point."""
    record = _prepare(_merge(kw)) if kw else current()
    ctx = _run()
    cfg = _cfg()
    max_new_tokens = int(max_new_tokens if max_new_tokens is not None else cfg["MAX_NEW_TOKENS"])
    temperature = float(temperature if temperature is not None else cfg["TEMPERATURE"])

    prompt = model.chat(question)
    start = model.start_pos_for(prompt, question)
    vec = ctx.vecs[record["layer"]] if steered else None

    # One prompt, so there is no padding and no per-row start position to get wrong (bug 25b
    # is a batching failure). Straight through Macar's single-stream path.
    return ctx.mw.generate_with_steering(
        prompt=prompt,
        layer_idx=int(record["layer"]),
        steering_vector=vec,
        strength=float(record["alpha"]) if steered else 0.0,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        steering_start_pos=start,
    )


def _merge(kw: dict) -> dict:
    base = dict(CURRENT or {})
    base.update({k: v for k, v in kw.items() if k in ("concept", "layer", "r", "alpha")})
    return base


def compare(question: str, *, max_new_tokens: int | None = None,
            temperature: float | None = None, n: int = 1, **kw: Any) -> dict:
    """Steered against unsteered on the same question, printed side by side.

    This is the paired view every judged measurement in the pipeline uses, and the reason is
    worth repeating here: the model has strong native habits - nearly every unsteered story in
    the v1 probes was the same lighthouse keeper - so a steered sample read on its own cannot
    be told apart from the model's usual output.
    """
    record = _prepare(_merge(kw)) if kw else current()
    unsteered = [ask(question, max_new_tokens=max_new_tokens, temperature=temperature,
                     steered=False) for _ in range(n)]
    steered = [ask(question, max_new_tokens=max_new_tokens, temperature=temperature,
                   steered=True) for _ in range(n)]

    print("=" * 78)
    print(f"{question}")
    print(f"{record['concept']}  L{record['layer']}  "
          + (f"r={record['r']:.3f}  " if record.get("r") is not None else "")
          + f"alpha={record['alpha']:.2f}")
    print("=" * 78)
    for i, (u, s) in enumerate(zip(unsteered, steered)):
        if n > 1:
            print(f"--- sample {i + 1} ---")
        print("  UNSTEERED:")
        print("   ", u.replace("\n", "\n    ")[:700])
        print("  STEERED:")
        print("   ", s.replace("\n", "\n    ")[:700])
        print("")
    return dict(question=question, unsteered=unsteered, steered=steered, **record)


def probe(questions: Sequence[str] | None = None, **kw: Any) -> list[dict]:
    """`compare` over a list of questions. Defaults to the three v1 behaviour probes."""
    questions = list(questions) if questions is not None else [
        "Tell me the first 10 words that come to mind.",
        "Tell me a short story.",
        "Tell me a fact related to water.",
    ]
    return [compare(q, **kw) for q in questions]


def session(**kw: Any) -> None:
    """Interactive loop: type a question, see steered against unsteered. Blank line to stop.

    The manual counterpart to the whole pipeline. Every number it produces is an aggregate,
    and an aggregate cannot tell you what the steering actually feels like at the operating
    point - every diagnosis in the v1 review needed someone to read a generation.
    """
    record = current()
    print(f"{record['concept']} at L{record['layer']} alpha={record['alpha']:.2f}. "
          "Blank line or Ctrl-C to stop.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nstopped")
            return
        if not question:
            print("stopped")
            return
        try:
            compare(question, **kw)
        except Exception as exc:                     # noqa: BLE001 - a session must survive
            print(f"  [{type(exc).__name__}] {exc}")


def sweep(question: str, doses: Sequence[float], **kw: Any) -> list[dict]:
    """The same question at several doses, to see where influence starts and sanity ends.

    The fastest way to sanity-check an operating point by eye: if the reported r sits in the
    middle of the range where the answer is on-concept and still coherent, the screen agrees
    with what you can see.
    """
    record = dict(CURRENT or {})
    record.update({k: v for k, v in kw.items() if k in ("concept", "layer")})
    out = []
    for dose in doses:
        prepared = _prepare(dict(record, r=float(dose), alpha=None))
        text = ask(question, **{k: prepared[k] for k in ("concept", "layer")},
                   r=float(dose))
        print(f"--- r={dose:.3f}  alpha={prepared['alpha']:.2f} ---")
        print("   ", text.replace("\n", "\n    ")[:500])
        out.append(dict(r=float(dose), alpha=prepared["alpha"], response=text))
    return out
