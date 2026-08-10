# =====================================================================================
# D3 - forced identification read from logits instead of generation + judge
# =====================================================================================
# D2 asks: with the noticing prefilled, does the model NAME the injected concept? It costs
# ~23 s/cell because it generates 25 completions and sends all of them to a judge.
#
# D3 asks the same question of the same prompt, but reads the answer off the next-token
# distribution instead of sampling and judging it. One forward pass per trial number, no
# generation, no judge: ~0.3 s/cell. That is what makes a full-depth layer scan affordable.
#
# It is a PROXY and is not interchangeable with D2. It is built to be faithful in the three
# ways that matter:
#   - identical prompt construction to measure_D2, including FORCED_PREFILL
#   - identical steering start position, so the framing stays unsteered
#   - scored against the same concept-word surface forms the judge would accept
#
# Where it can still diverge from D2, by construction:
#   - D2's judge accepts "closely related" concepts; token mass only counts the word itself
#     and the variants listed below
#   - the model may open with a filler token ("the", "a", "something") and name the concept
#     one position later - handled by ALLOW_FILLER below
#   - a model too damaged to produce anything will read low here AND low on D2, which is a
#     real confound for both and is why the forced-ID capability control exists
#
# None of that is a reason to skip validation. Run validate_d3() before any pipeline
# trusts this. If Spearman rho against real D2 is below ~0.7, the layer scan loses its
# detection axis and the shortlist must widen to select on effectiveness alone.
#
# Depends on the measurement lab having run Setup 7 (chat/encode/injected/mean_se) and
# prepare_concept() (VECS, CONFIG). Paste as a cell after measure_D2, or import.
# =====================================================================================

import torch, json, math, statistics

# Read the concept at the position right after "...The thought is about". Allowing one
# filler token catches "The thought is about the velocity of..." without opening the window
# so wide that any later mention counts - at that point it stops being identification and
# starts being the concept leaking into fluent text, which is effectiveness, not detection.
ALLOW_FILLER = True
D3_TRIALS = (1, 7, 13, 19, 25)   # a few trial numbers; the prompt differs only here


def _first_token_ids(strings):
    """Token ids that could START the concept, across the surface forms the judge accepts.

    The leading space matters: after "about", Gemma tokenizes " velocity" as one token and
    "velocity" as another. Scoring only the bare form silently reads near-zero everywhere,
    which would look exactly like a covert cell.
    """
    ids = set()
    for s in strings:
        toks = tok(s, add_special_tokens=False)["input_ids"]
        if toks:
            ids.add(int(toks[0]))
    return sorted(ids)


def concept_surface_ids(concept):
    """First-token ids for the concept in the post-prefill position."""
    c = concept.strip()
    forms = [f" {c}", f" {c.lower()}", f" {c.capitalize()}", f" {c.upper()}",
             c, c.lower(), c.capitalize()]
    # plural / possessive openings share the first token with the singular, so they are
    # already covered by the forms above; no need to enumerate them.
    return _first_token_ids(forms)


def _d3_logits(layer, alpha, trial_numbers=D3_TRIALS):
    """Next-token distributions at the forced-ID position, one per trial number.

    Reuses _forced_prompts so the prompt, the prefill and the steering start position are
    byte-identical to measure_D2's. If that function changes, this follows automatically -
    which is the point of calling it rather than rebuilding the prompt here.
    """
    prompts, start = _forced_prompts(list(trial_numbers))
    out = []
    for p in prompts:
        enc = encode(p)
        with injected(VECS[layer] if alpha else None, layer, alpha, start_pos=start):
            with torch.no_grad():
                logits = hf(**enc).logits[0, -1, :].float()
        out.append((enc, torch.softmax(logits, dim=-1)))
    return out


def measure_D3(layer, alpha, verbose=False):
    """Concept probability mass at the forced-identification position.

    Returns the steered mass, the unsteered mass on the same prompts, and the rank of the
    best concept token. Rank is reported because mass alone cannot distinguish "the concept
    is second choice behind a filler" from "the concept is nowhere".
    """
    cids = concept_surface_ids(CONFIG["concept"])
    if not cids:
        raise ValueError(f"no concept token ids for {CONFIG['concept']!r}")

    per = []
    steered = _d3_logits(layer, alpha)
    base    = _d3_logits(layer, 0.0)

    for (enc_s, p_s), (_, p_b) in zip(steered, base):
        mass_s = float(p_s[cids].sum())
        mass_b = float(p_b[cids].sum())
        rank   = int((p_s > float(p_s[cids].max())).sum()) + 1

        filler_mass = None
        if ALLOW_FILLER:
            # Greedy-extend one token, then re-read the concept mass. Cheap (one extra
            # forward pass) and it recovers the "about THE velocity" case.
            nxt = int(torch.argmax(p_s))
            if nxt not in cids:
                ids2 = torch.cat([enc_s["input_ids"],
                                  torch.tensor([[nxt]], device=enc_s["input_ids"].device)], dim=1)
                am2 = torch.cat([enc_s["attention_mask"],
                                 torch.ones((1, 1), dtype=enc_s["attention_mask"].dtype,
                                            device=enc_s["attention_mask"].device)], dim=1)
                prompts, start = _forced_prompts(list(D3_TRIALS))
                with injected(VECS[layer] if alpha else None, layer, alpha, start_pos=start):
                    with torch.no_grad():
                        l2 = hf(input_ids=ids2, attention_mask=am2).logits[0, -1, :].float()
                filler_mass = float(torch.softmax(l2, dim=-1)[cids].sum())

        per.append(dict(mass=mass_s, base_mass=mass_b, rank=rank,
                        filler_mass=filler_mass,
                        best=max(mass_s, filler_mass if filler_mass is not None else 0.0)))

    m,  se,  n = mean_se([r["best"] for r in per])
    bm, bse, _ = mean_se([r["base_mass"] for r in per])
    ranks = [r["rank"] for r in per]

    row = dict(
        d3            = m,
        d3_se         = se,
        d3_n          = n,
        d3_base       = bm,
        d3_lift       = (m - bm),
        # A rate-shaped companion so it can be read on the same axis as D2, which is a
        # fraction of trials. Threshold is arbitrary and must be tuned in validation.
        d3_rate       = sum(1 for r in per if r["best"] > 0.10) / len(per),
        d3_rank_med   = statistics.median(ranks),
        d3_rank_best  = min(ranks),
        d3_per_trial  = per,
    )
    if verbose:
        print(f"  concept token ids : {cids} -> {[tok.decode([i]) for i in cids]}")
        print(f"  {'trial':>6} {'mass':>10} {'+filler':>10} {'base':>10} {'rank':>7}")
        for t, r in zip(D3_TRIALS, per):
            fm = f"{r['filler_mass']:.4f}" if r["filler_mass"] is not None else "-"
            print(f"  {t:>6} {r['mass']:>10.4f} {fm:>10} {r['base_mass']:>10.4f} {r['rank']:>7}")
        print(f"\n  D3 mass  : {m:.4f}" + (f" +/- {se:.4f}" if se else ""))
        print(f"  unsteered     : {bm:.4f}")
        print(f"  rate (>0.10)  : {row['d3_rate']:.2f}")
        print(f"  rank median   : {row['d3_rank_med']}  (best {row['d3_rank_best']})")
    return row


# =====================================================================================
# VALIDATION - run this before any pipeline uses D3
# =====================================================================================

def _spearman(xs, ys):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v)
        i = 0
        while i < len(order):                      # average ties, or rho is wrong wherever
            j = i                                   # D2 saturates at 0.00 / 1.00 - which is
            while j+1 < len(order) and v[order[j+1]] == v[order[i]]:   # most of the grid
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j + 1
        return r
    a, b = rk(xs), rk(ys)
    n = len(xs)
    ma, mb = sum(a)/n, sum(b)/n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    den = math.sqrt(sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b))
    return num/den if den else 0.0


def validate_d3(summary=None, cells=None, min_rho=0.70, verbose=True):
    """Correlate D3 against real D2 on cells that already have a D2 number.

    Pass summary=SUMMARY from a completed run. Only cells with a real D2 are used, and
    unusable cells are kept deliberately: a proxy that works only where the model is
    healthy is not a proxy for a scan that has to cross the damaged region.
    """
    summary = SUMMARY if summary is None else summary
    rows = [r for r in summary if r.get("d2") is not None]
    if cells:
        rows = [r for r in rows if (r["layer"], r["alpha"]) in set(cells)]
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} cells with D2 - need >=10, ideally >=60")

    got = []
    for r in rows:
        lite = measure_D3(r["layer"], r["alpha"])
        got.append(dict(layer=r["layer"], alpha=r["alpha"], d2=r["d2"],
                        usable=r.get("usable"),
                        lite_mass=lite["d3"], lite_rate=lite["d3_rate"],
                        lite_rank=lite["d3_rank_med"]))

    d2   = [g["d2"] for g in got]
    variants = {
        "mass":        [g["lite_mass"] for g in got],
        "rate(>0.10)": [g["lite_rate"] for g in got],
        "rank(neg)":   [-g["lite_rank"] for g in got],
    }
    rhos = {k: _spearman(v, d2) for k, v in variants.items()}
    best = max(rhos, key=rhos.get)

    if verbose:
        print("="*70)
        print(f"D3 validation - {len(got)} cells, concept {CONFIG['concept']}")
        print("="*70)
        print(f"  {'L':>4} {'alpha':>6} {'D2':>6} {'lite_mass':>10} {'lite_rate':>10} {'rank':>6} {'use':>4}")
        for g in sorted(got, key=lambda g: (g["layer"], g["alpha"])):
            print(f"  {g['layer']:>4} {g['alpha']:>6} {g['d2']:>6.2f} {g['lite_mass']:>10.4f} "
                  f"{g['lite_rate']:>10.2f} {g['lite_rank']:>6} {'y' if g['usable'] else 'n':>4}")
        print("\n  Spearman rho vs real D2:")
        for k, v in sorted(rhos.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<14} {v:>6.3f}" + ("   <== best" if k == best else ""))
        print("")
        if rhos[best] >= min_rho:
            print(f"  PASS - '{best}' at rho {rhos[best]:.3f} >= {min_rho}. Usable as the scan's")
            print("  detection axis. Still verify shortlisted cells with real D2.")
        else:
            print(f"  FAIL - best rho {rhos[best]:.3f} < {min_rho}. Do NOT use D3 to")
            print("  prune layers. Scan on effectiveness alone and carry more candidates")
            print("  into verification.")
        print("="*70)

    return dict(n=len(got), rhos=rhos, best=best, passed=rhos[best] >= min_rho, rows=got)


print("D3 ready:  measure_D3(layer, alpha, verbose=True)")
print("                validate_d3(SUMMARY)   <- run this first")
