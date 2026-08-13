# 18 — The judge cache key is not float-normalised

**Status: DONE — `d82c03a`.** **Blocking defect fixed.** It fired twice in the 2026-08-12
shakedown, in two independent call sites, and prevented that run from entering Phase 4.

## The symptom

```
RuntimeError: judge_many returned results out of order: item
  ('PHASE4', 58, 1.3499999999999999, 'e5_01@9059d04ed4', 'E5', 'b03001ad639a') came back as
  ('PHASE4', 58, 1.35,               'e5_01@9059d04ed4', 'E5', 'b03001ad639a').
Every score would be attached to the wrong response.
```

And, independently, inside gate 4:

```
  ('GATE4', 37, 0.40312499999999996, 'e5_01@2197e6c93d', 'S1', '21057bbe301e') came back as
  ('GATE4', 37, 0.403125,            'e5_01@2197e6c93d', 'S1', '21057bbe301e').
```

## What is actually wrong

**The results are not out of order.** The guard is correct and must stay — attaching scores to the
wrong response is exactly the silent catastrophe it exists to prevent. What is wrong is that the
key carries a **raw float `r`**, and one side of the round trip normalises it while the other does
not. Both differences vanish at six decimal places.

`docs/CONTRACT.md` already mandates the convention: `R_DECIMALS`, *"Rounding applied to float
components of a resume key. MUST match `vectors.R_DECIMALS`"*. The judge cache key is not applying
it consistently.

**Why bisection triggers it and nothing else did.** Phase 3 computes `0.5 * (lo + hi)` repeatedly,
which produces values like `1.3499999999999999` and `0.40312499999999996` — non-terminating in
binary, so `repr` and any rounded form differ. Every `r` before Phase 3 comes from `SCAN_DOSES`,
which are literals that survive a round trip unchanged. **The bug could not exist before a run
reached bisection**, which is why the offline tests and every earlier run missed it.

## The fix

**Normalise `r` once, at the point it enters any key, using the existing `R_DECIMALS` convention.**
Not at the comparison — at construction, so both sides are built from the same value.

1. Find every place a cache key is built from a float and route them all through one helper.
   `judges.cache_key_for` is the obvious chokepoint; check `expensive._item`, `_issue`, and gate 4's
   `_s1_over_stored` are all going through it.
2. **Keep the order guard.** Do not relax it to compare approximately — an approximate identity
   check on a key is how two genuinely different cells get treated as the same one. Normalise the
   input, keep the comparison exact.
3. Make the guard's message print both keys with `repr` on the float, so the next instance of this
   is diagnosable in one line rather than requiring a numeric squint.

## Acceptance

- A test that trips it: build a key from `0.5 * (0.9140625 + 0.9421875)` and one from the same
  value round-tripped through JSON, and assert they are equal. Before the fix that test fails.
- A test asserting the order guard still raises when the items genuinely are out of order.
- Re-running the bisected values from the shakedown (`1.3499999999999999`, `0.40312499999999996`,
  `0.9281249999999999`, `1.0828125000000002`) through key construction yields stable keys.

## For the record

Append to `../DEBUG-LOG.md` once fixed. The *why nothing caught it* field: every `r` upstream of
Phase 3 is a config literal, so no test or earlier run ever put a bisected float into a cache key —
the pipeline had to reach Phase 4 for the defect to become reachable at all.
