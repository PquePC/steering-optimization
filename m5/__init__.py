"""m5 - the steering-awareness audit instrument described in `docs/M5-PLAN.md`.

Written from scratch. **Nothing in this package imports `m2` or `m3`**, and nothing in those
packages is changed by it. The old pipelines stay runnable and their archives stay readable; the
only contact is `m5.analysis`, which *reads* the files old runs wrote (never their code).

What M5 changes, in one paragraph. M3 and M4 gate the operating window on forced naming (`d2`),
which steering alone can produce by filling an open answer slot. M5 measures three separate
things -- whether the model *notices* an intervention (`d7`, `d8`, `d9`, `d10`), whether it can
*name* it (`d2`, `d3`, with the open-slot control `d5` and the past-only arm `d6`), and whether an
injection signal is *registered* inside the model (`q1`-`q4`) -- beside effect (`e7`-`e12`),
integrity (`s1`, `s2`, `s4`-`s7`) and consequence (`a1`-`a3`). A forward-pass screen maps every
layer; generated-and-judged measurement is spent adaptively where the screen says the answer lives;
reported numbers come only from a fixed-n confirmation on held-out prompts.

Start at:

    m5.config      every tunable, one file, each with its reason; unknown keys raise
    m5.codes       the measure codes and what each one is for
    m5.safety      what this package refuses to do, in code
    m5.runio       run directories (always outside the repository) and their files
    m5.intervention  what a steering intervention *is*: vector, operation, layers, positions
    m5.backend     the one seam to a model: `Backend` (protocol), `hf` (real), `fake` (simulator)

Measure codes follow the repository convention (`AGENTS.md`): every quantity is referred to by its
code in code, logs and messages. The inventory is `m5.codes.CODES`.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Every row any M5 run writes carries this, so a reader can refuse rows from a different schema
# rather than misread them. Bump it whenever the meaning of a stored field changes.
SCHEMA = "m5.v1"
