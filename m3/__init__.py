"""m3 - operating-point search by measuring every cell, judged, and writing down everything.

M2 screened a grid with cheap forward-pass proxies and measured only the survivors properly.
Every proxy read a single next-token distribution, and influence, detection and coherence are all
properties of generated text, so all three measured something other than what they screened for.
The result was a correct set of numbers and a false conclusion.

M3 has no screening tier. Generation costs about 8.6s per batch regardless of batch size, and a
judge call costs $0.0007, so the whole surface can be measured properly for about $4 a concept --
the premise that made proxies necessary expired. See `docs/M3-DESIGN.md`.

Start at `m3.config`: every tunable is there, in one file, with a comment on each.
"""

from __future__ import annotations

__version__ = "0.1.0-dev"
