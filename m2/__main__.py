"""Allow `python -m m2` as a shorthand for `python -m m2.run`."""

from __future__ import annotations

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
