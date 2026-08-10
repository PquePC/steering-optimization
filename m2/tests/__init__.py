"""Offline tests for the m2 package.

Everything here runs without a GPU, a model, a judge key or the Macar repo. What it protects is
the class of failure the v1 measurement lab kept producing: a plausible wrong number rather than
an error. See `../CONTRACT.md` section 6 and `../../DEBUG LOG.md` section 6.
"""
