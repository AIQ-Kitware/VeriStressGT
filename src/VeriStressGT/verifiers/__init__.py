"""Verifier integration modules.

This package holds two kinds of things:

1. Python modules that wrap individual verifier CLIs and are imported by
   VeriStressGT (e.g. abcrown_vnncomp2024.py). These require this
   __init__.py to be importable under the VeriStressGT.verifiers.*
   namespace.

2. Git submodule checkouts of the verifier repositories themselves
   (alpha-beta-CROWN_vnncomp2024/, neuralsat/, marabou/, nnenum/, nnv/,
   pyrat/). These are NOT Python subpackages — they are self-contained
   external projects. Do not add them to __all__.
"""
