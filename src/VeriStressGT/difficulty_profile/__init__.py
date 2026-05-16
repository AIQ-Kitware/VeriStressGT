# src/VeriStressGT/difficulty_profile/__init__.py
"""
Difficulty Profile estimation for neural network verification instances.

This package estimates architecture-agnostic difficulty components for
runtime / timeout analysis.  The current profile avoids CROWN-style estimates
and is based on generic margin, gradient, nonlinearity, effective-dimension,
and IBP quantities.

Canonical components are registered in ``profile.COMPONENT_HYPOTHESES``.
"""

from .profile import (
    CANONICAL_COMPONENT_NAMES,
    COMPONENT_HYPOTHESES,
    ComponentHypothesis,
    DifficultyProfile,
    estimate_profile,
)

from .components import (
    estimate_generic_components,
    estimate_representation_stats,
    estimate_local_region_count,
    estimate_clarke_lipschitz)

__all__ = [
    "DifficultyProfile",
    "ComponentHypothesis",
    "COMPONENT_HYPOTHESES",
    "CANONICAL_COMPONENT_NAMES",
    "estimate_profile",
    "estimate_generic_components",
    "estimate_representation_stats",
    "estimate_local_region_count",
    # Backward-compatible exported aliases:
    "estimate_clarke_lipschitz"
]