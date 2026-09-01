"""
Result loader for the kwdagger form of the mini-sweep card.

New file. Nothing else in this repository imports it, and
``cards/evaluation.yaml`` is unaffected.

Why this exists rather than the generic loader. ``mini_sweep_runner.py`` writes
its flat per-verifier scalars alongside two nested structures::

    {"result": {"abcrown_correct_fraction": 0.98, ..., "total_instances": 50,
                "per_verifier": {...},        # dicts, and per_instance lists
                "summary": {"verifiers_run": [...], ...}}}

The generic loader takes a single dotted path, so it can only lift *all* of
``result`` -- which drags list- and dict-valued entries into the evidence row.
Those are not metrics: a BAA metric is a number with an objective and a
reduction, and a per-instance breakdown is neither.

This selects the scalars and drops the rest. ``None`` is preserved, because a
skipped verifier reports ``correct_fraction: None`` and the card's claim
distinguishes "did not run" (INCONCLUSIVE) from "ran and scored low"
(FALSIFIED).
"""
import json

import ubelt as ub


#: bool is a subclass of int; listed for clarity, not because it is needed.
_SCALAR_TYPES = (bool, int, float, str)


def load_kwdagger_result(node, node_dpath):
    """
    Load the mini-sweep artifact for ``kwdagger aggregate``.

    Args:
        node: the configured node being loaded.
        node_dpath: directory holding this node's outputs.

    Returns:
        A flat DotDict of ``metrics.<node>.<name>`` entries.
    """
    from kwdagger.utils import util_dotdict

    node_dpath = ub.Path(node_dpath)
    output_fpath = node_dpath / node.out_paths[node.primary_out_key]
    payload = json.loads(output_fpath.read_text())

    # GenericPipelineProcessor wraps the runner's dict in "result"; tolerate
    # either shape so this keeps working if that envelope changes.
    result = payload.get('result', payload)

    metrics = {
        key: value
        for key, value in result.items()
        if value is None or isinstance(value, _SCALAR_TYPES)
    }

    flat = util_dotdict.DotDict.from_nested({'metrics': metrics})
    return flat.insert_prefix(node.name, index=1)
