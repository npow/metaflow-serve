"""Metaflow plugin registration entry point.

The ServiceSpec + Deployment pattern does not use StepDecorators,
so no decorators are registered. This file is kept for the cards
submodule promotion.
"""

STEP_DECORATORS_DESC: list[tuple[str, str]] = []

__mf_promote_submodules__: list[str] = []
