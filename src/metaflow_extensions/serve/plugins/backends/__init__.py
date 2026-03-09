"""Backend registry for serving backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import ServingBackend

_REGISTRY: dict[str, type[ServingBackend]] = {}


def register(name: str, cls: type[ServingBackend]) -> None:
    """Register a serving backend by name."""
    _REGISTRY[name] = cls


def get_backend(name: str) -> ServingBackend:
    """Instantiate and return a registered backend by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown serving backend {name!r}. Available: {available}")
    return _REGISTRY[name]()


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    return sorted(_REGISTRY)


# Auto-register built-in backends.
try:
    from .huggingface import HuggingFaceBackend

    register("huggingface", HuggingFaceBackend)
except ImportError:
    pass
