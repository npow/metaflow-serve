"""Artifact access layer for ServiceSpec.

Provides ``artifacts.flow.<name>`` access to Metaflow step artifacts,
loading them lazily from the underlying task datastore.
"""

from __future__ import annotations

from typing import Any


class Artifacts:
    """Lightweight wrapper giving ``artifacts.flow.<name>`` access to Metaflow artifacts."""

    def __init__(self) -> None:
        self._sources: dict[str, Any] = {}

    @classmethod
    def from_step(cls, step: Any) -> Artifacts:
        """Load artifacts from a Metaflow step object.

        Parameters
        ----------
        step : metaflow.Step or flow step reference
            The step whose artifacts should be accessible.
        """
        a = cls()
        a._sources["flow"] = _StepArtifacts(step)
        return a

    @classmethod
    def from_flows(cls, flows: dict[str, Any] | None) -> Artifacts:
        """Load artifacts from a dict of named flow/step references."""
        a = cls()
        if flows:
            for name, step in flows.items():
                a._sources[name] = _StepArtifacts(step)
        return a

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        sources = self.__dict__.get("_sources", {})
        if name in sources:
            return sources[name]
        raise AttributeError(f"No artifact source '{name}'. Available: {list(sources)}")

    def __repr__(self) -> str:
        return f"Artifacts(sources={list(self._sources)})"


class _StepArtifacts:
    """Lazy accessor for artifacts from a Metaflow step."""

    def __init__(self, step: Any) -> None:
        self._step = step

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        step = self.__dict__.get("_step")
        if step is None:
            raise AttributeError(name)
        # If it's a raw Metaflow Step object, access via task data
        if hasattr(step, "task"):
            return getattr(step.task.data, name)
        # If it's a flow step reference (from inside a running flow),
        # try direct attribute access
        if hasattr(step, name):
            return getattr(step, name)
        raise AttributeError(f"Artifact '{name}' not found on step {step!r}")

    def __repr__(self) -> str:
        return f"_StepArtifacts(step={self._step!r})"
