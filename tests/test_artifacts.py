"""Tests for Artifacts access layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from metaflow_extensions.serve.plugins.artifacts import Artifacts
from metaflow_extensions.serve.plugins.artifacts import _StepArtifacts


class TestArtifacts:
    def test_empty_artifacts(self):
        a = Artifacts()
        with pytest.raises(AttributeError, match="No artifact source"):
            a.flow  # noqa: B018

    def test_from_step(self):
        step = MagicMock()
        a = Artifacts.from_step(step)
        assert isinstance(a._sources["flow"], _StepArtifacts)

    def test_from_flows(self):
        step1 = MagicMock()
        step2 = MagicMock()
        a = Artifacts.from_flows({"train": step1, "eval": step2})
        assert "train" in a._sources
        assert "eval" in a._sources

    def test_from_flows_none(self):
        a = Artifacts.from_flows(None)
        assert a._sources == {}

    def test_repr(self):
        a = Artifacts.from_step(MagicMock())
        assert "flow" in repr(a)


class TestStepArtifacts:
    def test_access_via_task_data(self):
        step = MagicMock()
        step.task.data.model = "my-model"
        sa = _StepArtifacts(step)
        assert sa.model == "my-model"

    def test_access_via_direct_attr(self):
        step = MagicMock(spec=["my_artifact"])
        step.my_artifact = 42
        # Remove task attr so it falls through to direct access
        del step.task
        sa = _StepArtifacts(step)
        assert sa.my_artifact == 42

    def test_missing_artifact_raises(self):
        step = MagicMock(spec=[])
        del step.task
        sa = _StepArtifacts(step)
        with pytest.raises(AttributeError, match="not found"):
            sa.nonexistent  # noqa: B018

    def test_repr(self):
        step = MagicMock()
        sa = _StepArtifacts(step)
        assert "_StepArtifacts" in repr(sa)


class TestArtifactsIntegration:
    def test_flow_artifact_access(self):
        step = MagicMock()
        step.task.data.model = {"weights": [1, 2, 3]}
        step.task.data.config = {"lr": 0.01}

        a = Artifacts.from_step(step)
        assert a.flow.model == {"weights": [1, 2, 3]}
        assert a.flow.config == {"lr": 0.01}
