"""Shared test fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from metaflow_extensions.serve.plugins.backends.backend import EndpointInfo
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.backend import ModelReference
from metaflow_extensions.serve.plugins.backends.backend import ServingBackend


def make_model_ref(**overrides: Any) -> ModelReference:
    defaults = {
        "flow_name": "TestFlow",
        "run_id": "1",
        "step_name": "train",
        "task_id": "1",
        "artifact_name": "model",
        "pathspec": "TestFlow/1/train/1",
        "framework": "transformers",
        "task_type": "text-generation",
    }
    defaults.update(overrides)
    return ModelReference(**defaults)


def make_endpoint_info(model_ref: ModelReference | None = None, **overrides: Any) -> EndpointInfo:
    if model_ref is None:
        model_ref = make_model_ref()
    defaults = {
        "name": "test-endpoint",
        "url": "https://api.example.com/v1/test-endpoint",
        "backend": "mock",
        "status": EndpointStatus.RUNNING,
        "model_ref": model_ref,
        "deploy_pathspec": "TestFlow/1/deploy/1",
        "backend_id": "ep-123",
        "backend_metadata": {"region": "us-east-1"},
        "created_at": "2025-01-01T00:00:00+00:00",
    }
    defaults.update(overrides)
    return EndpointInfo(**defaults)


class MockBackend(ServingBackend):
    """Test backend that records calls."""

    name = "mock"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def deploy(
        self,
        model_ref: ModelReference,
        endpoint_name: str,
        *,
        config: dict | None = None,
    ) -> EndpointInfo:
        self.calls.append(("deploy", (model_ref, endpoint_name), {"config": config}))
        return EndpointInfo(
            name=endpoint_name,
            url=f"https://mock.api/{endpoint_name}",
            backend=self.name,
            status=EndpointStatus.RUNNING,
            model_ref=model_ref,
            deploy_pathspec="",
        )

    def get_status(self, endpoint_info: EndpointInfo) -> EndpointStatus:
        self.calls.append(("get_status", (endpoint_info,), {}))
        return EndpointStatus.RUNNING

    def delete(self, endpoint_info: EndpointInfo) -> None:
        self.calls.append(("delete", (endpoint_info,), {}))


@pytest.fixture
def model_ref() -> ModelReference:
    return make_model_ref()


@pytest.fixture
def endpoint_info(model_ref: ModelReference) -> EndpointInfo:
    return make_endpoint_info(model_ref)


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()
