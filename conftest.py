"""Root conftest — provides globals for pytest-markdown-docs README tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from metaflow_extensions.serve import Deployment
from metaflow_extensions.serve import ServiceSpec
from metaflow_extensions.serve import endpoint
from metaflow_extensions.serve import initialize
from metaflow_extensions.serve.plugins.artifacts import Artifacts
from metaflow_extensions.serve.plugins.backends import _REGISTRY
from metaflow_extensions.serve.plugins.backends import register
from metaflow_extensions.serve.plugins.backends.backend import EndpointInfo
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.backend import ModelReference
from metaflow_extensions.serve.plugins.backends.backend import ServingBackend


class _ReadmeMockBackend(ServingBackend):
    """Backend used by README snippets — deploys instantly, no network."""

    name = "readme-mock"

    def deploy(self, model_ref, endpoint_name, *, config=None):  # type: ignore[override]
        return EndpointInfo(
            name=endpoint_name,
            url="",
            backend=self.name,
            status=EndpointStatus.RUNNING,
            model_ref=model_ref,
            deploy_pathspec="",
        )

    def get_status(self, endpoint_info):  # type: ignore[override]
        return EndpointStatus.RUNNING

    def delete(self, endpoint_info):  # type: ignore[override]
        pass


def _mock_step(**artifacts):  # type: ignore[no-untyped-def]
    """Create a mock Metaflow step with the given artifacts on task.data."""
    step = MagicMock()
    for name, value in artifacts.items():
        setattr(step.task.data, name, value)
    return step


# Track backends registered by README snippets so we can clean up after each block.
_readme_registered: list[str] = []


def _readme_register(name, backend_cls):  # type: ignore[no-untyped-def]
    """Register a backend and track it for automatic cleanup."""
    register(name, backend_cls)
    _readme_registered.append(name)


def pytest_markdown_docs_globals():  # type: ignore[no-untyped-def]
    """Inject these names into every README code block automatically."""
    # Clean up any backends registered by the previous block
    while _readme_registered:
        _REGISTRY.pop(_readme_registered.pop(), None)

    return {
        # Core API
        "ServiceSpec": ServiceSpec,
        "endpoint": endpoint,
        "initialize": initialize,
        "Deployment": Deployment,
        "Artifacts": Artifacts,
        # Backend internals (for custom backend example)
        "ServingBackend": ServingBackend,
        "EndpointInfo": EndpointInfo,
        "EndpointStatus": EndpointStatus,
        "ModelReference": ModelReference,
        "register": _readme_register,
        # Test helpers
        "MagicMock": MagicMock,
        "mock_step": _mock_step,
        "_ReadmeMockBackend": _ReadmeMockBackend,
    }
