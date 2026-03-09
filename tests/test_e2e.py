"""End-to-end integration tests.

These tests exercise the full ServiceSpec -> Deployment pipeline
and optionally hit the free HF Serverless Inference API.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import MockBackend

from metaflow_extensions.serve.plugins.artifacts import Artifacts
from metaflow_extensions.serve.plugins.backends import _REGISTRY
from metaflow_extensions.serve.plugins.backends import register
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.deployment import Deployment
from metaflow_extensions.serve.plugins.service import ServiceSpec
from metaflow_extensions.serve.plugins.service import _get_endpoints
from metaflow_extensions.serve.plugins.service import _get_init_config
from metaflow_extensions.serve.plugins.service import endpoint
from metaflow_extensions.serve.plugins.service import initialize

# ── Helpers ──────────────────────────────────────────────────────────────────


class _E2EMockBackend(MockBackend):
    """MockBackend that returns an empty URL so audit skips HTTP."""

    def deploy(self, model_ref, endpoint_name, *, config=None):
        info = super().deploy(model_ref, endpoint_name, config=config)
        info.url = ""
        return info


class MyModelService(ServiceSpec):
    @initialize(backend="huggingface", cpu=1, memory=2048)
    def init(self):
        self.model_loaded = True

    @endpoint
    def predict(self, request_dict):
        return {"prediction": sum(request_dict.get("input", []))}

    @endpoint(name="health", description="Health check")
    def healthcheck(self, request_dict):
        return {"status": "ok", "model_loaded": self.model_loaded}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestServiceSpecE2E:
    def test_full_lifecycle(self):
        svc = MyModelService()
        assert svc.model_loaded is True
        assert svc.predict({"input": [1, 2, 3]}) == {"prediction": 6}
        assert svc.healthcheck({}) == {"status": "ok", "model_loaded": True}

    def test_introspection(self):
        config = _get_init_config(MyModelService)
        assert config["backend"] == "huggingface"
        assert config["cpu"] == 1

        eps = _get_endpoints(MyModelService)
        names = {ep["name"] for ep in eps}
        assert names == {"predict", "health"}


class TestArtifactsE2E:
    def test_load_artifacts_from_step(self):
        mock_step = MagicMock()
        mock_step.task.data.model = {"weights": [0.1, 0.2]}
        mock_step.task.data.scaler = "standard"

        class ArtifactService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model
                self.scaler = self.artifacts.flow.scaler

            @endpoint
            def predict(self, request_dict):
                return {"weights": self.model["weights"]}

        svc = ArtifactService(artifacts=Artifacts.from_step(mock_step))
        assert svc.model == {"weights": [0.1, 0.2]}
        assert svc.scaler == "standard"
        assert svc.predict({}) == {"weights": [0.1, 0.2]}


class TestDeploymentE2E:
    @pytest.fixture(autouse=True)
    def _register_mock(self):
        register("e2e-mock", _E2EMockBackend)
        yield
        _REGISTRY.pop("e2e-mock", None)

    def test_deploy_audit_promote_chain(self):
        dep = (
            Deployment(MyModelService, config={"backend": "e2e-mock"})
            .audit("predict", payload={"input": [1, 2, 3]})
            .promote()
        )
        assert dep._promoted is True
        assert dep.version.status == EndpointStatus.RUNNING
        d = dep.as_dict()
        assert d["status"] == "running"
        assert d["name"] == "MyModelService"

    def test_serialization_roundtrip(self):
        dep = Deployment(MyModelService, config={"backend": "e2e-mock"})
        d = dep.as_dict()
        assert isinstance(d, dict)
        assert d["backend"] == "mock"
        assert "model_ref" in d
