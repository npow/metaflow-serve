"""End-to-end integration tests.

These tests exercise the full ServiceSpec -> Deployment pipeline
and optionally hit the free HF Serverless Inference API.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import MockBackend

from metaflow_extensions.serve.plugins.artifacts import Artifacts
from metaflow_extensions.serve.plugins.backends import _REGISTRY
from metaflow_extensions.serve.plugins.backends import register
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.codegen import generate_handler
from metaflow_extensions.serve.plugins.codegen import generate_requirements
from metaflow_extensions.serve.plugins.codegen import get_artifact_names
from metaflow_extensions.serve.plugins.deployment import Deployment
from metaflow_extensions.serve.plugins.service import ServiceSpec
from metaflow_extensions.serve.plugins.service import _get_endpoints
from metaflow_extensions.serve.plugins.service import _get_init_config
from metaflow_extensions.serve.plugins.service import endpoint
from metaflow_extensions.serve.plugins.service import initialize
from metaflow_extensions.serve.plugins.simulator import HFLocalSimulator

# ── Helpers ──────────────────────────────────────────────────────────────────


class _E2EMockBackend(MockBackend):
    """MockBackend that returns an empty URL so audit skips HTTP."""

    def deploy(self, model_ref, endpoint_name, *, config=None, service_cls=None, artifacts=None):
        info = super().deploy(
            model_ref, endpoint_name, config=config, service_cls=service_cls, artifacts=artifacts
        )
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

    def test_deploy_passes_service_cls(self):
        """Deployment passes service_cls through to the backend."""
        dep = Deployment(MyModelService, config={"backend": "e2e-mock"})
        # The _E2EMockBackend inherits from MockBackend which records calls
        backend = dep._backend
        deploy_call = backend.calls[0]
        assert deploy_call[0] == "deploy"
        assert deploy_call[2]["service_cls"] is MyModelService

    def test_serialization_roundtrip(self):
        dep = Deployment(MyModelService, config={"backend": "e2e-mock"})
        d = dep.as_dict()
        assert isinstance(d, dict)
        assert d["backend"] == "mock"
        assert "model_ref" in d


class TestCodegenE2E:
    """Full round-trip: ServiceSpec + artifacts → codegen → execute generated handler."""

    def test_codegen_round_trip_with_artifacts(self):
        """Simulate what HF would do: generate handler, pickle artifacts, run it."""
        mock_step = MagicMock()
        mock_step.task.data.model = {"weights": [0.5, 1.5]}
        mock_step.task.data.scaler = "minmax"

        class TrainedService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model
                self.scaler = self.artifacts.flow.scaler

            @endpoint
            def predict(self, request_dict):
                return {
                    "weights": self.model["weights"],
                    "scaler": self.scaler,
                }

        # 1. Extract artifact names (what _push_service_files does)
        artifact_names = get_artifact_names(TrainedService)
        assert artifact_names == ["model", "scaler"]

        # 2. Generate handler.py (what _push_service_files does)
        handler_code = generate_handler(TrainedService, artifact_names)

        # 3. Simulate HF repo: write handler + pickled artifacts to a tmpdir
        tmpdir = tempfile.mkdtemp()
        art_dir = Path(tmpdir) / "artifacts"
        art_dir.mkdir()

        # Pickle artifacts from the Artifacts object (what _push_service_files does)
        artifacts = Artifacts.from_step(mock_step)
        for name in artifact_names:
            data = getattr(artifacts.flow, name)
            with open(art_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(data, f)

        # 4. Execute the generated handler (what HF does on endpoint startup)
        ns: dict = {}
        exec(compile(handler_code, "<handler>", "exec"), ns)  # noqa: S102
        handler = ns["EndpointHandler"](tmpdir)

        # 5. Call the handler (what HF does on each request)
        result = handler({"inputs": "test"})
        assert result == {"weights": [0.5, 1.5], "scaler": "minmax"}

    def test_codegen_generates_requirements_from_env_info(self):
        """When env_info is available, requirements.txt includes resolved deps."""
        env_info = {
            "pypi": [
                {"url": "https://files.pythonhosted.org/numpy-1.26.4-cp310-cp310-linux_x86_64.whl"},
                {
                    "url": "https://files.pythonhosted.org/scikit_learn-1.4.0-cp310-cp310-linux_x86_64.whl"
                },
            ]
        }
        reqs = generate_requirements(env_info=env_info)
        lines = reqs.strip().split("\n")
        assert "numpy==1.26.4" in lines
        assert "scikit-learn==1.4.0" in lines

    def test_codegen_round_trip_with_simulator(self):
        """Full round-trip through subprocess simulator instead of exec()."""
        mock_step = MagicMock()
        mock_step.task.data.model = {"weights": [0.5, 1.5]}
        mock_step.task.data.scaler = "minmax"

        class SimService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model
                self.scaler = self.artifacts.flow.scaler

            @endpoint
            def predict(self, request_dict):
                return {"weights": self.model["weights"], "scaler": self.scaler}

        artifact_names = get_artifact_names(SimService)
        handler_code = generate_handler(SimService, artifact_names)

        # Pickle artifacts
        artifacts = Artifacts.from_step(mock_step)
        pickled = {}
        for name in artifact_names:
            pickled[name] = pickle.dumps(getattr(artifacts.flow, name))

        # Run through simulator
        with HFLocalSimulator(handler_code=handler_code, artifacts=pickled) as sim:
            result = sim.call({"inputs": "test"})
            assert result == {"weights": [0.5, 1.5], "scaler": "minmax"}

    def test_codegen_round_trip_with_http_simulator(self):
        """Full round-trip through HTTP simulator instead of stdin/stdout."""
        mock_step = MagicMock()
        mock_step.task.data.model = {"weights": [0.5, 1.5]}
        mock_step.task.data.scaler = "minmax"

        class HTTPSimService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model
                self.scaler = self.artifacts.flow.scaler

            @endpoint
            def predict(self, request_dict):
                return {"weights": self.model["weights"], "scaler": self.scaler}

        artifact_names = get_artifact_names(HTTPSimService)
        handler_code = generate_handler(HTTPSimService, artifact_names)

        # Pickle artifacts
        artifacts = Artifacts.from_step(mock_step)
        pickled = {}
        for name in artifact_names:
            pickled[name] = pickle.dumps(getattr(artifacts.flow, name))

        # Run through HTTP simulator
        with HFLocalSimulator(handler_code=handler_code, artifacts=pickled, http=True) as sim:
            result = sim.call({"inputs": "test"})
            assert result == {"weights": [0.5, 1.5], "scaler": "minmax"}

    def test_codegen_round_trip_with_numpy(self):
        """Full round-trip with a real third-party lib (numpy)."""
        numpy = __import__("numpy")

        mock_step = MagicMock()
        mock_step.task.data.weights = numpy.array([1.0, 2.0, 3.0])

        class NumpyService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.weights = self.artifacts.flow.weights

            @endpoint
            def predict(self, request_dict):
                return {"mean": float(self.weights.mean()), "sum": float(self.weights.sum())}

        artifact_names = get_artifact_names(NumpyService)
        handler_code = generate_handler(NumpyService, artifact_names)

        # Write pickled numpy array
        tmpdir = tempfile.mkdtemp()
        art_dir = Path(tmpdir) / "artifacts"
        art_dir.mkdir()
        artifacts = Artifacts.from_step(mock_step)
        for name in artifact_names:
            with open(art_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(getattr(artifacts.flow, name), f)

        # Execute handler
        ns: dict = {}
        exec(compile(handler_code, "<handler>", "exec"), ns)  # noqa: S102
        handler = ns["EndpointHandler"](tmpdir)
        result = handler({"inputs": "ignored"})
        assert result == {"mean": 2.0, "sum": 6.0}
