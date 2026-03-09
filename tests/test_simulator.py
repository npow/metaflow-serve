"""Tests for the HFLocalSimulator."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from metaflow_extensions.serve.plugins.codegen import generate_handler
from metaflow_extensions.serve.plugins.codegen import get_artifact_names
from metaflow_extensions.serve.plugins.service import ServiceSpec
from metaflow_extensions.serve.plugins.service import endpoint
from metaflow_extensions.serve.plugins.service import initialize
from metaflow_extensions.serve.plugins.simulator import HFLocalSimulator
from metaflow_extensions.serve.plugins.simulator import SimulatorError

# ── Test services ────────────────────────────────────────────────────────────


class SimpleService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.ready = True

    @endpoint
    def predict(self, request_dict):
        return {"echo": request_dict.get("input", None), "ready": self.ready}


class ArtifactService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.model = self.artifacts.flow.model
        self.scaler = self.artifacts.flow.scaler

    @endpoint
    def predict(self, request_dict):
        return {"weights": self.model["weights"], "scaler": self.scaler}


class NumpyService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.weights = self.artifacts.flow.weights

    @endpoint
    def predict(self, request_dict):
        return {"mean": float(self.weights.mean()), "sum": float(self.weights.sum())}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_simulator(service_cls, artifact_data=None, **kwargs):
    """Build an HFLocalSimulator from a ServiceSpec and optional artifact data."""
    artifact_names = get_artifact_names(service_cls)
    handler_code = generate_handler(service_cls, artifact_names)
    pickled: dict[str, bytes] = {}
    if artifact_data:
        for name, value in artifact_data.items():
            pickled[name] = pickle.dumps(value)
    return HFLocalSimulator(handler_code=handler_code, artifacts=pickled, **kwargs)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSimulatorBasic:
    def test_simple_call(self):
        """SimpleService with no artifacts — call predict, verify response."""
        with _make_simulator(SimpleService) as sim:
            result = sim.call({"input": "hello"})
            assert result == {"echo": "hello", "ready": True}

    def test_multiple_calls(self):
        """Multiple sequential calls work."""
        with _make_simulator(SimpleService) as sim:
            r1 = sim.call({"input": 1})
            r2 = sim.call({"input": 2})
            assert r1["echo"] == 1
            assert r2["echo"] == 2


class TestSimulatorWithArtifacts:
    def test_artifacts_loaded(self):
        """Service loading pickled artifacts — verify values come through."""
        sim = _make_simulator(
            ArtifactService,
            {"model": {"weights": [0.5, 1.5]}, "scaler": "minmax"},
        )
        with sim:
            result = sim.call({"inputs": "test"})
            assert result == {"weights": [0.5, 1.5], "scaler": "minmax"}


class TestSimulatorWithNumpy:
    def test_numpy_artifact(self):
        """numpy array artifact deserializes correctly in subprocess."""
        numpy = pytest.importorskip("numpy")
        arr = numpy.array([1.0, 2.0, 3.0])
        with _make_simulator(NumpyService, {"weights": arr}) as sim:
            result = sim.call({"inputs": "ignored"})
            assert result == {"mean": 2.0, "sum": 6.0}


class TestSimulatorMissingImport:
    def test_missing_import_fails(self):
        """Service referencing an uninstalled package — simulator reports error at startup."""
        # Handcraft a handler that imports a nonexistent module
        bad_handler = """\
import nonexistent_package_xyz

from types import SimpleNamespace

def initialize(**kwargs):
    def decorator(func):
        func._serve_tag = "_serve_initialize"
        func._serve_config = kwargs
        return func
    return decorator

def endpoint(func=None, *, name=None, description=None):
    def decorator(f):
        f._serve_tag = "_serve_endpoint"
        f._serve_endpoint_name = name or f.__name__
        f._serve_description = description or ""
        return f
    if func is not None:
        return decorator(func)
    return decorator

class ServiceSpec:
    def __init__(self, artifacts=None):
        self.artifacts = artifacts or SimpleNamespace()
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(self, attr_name)
            except AttributeError:
                continue
            if callable(attr) and getattr(attr, "_serve_tag", None) == "_serve_initialize":
                attr()
                break

class BadService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.data = nonexistent_package_xyz.load()

    @endpoint
    def predict(self, request_dict):
        return {"result": "ok"}

class EndpointHandler:
    def __init__(self, path=""):
        artifacts_obj = SimpleNamespace(flow=SimpleNamespace())
        self.service = BadService(artifacts=artifacts_obj)

    def __call__(self, data):
        return self.service.predict(data)
"""
        sim = HFLocalSimulator(handler_code=bad_handler, artifacts={})
        with pytest.raises(SimulatorError, match="nonexistent_package_xyz"):
            sim.start()


class TestSimulatorContextManager:
    def test_cleanup_on_exit(self):
        """Verify cleanup: tmpdir removed and subprocess terminated."""
        sim = _make_simulator(SimpleService)
        sim.start()
        tmpdir = sim._tmpdir
        proc = sim._proc
        assert tmpdir is not None
        assert proc is not None
        assert proc.poll() is None  # still running

        sim.stop()
        assert sim._tmpdir is None
        assert proc.poll() is not None  # terminated

    def test_context_manager_cleans_up(self):
        """Context manager calls stop on exit."""
        sim = _make_simulator(SimpleService)
        with sim:
            tmpdir = sim._tmpdir
            assert tmpdir is not None
        # After exiting context, should be cleaned up
        assert sim._tmpdir is None
        assert sim._proc is None

    def test_call_after_stop_raises(self):
        """Calling after stop raises SimulatorError."""
        sim = _make_simulator(SimpleService)
        sim.start()
        sim.stop()
        with pytest.raises(SimulatorError, match="not running"):
            sim.call({"input": "test"})


class TestSimulatorHTTP:
    def test_http_basic(self):
        """SimpleService over HTTP — call predict, verify response."""
        with _make_simulator(SimpleService, http=True) as sim:
            result = sim.call({"input": "hello"})
            assert result == {"echo": "hello", "ready": True}

    def test_http_with_artifacts(self):
        """ArtifactService over HTTP — verify artifacts loaded."""
        sim = _make_simulator(
            ArtifactService,
            {"model": {"weights": [0.5, 1.5]}, "scaler": "minmax"},
            http=True,
        )
        with sim:
            result = sim.call({"inputs": "test"})
            assert result == {"weights": [0.5, 1.5], "scaler": "minmax"}


class TestSimulatorIsolated:
    def test_isolated_with_valid_requirements(self):
        """Isolated venv with no extra deps — SimpleService uses only stdlib."""
        with _make_simulator(SimpleService, isolate=True) as sim:
            result = sim.call({"input": "isolated"})
            assert result == {"echo": "isolated", "ready": True}


class TestSimulatorIsolatedMissingDep:
    def test_isolated_missing_dep_fails(self):
        """Handler needs numpy but requirements is empty — fails on import."""
        bad_handler = """\
import numpy

from types import SimpleNamespace

class EndpointHandler:
    def __init__(self, path=""):
        self.arr = numpy.array([1, 2, 3])

    def __call__(self, data):
        return {"sum": int(self.arr.sum())}
"""
        sim = HFLocalSimulator(
            handler_code=bad_handler,
            artifacts={},
            isolate=True,
        )
        with pytest.raises(SimulatorError, match="numpy"):
            sim.start()
        sim.stop()


class TestSimulatorModelRepo:
    def test_config_json_written(self):
        """config.json is written to tmpdir like a real HF model repo."""
        sim = _make_simulator(SimpleService)
        sim.start()
        try:
            config_path = Path(sim._tmpdir) / "config.json"
            assert config_path.exists()
            config = json.loads(config_path.read_text())
            assert config == {"framework": "custom"}
        finally:
            sim.stop()
