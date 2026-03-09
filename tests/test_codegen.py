"""Tests for the codegen module."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

from metaflow_extensions.serve.plugins.codegen import extract_requirements_from_env_info
from metaflow_extensions.serve.plugins.codegen import generate_handler
from metaflow_extensions.serve.plugins.codegen import generate_requirements
from metaflow_extensions.serve.plugins.codegen import get_artifact_names
from metaflow_extensions.serve.plugins.service import ServiceSpec
from metaflow_extensions.serve.plugins.service import endpoint
from metaflow_extensions.serve.plugins.service import initialize

# ── Test services ────────────────────────────────────────────────────────────


class ArtifactService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.model = self.artifacts.flow.model
        self.tokenizer = self.artifacts.flow.tokenizer

    @endpoint
    def predict(self, request_dict):
        return {"result": "ok"}


class NoArtifactService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.ready = True

    @endpoint
    def predict(self, request_dict):
        return {"result": "ok"}


class MultiEndpointService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.data = self.artifacts.flow.data

    @endpoint
    def predict(self, request_dict):
        return {"prediction": 1}

    @endpoint(name="health")
    def health(self, request_dict):
        return {"status": "ok"}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestGetArtifactNames:
    def test_extracts_artifact_names(self):
        names = get_artifact_names(ArtifactService)
        assert names == ["model", "tokenizer"]

    def test_no_artifacts(self):
        names = get_artifact_names(NoArtifactService)
        assert names == []

    def test_single_artifact(self):
        names = get_artifact_names(MultiEndpointService)
        assert names == ["data"]


class TestGenerateHandler:
    def test_valid_python(self):
        code = generate_handler(ArtifactService, ["model", "tokenizer"])
        compile(code, "<handler>", "exec")

    def test_has_endpoint_handler(self):
        code = generate_handler(ArtifactService, ["model", "tokenizer"])
        assert "class EndpointHandler" in code

    def test_loads_artifacts(self):
        code = generate_handler(ArtifactService, ["model", "tokenizer"])
        assert "'model'" in code
        assert "'tokenizer'" in code
        assert "pickle.load" in code

    def test_delegates_to_endpoint_method(self):
        code = generate_handler(ArtifactService, ["model", "tokenizer"])
        assert "self.service.predict(data)" in code

    def test_embeds_service_class(self):
        code = generate_handler(ArtifactService, ["model", "tokenizer"])
        assert "class ArtifactService" in code

    def test_no_artifacts_handler(self):
        code = generate_handler(NoArtifactService, [])
        compile(code, "<handler>", "exec")
        assert "class EndpointHandler" in code


class TestGenerateHandlerExecution:
    """Tests that actually execute the generated handler code end-to-end."""

    @staticmethod
    def _exec_handler(service_cls, artifact_names, artifact_data):
        """Generate handler, write pickled artifacts to tmpdir, exec, return EndpointHandler."""
        code = generate_handler(service_cls, artifact_names)
        tmpdir = tempfile.mkdtemp()
        art_dir = Path(tmpdir) / "artifacts"
        art_dir.mkdir()
        for name, value in artifact_data.items():
            with open(art_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(value, f)
        ns: dict = {}
        exec(compile(code, "<handler>", "exec"), ns)  # noqa: S102
        return ns["EndpointHandler"](tmpdir)

    def test_handler_runs_with_artifacts(self):
        """Generated handler loads pickled artifacts and serves predictions."""
        handler = self._exec_handler(
            ArtifactService,
            ["model", "tokenizer"],
            {"model": {"weights": [1, 2, 3]}, "tokenizer": "tok-v1"},
        )
        result = handler({"inputs": "test"})
        assert result == {"result": "ok"}

    def test_handler_no_artifacts(self):
        """Generated handler works when service uses no artifacts."""
        handler = self._exec_handler(NoArtifactService, [], {})
        result = handler({"inputs": "test"})
        assert result == {"result": "ok"}

    def test_handler_uses_artifact_values(self):
        """Generated handler gives the service access to the actual artifact data."""

        class SumService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.weights = self.artifacts.flow.weights

            @endpoint
            def predict(self, request_dict):
                return {"total": sum(self.weights)}

        handler = self._exec_handler(
            SumService,
            ["weights"],
            {"weights": [10, 20, 30]},
        )
        result = handler({"inputs": "ignored"})
        assert result == {"total": 60}

    def test_handler_with_numpy_artifact(self):
        """Generated handler works with third-party lib artifacts (numpy)."""
        numpy = __import__("numpy")

        class NumpyService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.arr = self.artifacts.flow.arr

            @endpoint
            def predict(self, request_dict):
                return {"mean": float(self.arr.mean())}

        arr = numpy.array([1.0, 2.0, 3.0, 4.0])
        handler = self._exec_handler(NumpyService, ["arr"], {"arr": arr})
        result = handler({"inputs": "ignored"})
        assert result == {"mean": 2.5}


class TestExtractRequirementsFromEnvInfo:
    """Tests for extracting pip deps from Metaflow task environment_info."""

    def test_extracts_from_wheel_urls(self):
        env_info = {
            "pypi": [
                {"url": "https://files.pythonhosted.org/numpy-1.26.4-cp310-cp310-linux_x86_64.whl"},
                {"url": "https://files.pythonhosted.org/pandas-2.1.0-cp310-cp310-linux_x86_64.whl"},
            ]
        }
        deps = extract_requirements_from_env_info(env_info)
        assert "numpy==1.26.4" in deps
        assert "pandas==2.1.0" in deps

    def test_extracts_from_sdist_urls(self):
        env_info = {
            "pypi": [
                {"url": "https://files.pythonhosted.org/my_package-0.3.1.tar.gz"},
            ]
        }
        deps = extract_requirements_from_env_info(env_info)
        assert "my-package==0.3.1" in deps

    def test_empty_env_info(self):
        assert extract_requirements_from_env_info({}) == []
        assert extract_requirements_from_env_info({"pypi": []}) == []

    def test_handles_underscores_in_names(self):
        env_info = {
            "pypi": [
                {"url": "https://example.com/scikit_learn-1.4.0-cp310-cp310-linux_x86_64.whl"},
            ]
        }
        deps = extract_requirements_from_env_info(env_info)
        assert "scikit-learn==1.4.0" in deps

    def test_ignores_entries_without_url(self):
        env_info = {"pypi": [{"other_key": "value"}, {}]}
        assert extract_requirements_from_env_info(env_info) == []


class TestGenerateRequirements:
    def test_fallback_metaflow_serve_when_no_env(self):
        reqs = generate_requirements()
        assert "metaflow-serve" in reqs

    def test_extra_deps(self):
        reqs = generate_requirements(extra_deps=["torch", "transformers"])
        lines = reqs.strip().split("\n")
        assert "metaflow-serve" in lines
        assert "torch" in lines
        assert "transformers" in lines

    def test_no_extras_no_env(self):
        reqs = generate_requirements()
        lines = reqs.strip().split("\n")
        assert lines == ["metaflow-serve"]

    def test_uses_env_info_when_provided(self):
        env_info = {
            "pypi": [
                {"url": "https://example.com/numpy-1.26.4-cp310-cp310-linux_x86_64.whl"},
                {"url": "https://example.com/torch-2.1.0-cp310-cp310-linux_x86_64.whl"},
            ]
        }
        reqs = generate_requirements(env_info=env_info)
        lines = reqs.strip().split("\n")
        assert "numpy==1.26.4" in lines
        assert "torch==2.1.0" in lines
        # When env_info provides deps, metaflow-serve fallback is not added
        assert "metaflow-serve" not in lines

    def test_env_info_plus_extra_deps(self):
        env_info = {
            "pypi": [
                {"url": "https://example.com/numpy-1.26.4-cp310-cp310-linux_x86_64.whl"},
            ]
        }
        reqs = generate_requirements(env_info=env_info, extra_deps=["my-custom-lib"])
        lines = reqs.strip().split("\n")
        assert "numpy==1.26.4" in lines
        assert "my-custom-lib" in lines

    def test_empty_env_info_falls_back(self):
        """When env_info has no pypi packages, fall back to metaflow-serve."""
        reqs = generate_requirements(env_info={})
        lines = reqs.strip().split("\n")
        assert "metaflow-serve" in lines
