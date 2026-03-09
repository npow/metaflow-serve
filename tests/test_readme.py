"""Test that all Python code snippets in README.md parse and execute.

Extracts every ```python block from README.md and runs it. Blocks that
reference Metaflow artifacts or the Deployment API are executed with
lightweight mocks so they validate imports and class definitions without
needing a live backend.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from metaflow_extensions.serve.plugins.backends import _REGISTRY
from metaflow_extensions.serve.plugins.backends import register
from metaflow_extensions.serve.plugins.backends.backend import EndpointInfo
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.backend import ModelReference
from metaflow_extensions.serve.plugins.backends.backend import ServingBackend

README = Path(__file__).resolve().parent.parent / "README.md"

# ── Extract code blocks ──────────────────────────────────────────────────────

_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _PYTHON_BLOCK_RE.finditer(text)]


BLOCKS = _extract_python_blocks(README.read_text())

# Sanity check: README should have at least 4 python blocks
assert len(BLOCKS) >= 4, f"Expected >=4 python blocks in README, found {len(BLOCKS)}"


# ── Mock artifacts ───────────────────────────────────────────────────────────


class _MockModel:
    """Stand-in for a model loaded from artifacts."""

    def predict(self, x: Any) -> Any:
        return x

    def __call__(self, x: Any) -> Any:
        return x


class _ReadmeMockBackend(ServingBackend):
    name = "readme-mock"

    def deploy(
        self,
        model_ref: ModelReference,
        endpoint_name: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> EndpointInfo:
        return EndpointInfo(
            name=endpoint_name,
            url="",
            backend=self.name,
            status=EndpointStatus.RUNNING,
            model_ref=model_ref,
            deploy_pathspec="",
        )

    def get_status(self, endpoint_info: EndpointInfo) -> EndpointStatus:
        return EndpointStatus.RUNNING

    def delete(self, endpoint_info: EndpointInfo) -> None:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────


class TestReadmeQuickStart:
    """Block 0: Quick start ServiceSpec definition."""

    def test_class_definition_and_imports(self):
        block = BLOCKS[0]
        # This block defines MyService — exec it and verify the class exists
        ns: dict[str, Any] = {}
        exec(compile(block, "README.md:quick-start", "exec"), ns)  # noqa: S102
        assert "MyService" in ns
        svc_cls = ns["MyService"]
        assert hasattr(svc_cls, "init")
        assert hasattr(svc_cls, "predict")


class TestReadmeSentimentService:
    """Block 1: SentimentService with artifact access."""

    def test_class_definition(self):
        block = BLOCKS[1]
        ns: dict[str, Any] = {}
        exec(compile(block, "README.md:sentiment-service", "exec"), ns)  # noqa: S102
        assert "SentimentService" in ns
        svc_cls = ns["SentimentService"]
        assert hasattr(svc_cls, "predict")


class TestReadmeDeployFlow:
    """Block 2: Full flow with Deployment chain."""

    def test_syntax_valid(self):
        block = BLOCKS[2]
        # FlowSpec uses inspect.getsourcelines which fails on exec'd code,
        # so we verify syntax by compiling without executing.
        code = compile(block, "README.md:deploy-flow", "exec")
        assert code is not None

    def test_imports_resolve(self):
        """Verify all imports in the deploy flow block are valid."""
        # Extract and run just the import lines
        block = BLOCKS[2]
        import_lines = [
            line for line in block.splitlines()
            if line.startswith(("from ", "import "))
        ]
        ns: dict[str, Any] = {}
        exec(compile("\n".join(import_lines), "README.md:deploy-flow-imports", "exec"), ns)  # noqa: S102
        assert "FlowSpec" in ns
        assert "step" in ns
        assert "Deployment" in ns
        assert "ServiceSpec" in ns
        assert "endpoint" in ns
        assert "initialize" in ns


class TestReadmeCustomBackend:
    """Block 3: Custom backend registration."""

    def test_backend_registers(self):
        block = BLOCKS[3]
        ns: dict[str, Any] = {}
        exec(compile(block, "README.md:custom-backend", "exec"), ns)  # noqa: S102
        assert "ModalBackend" in ns
        assert "modal" in _REGISTRY
        # Clean up
        _REGISTRY.pop("modal", None)


class TestReadmeServiceInstantiation:
    """Verify README services can be instantiated with mock artifacts."""

    def test_quick_start_service_runs(self):
        """MyService from quick start works with mocked artifacts."""
        from metaflow_extensions.serve import ServiceSpec
        from metaflow_extensions.serve import endpoint
        from metaflow_extensions.serve import initialize
        from metaflow_extensions.serve.plugins.artifacts import Artifacts

        mock_step = MagicMock()
        mock_step.task.data.model = _MockModel()

        class MyService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model

            @endpoint
            def predict(self, request_dict):
                return {"result": self.model.predict(request_dict["input"])}

        svc = MyService(artifacts=Artifacts.from_step(mock_step))
        result = svc.predict({"input": [1, 2, 3]})
        assert result == {"result": [1, 2, 3]}

    def test_sentiment_service_runs(self):
        """SentimentService from usage section works with mocked artifacts."""
        from metaflow_extensions.serve import ServiceSpec
        from metaflow_extensions.serve import endpoint
        from metaflow_extensions.serve import initialize
        from metaflow_extensions.serve.plugins.artifacts import Artifacts

        mock_step = MagicMock()
        mock_step.task.data.model = _MockModel()
        mock_step.task.data.tokenizer = lambda text: f"tok:{text}"

        class SentimentService(ServiceSpec):
            @initialize(backend="huggingface", cpu=1, memory=2048)
            def init(self):
                self.model = self.artifacts.flow.model
                self.tokenizer = self.artifacts.flow.tokenizer

            @endpoint
            def predict(self, request_dict):
                tokens = self.tokenizer(request_dict["text"])
                return {"sentiment": self.model(tokens)}

        svc = SentimentService(artifacts=Artifacts.from_step(mock_step))
        result = svc.predict({"text": "great movie"})
        assert result == {"sentiment": "tok:great movie"}

    def test_deployment_chain_runs(self):
        """Deployment chain from README works with mock backend."""
        from metaflow_extensions.serve import Deployment
        from metaflow_extensions.serve import ServiceSpec
        from metaflow_extensions.serve import endpoint
        from metaflow_extensions.serve import initialize

        register("readme-mock", _ReadmeMockBackend)
        try:

            class SentimentService(ServiceSpec):
                @initialize(backend="readme-mock")
                def init(self):
                    pass

                @endpoint
                def predict(self, request_dict):
                    return {"sentiment": "positive"}

            dep = (
                Deployment(SentimentService, config={"backend": "readme-mock"})
                .audit("predict")
                .promote()
            )
            assert dep._promoted is True
            assert dep.version.status == EndpointStatus.RUNNING
        finally:
            _REGISTRY.pop("readme-mock", None)
