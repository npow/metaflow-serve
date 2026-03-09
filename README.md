# metaflow-serve

[![CI](https://github.com/npow/metaflow-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/metaflow-serve/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/metaflow-serve)](https://pypi.org/project/metaflow-serve/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Deploy ML models from Metaflow flows with full lineage tracking.

## The problem

You trained a model in a Metaflow flow — now you need to serve it. Getting from a trained artifact to a live endpoint means writing deployment glue, losing track of which model version is running, and hoping someone documented the config. When something breaks in production, tracing back to the exact training run is a manual archaeology project.

## Quick start

```bash
pip install "metaflow-serve[huggingface]"
```

```python
from metaflow_extensions.serve import ServiceSpec, endpoint, initialize
from metaflow_extensions.serve.plugins.artifacts import Artifacts
from unittest.mock import MagicMock

class MyService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.model = self.artifacts.flow.model

    @endpoint
    def predict(self, request_dict):
        return {"result": self.model.predict(request_dict["input"])}

# --- test it works ---
mock_step = MagicMock()
mock_step.task.data.model = MagicMock()
mock_step.task.data.model.predict = lambda x: x
svc = MyService(artifacts=Artifacts.from_step(mock_step))
assert svc.predict({"input": [1, 2, 3]}) == {"result": [1, 2, 3]}
```

## Install

```bash
pip install metaflow-serve

# With HuggingFace Inference Endpoints backend
pip install "metaflow-serve[huggingface]"
```

## Usage

### Define a service with artifact access

```python
from metaflow_extensions.serve import ServiceSpec, endpoint, initialize
from metaflow_extensions.serve.plugins.artifacts import Artifacts
from unittest.mock import MagicMock

class SentimentService(ServiceSpec):
    @initialize(backend="huggingface", cpu=1, memory=2048)
    def init(self):
        self.model = self.artifacts.flow.model
        self.tokenizer = self.artifacts.flow.tokenizer

    @endpoint
    def predict(self, request_dict):
        tokens = self.tokenizer(request_dict["text"])
        return {"sentiment": self.model(tokens)}

# --- test it works ---
mock_step = MagicMock()
mock_step.task.data.model = lambda tokens: "positive"
mock_step.task.data.tokenizer = lambda text: f"tok:{text}"
svc = SentimentService(artifacts=Artifacts.from_step(mock_step))
assert svc.predict({"text": "great movie"}) == {"sentiment": "positive"}
```

### Deploy, audit, and promote from a flow

```python notest
from metaflow import FlowSpec, step
from metaflow_extensions.serve import Deployment, ServiceSpec, endpoint, initialize


class SentimentService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.model = self.artifacts.flow.model

    @endpoint
    def predict(self, request_dict):
        return {"sentiment": self.model(request_dict["text"])}


class TrainAndDeployFlow(FlowSpec):
    @step
    def start(self):
        self.model = "trained-model-placeholder"
        self.next(self.deploy)

    @step
    def deploy(self):
        self.deployment = (
            Deployment(SentimentService, step=self, config={
                "repository": "user/sentiment-model",
                "instance_type": "nvidia-a10g.xlarge",
            })
            .audit("predict", payload={"text": "great movie"})
            .promote()
        )
        self.next(self.end)

    @step
    def end(self):
        print(f"Endpoint: {self.deployment.endpoint_url}")


if __name__ == "__main__":
    TrainAndDeployFlow()
```

The deploy flow block above uses Metaflow's `FlowSpec` which requires source file introspection
and cannot be tested in isolation. See `tests/test_readme.py` for a mock-backed end-to-end test
of the Deployment chain.

### Test the Deployment chain

```python
from metaflow_extensions.serve import Deployment, ServiceSpec, endpoint, initialize
from metaflow_extensions.serve.plugins.backends.backend import (
    EndpointInfo, EndpointStatus, ModelReference, ServingBackend,
)
from metaflow_extensions.serve.plugins.backends import register, _REGISTRY

class MockBackend(ServingBackend):
    name = "readme-test"
    def deploy(self, model_ref, endpoint_name, *, config=None):
        return EndpointInfo(
            name=endpoint_name, url="", backend=self.name,
            status=EndpointStatus.RUNNING, model_ref=model_ref, deploy_pathspec="",
        )
    def get_status(self, endpoint_info):
        return EndpointStatus.RUNNING
    def delete(self, endpoint_info):
        pass

register("readme-test", MockBackend)

class TestService(ServiceSpec):
    @initialize(backend="readme-test")
    def init(self):
        pass
    @endpoint
    def predict(self, request_dict):
        return {"ok": True}

dep = Deployment(TestService, config={"backend": "readme-test"}).audit("predict").promote()
assert dep._promoted is True
assert dep.version.status == EndpointStatus.RUNNING
assert dep.as_dict()["status"] == "running"
_REGISTRY.pop("readme-test", None)
```

### Add a custom backend

```python
from metaflow_extensions.serve.plugins.backends.backend import (
    EndpointInfo,
    EndpointStatus,
    ModelReference,
    ServingBackend,
)
from metaflow_extensions.serve.plugins.backends import register, _REGISTRY


class ModalBackend(ServingBackend):
    name = "modal"

    def deploy(self, model_ref: ModelReference, endpoint_name: str, *, config=None) -> EndpointInfo:
        # Deploy to Modal and return endpoint info
        ...

    def get_status(self, endpoint_info: EndpointInfo) -> EndpointStatus:
        # Query Modal for current endpoint status
        ...

    def delete(self, endpoint_info: EndpointInfo) -> None:
        # Tear down the Modal endpoint
        ...


register("modal", ModalBackend)
assert "modal" in _REGISTRY
_REGISTRY.pop("modal", None)
```

## How it works

**ServiceSpec** defines your serving logic as a class — `@initialize` loads artifacts, `@endpoint` handles requests. **Deployment** takes the service class plus a Metaflow step reference, resolves artifacts, calls the backend to create an endpoint, and returns a chainable object for auditing and promotion. Every deployment carries a `ModelReference` linking back to the exact flow/run/step/task that produced the model.

Built-in backends: HuggingFace Inference Endpoints. Extensible via the `ServingBackend` ABC.

## Development

```bash
pip install -e ".[huggingface,dev]"

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Tests (includes README snippet validation)
pytest --markdown-docs

# Type checking
mypy src/
```

## License

[Apache-2.0](LICENSE)
