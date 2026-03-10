# metaflow-serve

[![CI](https://github.com/npow/metaflow-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/metaflow-serve/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/metaflow-serve)](https://pypi.org/project/metaflow-serve/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Docs](https://img.shields.io/badge/docs-mintlify-18a34a?style=flat-square)](https://mintlify.com/npow/metaflow-serve)

Deploy ML models from Metaflow flows with full lineage tracking.

## The problem

You trained a model in a Metaflow flow — now you need to serve it. Getting from a trained artifact to a live endpoint means writing deployment glue, losing track of which model version is running, and hoping someone documented the config. When something breaks in production, tracing back to the exact training run is a manual archaeology project.

## Quick start

```bash
pip install "metaflow-serve[huggingface]"
```

```python
class MyService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.model = self.artifacts.flow.model

    @endpoint
    def predict(self, request_dict):
        return {"result": self.model.predict(request_dict["input"])}
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
class SentimentService(ServiceSpec):
    @initialize(backend="huggingface", cpu=1, memory=2048)
    def init(self):
        self.model = self.artifacts.flow.model
        self.tokenizer = self.artifacts.flow.tokenizer

    @endpoint
    def predict(self, request_dict):
        tokens = self.tokenizer(request_dict["text"])
        return {"sentiment": self.model(tokens)}
```

### Specify Python packages

When deploying from a Metaflow step that uses `@conda` or `@pypi`, the
step's resolved environment is used automatically. You can also specify
packages explicitly via `@initialize`:

```python notest
class TorchService(ServiceSpec):
    @initialize(
        backend="huggingface",
        packages={"torch": ">=2.0", "numpy": ">=1.24"},
    )
    def init(self):
        self.model = self.artifacts.flow.model

    @endpoint
    def predict(self, data):
        import torch
        tensor = torch.tensor(data["input"])
        return {"result": self.model(tensor).tolist()}
```

Explicit `packages` are merged with (and override) any packages resolved
from the Metaflow step environment.

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

### Deploy and verify programmatically

```python
register("readme-mock", _ReadmeMockBackend)

class QuickService(ServiceSpec):
    @initialize(backend="readme-mock")
    def init(self):
        pass

    @endpoint
    def predict(self, request_dict):
        return {"ok": True}

dep = Deployment(QuickService, config={"backend": "readme-mock"})
dep = dep.audit("predict").promote()
print(dep.version.status)   # EndpointStatus.RUNNING
print(dep.as_dict()["name"])  # QuickService
```

### Add a custom backend

```python
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

# Tests (includes README snippet validation via pytest-markdown-docs)
pytest

# Type checking
mypy src/
```

## License

[Apache-2.0](LICENSE)
