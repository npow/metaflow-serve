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
from metaflow_extensions.serve import ServiceSpec, Deployment, endpoint, initialize

class MyService(ServiceSpec):
    @initialize(backend="huggingface")
    def init(self):
        self.model = self.artifacts.flow.model

    @endpoint
    def predict(self, request_dict):
        return {"result": self.model.predict(request_dict["input"])}

# Deploy from a Metaflow step
deployment = (
    Deployment(MyService, step=self.train, config={"repository": "user/model"})
    .audit("predict", payload={"input": [1, 2, 3]})
    .promote()
)
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

### Deploy, audit, and promote from a flow

```python
from metaflow import FlowSpec, step
from metaflow_extensions.serve import Deployment

class TrainAndDeployFlow(FlowSpec):
    @step
    def train(self):
        self.model = train_model()
        self.tokenizer = load_tokenizer()
        self.next(self.deploy)

    @step
    def deploy(self):
        self.deployment = (
            Deployment(SentimentService, step=self.train, config={
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
```

### Add a custom backend

```python
from metaflow_extensions.serve.plugins.backends.backend import ServingBackend
from metaflow_extensions.serve.plugins.backends import register

class ModalBackend(ServingBackend):
    name = "modal"

    def deploy(self, model_ref, endpoint_name, *, config=None):
        ...

    def get_status(self, endpoint_info):
        ...

    def delete(self, endpoint_info):
        ...

register("modal", ModalBackend)
```

## How it works

**ServiceSpec** defines your serving logic as a class — `@initialize` loads artifacts, `@endpoint` handles requests. **Deployment** takes the service class plus a Metaflow step reference, resolves artifacts, calls the backend to create an endpoint, and returns a chainable object for auditing and promotion. Every deployment carries a `ModelReference` linking back to the exact flow/run/step/task that produced the model.

Built-in backends: HuggingFace Inference Endpoints. Extensible via the `ServingBackend` ABC.

## Development

```bash
git clone https://github.com/npow/metaflow-serve.git
cd metaflow-serve
pip install -e ".[huggingface,dev]"

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Tests
pytest

# Type checking
mypy src/
```

## License

[Apache-2.0](LICENSE)
