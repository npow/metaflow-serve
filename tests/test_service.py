"""Tests for ServiceSpec, @initialize, and @endpoint."""

from __future__ import annotations

from metaflow_extensions.serve.plugins.artifacts import Artifacts
from metaflow_extensions.serve.plugins.service import ServiceSpec
from metaflow_extensions.serve.plugins.service import _get_endpoints
from metaflow_extensions.serve.plugins.service import _get_init_config
from metaflow_extensions.serve.plugins.service import endpoint
from metaflow_extensions.serve.plugins.service import initialize


class TestInitializeDecorator:
    def test_tags_method(self):
        @initialize(backend="huggingface")
        def init(self):
            pass

        assert init._serve_tag == "_serve_initialize"
        assert init._serve_config["backend"] == "huggingface"

    def test_default_backend(self):
        @initialize()
        def init(self):
            pass

        assert init._serve_config["backend"] == "huggingface"

    def test_custom_config(self):
        @initialize(backend="modal", cpu=2, memory=4096, gpu="A10G")
        def init(self):
            pass

        assert init._serve_config["backend"] == "modal"
        assert init._serve_config["cpu"] == 2
        assert init._serve_config["memory"] == 4096
        assert init._serve_config["gpu"] == "A10G"

    def test_none_defaults(self):
        @initialize()
        def init(self):
            pass

        assert init._serve_config["cpu"] is None
        assert init._serve_config["memory"] is None


class TestEndpointDecorator:
    def test_bare_decorator(self):
        @endpoint
        def predict(self, request_dict):
            pass

        assert predict._serve_tag == "_serve_endpoint"
        assert predict._serve_endpoint_name == "predict"

    def test_with_parens(self):
        @endpoint()
        def predict(self, request_dict):
            pass

        assert predict._serve_tag == "_serve_endpoint"
        assert predict._serve_endpoint_name == "predict"

    def test_custom_name(self):
        @endpoint(name="inference")
        def predict(self, request_dict):
            pass

        assert predict._serve_endpoint_name == "inference"

    def test_description_from_docstring(self):
        @endpoint
        def predict(self, request_dict):
            """Run model inference."""

        assert predict._serve_description == "Run model inference."

    def test_explicit_description(self):
        @endpoint(description="Custom description")
        def predict(self, request_dict):
            """Ignored docstring."""

        assert predict._serve_description == "Custom description"


class TestServiceSpec:
    def test_subclass_with_endpoints(self):
        class MyService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.ready = True

            @endpoint
            def predict(self, request_dict):
                return {"result": 42}

        svc = MyService()
        assert svc.ready is True
        assert isinstance(svc.artifacts, Artifacts)

    def test_no_initialize_method(self):
        class SimpleService(ServiceSpec):
            @endpoint
            def predict(self, request_dict):
                return {}

        svc = SimpleService()
        assert isinstance(svc.artifacts, Artifacts)

    def test_custom_artifacts(self):
        artifacts = Artifacts()
        artifacts._sources["flow"] = object()

        class MyService(ServiceSpec):
            @endpoint
            def predict(self, request_dict):
                return {}

        svc = MyService(artifacts=artifacts)
        assert svc.artifacts is artifacts

    def test_initialize_receives_artifacts(self):
        class MyService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.has_artifacts = self.artifacts is not None

            @endpoint
            def predict(self, request_dict):
                return {}

        svc = MyService()
        assert svc.has_artifacts is True


class TestIntrospection:
    def test_get_init_config(self):
        class MyService(ServiceSpec):
            @initialize(backend="modal", cpu=4)
            def init(self):
                pass

            @endpoint
            def predict(self, request_dict):
                return {}

        config = _get_init_config(MyService)
        assert config["backend"] == "modal"
        assert config["cpu"] == 4

    def test_get_init_config_no_init(self):
        class MyService(ServiceSpec):
            @endpoint
            def predict(self, request_dict):
                return {}

        config = _get_init_config(MyService)
        assert config == {}

    def test_get_endpoints(self):
        class MyService(ServiceSpec):
            @endpoint
            def predict(self, request_dict):
                return {}

            @endpoint(name="health")
            def healthcheck(self, request_dict):
                """Health check endpoint."""
                return {"status": "ok"}

        endpoints = _get_endpoints(MyService)
        names = {ep["name"] for ep in endpoints}
        assert "predict" in names
        assert "health" in names

    def test_get_endpoints_empty(self):
        class EmptyService(ServiceSpec):
            pass

        endpoints = _get_endpoints(EmptyService)
        assert endpoints == []
