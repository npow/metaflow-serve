"""Tests for backend data types, ABC, and registry."""

from __future__ import annotations

import dataclasses

from conftest import MockBackend
from conftest import make_endpoint_info
from conftest import make_model_ref

from metaflow_extensions.serve.plugins.backends import get_backend
from metaflow_extensions.serve.plugins.backends import list_backends
from metaflow_extensions.serve.plugins.backends import register
from metaflow_extensions.serve.plugins.backends.backend import EndpointInfo
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.backend import ModelReference
from metaflow_extensions.serve.plugins.backends.backend import asdict_with_enums


class TestModelReference:
    def test_fields(self):
        ref = make_model_ref()
        assert ref.flow_name == "TestFlow"
        assert ref.pathspec == "TestFlow/1/train/1"
        assert ref.framework == "transformers"

    def test_optional_fields_default_none(self):
        ref = ModelReference(
            flow_name="F",
            run_id="1",
            step_name="s",
            task_id="1",
            artifact_name="m",
            pathspec="F/1/s/1",
        )
        assert ref.framework is None
        assert ref.task_type is None

    def test_roundtrip_serialization(self):
        ref = make_model_ref()
        d = dataclasses.asdict(ref)
        ref2 = ModelReference(**d)
        assert ref == ref2


class TestEndpointInfo:
    def test_fields(self):
        info = make_endpoint_info()
        assert info.name == "test-endpoint"
        assert info.status == EndpointStatus.RUNNING
        assert info.model_ref.flow_name == "TestFlow"

    def test_roundtrip_serialization(self):
        info = make_endpoint_info()
        d = asdict_with_enums(info)
        # EndpointStatus is serialized as its value string
        assert d["status"] == "running"
        assert d["model_ref"]["flow_name"] == "TestFlow"

    def test_default_metadata(self):
        ref = make_model_ref()
        info = EndpointInfo(
            name="e",
            url="",
            backend="b",
            status=EndpointStatus.PENDING,
            model_ref=ref,
            deploy_pathspec="",
        )
        assert info.backend_metadata == {}
        assert info.backend_id is None


class TestEndpointStatus:
    def test_all_values(self):
        expected = {
            "pending",
            "initializing",
            "running",
            "updating",
            "failed",
            "scaled_to_zero",
            "deleted",
            "unknown",
        }
        assert {s.value for s in EndpointStatus} == expected


class TestRegistry:
    def test_register_and_get(self):
        register("mock_test", MockBackend)
        backend = get_backend("mock_test")
        assert isinstance(backend, MockBackend)
        assert backend.name == "mock"

    def test_get_unknown_raises(self):
        import pytest

        with pytest.raises(KeyError, match="Unknown serving backend"):
            get_backend("nonexistent_backend_xyz")

    def test_list_backends(self):
        register("mock_test", MockBackend)
        backends = list_backends()
        assert "mock_test" in backends


class TestMockBackend:
    def test_deploy(self, model_ref):
        backend = MockBackend()
        info = backend.deploy(model_ref, "ep-1")
        assert info.name == "ep-1"
        assert info.status == EndpointStatus.RUNNING
        assert len(backend.calls) == 1
        assert backend.calls[0][0] == "deploy"

    def test_get_status(self, endpoint_info):
        backend = MockBackend()
        status = backend.get_status(endpoint_info)
        assert status == EndpointStatus.RUNNING

    def test_delete(self, endpoint_info):
        backend = MockBackend()
        backend.delete(endpoint_info)
        assert backend.calls[0][0] == "delete"
