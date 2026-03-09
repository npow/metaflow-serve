"""Tests for Deployment class."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from metaflow_extensions.serve.plugins.backends.backend import EndpointInfo
from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.backend import ModelReference
from metaflow_extensions.serve.plugins.deployment import AuditError
from metaflow_extensions.serve.plugins.deployment import Deployment
from metaflow_extensions.serve.plugins.service import ServiceSpec
from metaflow_extensions.serve.plugins.service import endpoint
from metaflow_extensions.serve.plugins.service import initialize


class _TestService(ServiceSpec):
    @initialize(backend="mock")
    def init(self):
        pass

    @endpoint
    def predict(self, request_dict):
        return {"prediction": 42}


def _make_mock_backend(status=EndpointStatus.RUNNING):
    """Create a mock backend that returns a running endpoint."""
    mock_backend = MagicMock()
    mock_info = EndpointInfo(
        name="TestService",
        url="https://mock.api/TestService",
        backend="mock",
        status=status,
        model_ref=ModelReference(
            flow_name="unknown",
            run_id="0",
            step_name="unknown",
            task_id="0",
            artifact_name="model",
            pathspec="unknown/0/unknown/0",
        ),
        deploy_pathspec="",
    )
    mock_backend.deploy.return_value = mock_info
    mock_backend.wait_for_ready.return_value = mock_info
    return mock_backend


class TestDeploymentInit:
    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_deploys_with_backend(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"})

        mock_get_backend.assert_called_once_with("mock")
        mock_backend.deploy.assert_called_once()
        assert dep.endpoint_url == "https://mock.api/TestService"

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_uses_service_class_name_as_endpoint(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        Deployment(_TestService, config={"backend": "mock"})

        _, call_args, _call_kwargs = mock_backend.deploy.mock_calls[0]
        assert call_args[1] == "_TestService"

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_custom_endpoint_name(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        Deployment(_TestService, config={"backend": "mock", "endpoint_name": "my-ep"})

        _, call_args, _call_kwargs = mock_backend.deploy.mock_calls[0]
        assert call_args[1] == "my-ep"

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_merges_init_config_with_explicit(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        class CustomService(ServiceSpec):
            @initialize(backend="mock", region="us-west-2")
            def init(self):
                pass

            @endpoint
            def predict(self, request_dict):
                return {}

        Deployment(CustomService, config={"backend": "mock", "instance_type": "large"})

        # Config should have merged values from @initialize + explicit config
        assert mock_backend.deploy.called

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_waits_for_ready(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        Deployment(_TestService, config={"backend": "mock"}, timeout=120)

        mock_backend.wait_for_ready.assert_called_once()
        call_kwargs = mock_backend.wait_for_ready.call_args[1]
        assert call_kwargs["timeout"] == 120

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_skip_wait_with_zero_timeout(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        Deployment(_TestService, config={"backend": "mock"}, timeout=0)

        mock_backend.wait_for_ready.assert_not_called()

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_version_property(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"})

        assert isinstance(dep.version, EndpointInfo)
        assert dep.version.name == "TestService"


class TestDeploymentAudit:
    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_audit_rejects_non_running(self, mock_get_backend):
        mock_backend = _make_mock_backend(status=EndpointStatus.PENDING)
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"}, timeout=0)

        with pytest.raises(AuditError, match="endpoint status is pending"):
            dep.audit("predict")

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_audit_rejects_unknown_endpoint(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"})

        with pytest.raises(AuditError, match="not found"):
            dep.audit("nonexistent")

    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_audit_returns_self(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        # Set URL to empty so the actual HTTP call is skipped
        mock_backend.deploy.return_value.url = ""
        mock_backend.wait_for_ready.return_value.url = ""
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"})
        result = dep.audit("predict", payload={"input": [1, 2, 3]})

        assert result is dep


class TestDeploymentPromote:
    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_promote_returns_self(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"})
        result = dep.promote()

        assert result is dep
        assert dep._promoted is True


class TestDeploymentChaining:
    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_fluent_chain(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_backend.deploy.return_value.url = ""
        mock_backend.wait_for_ready.return_value.url = ""
        mock_get_backend.return_value = mock_backend

        dep = (
            Deployment(_TestService, config={"backend": "mock"})
            .audit("predict", payload={"input": [1]})
            .promote()
        )

        assert dep._promoted is True
        assert isinstance(dep.version, EndpointInfo)


class TestDeploymentSerialization:
    @patch("metaflow_extensions.serve.plugins.deployment.get_backend")
    def test_as_dict(self, mock_get_backend):
        mock_backend = _make_mock_backend()
        mock_get_backend.return_value = mock_backend

        dep = Deployment(_TestService, config={"backend": "mock"})
        d = dep.as_dict()

        assert isinstance(d, dict)
        assert d["name"] == "TestService"
        assert d["status"] == "running"
        assert d["backend"] == "mock"
