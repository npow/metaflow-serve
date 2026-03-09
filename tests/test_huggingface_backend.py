"""Tests for the HuggingFace backend with mocked HfApi."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from conftest import make_model_ref

from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.huggingface import HuggingFaceBackend


@pytest.fixture
def hf_backend():
    return HuggingFaceBackend(token="test-token")


@pytest.fixture
def mock_hf_api():
    with patch(
        "metaflow_extensions.serve.plugins.backends.huggingface.HuggingFaceBackend._get_api"
    ) as mock_get:
        api = MagicMock()
        mock_get.return_value = api
        yield api


class TestHuggingFaceDeploy:
    def test_deploy_creates_endpoint(self, hf_backend, mock_hf_api):
        mock_endpoint = MagicMock()
        mock_endpoint.name = "my-endpoint"
        mock_endpoint.url = "https://hf.co/api/my-endpoint"
        mock_endpoint.status = "pending"
        mock_hf_api.create_inference_endpoint.return_value = mock_endpoint

        ref = make_model_ref()
        config = {
            "repository": "user/model",
            "instance_type": "nvidia-a10g.xlarge",
            "region": "us-east-1",
        }

        info = hf_backend.deploy(ref, "my-endpoint", config=config)

        assert info.name == "my-endpoint"
        assert info.url == "https://hf.co/api/my-endpoint"
        assert info.status == EndpointStatus.PENDING
        assert info.backend == "huggingface"
        assert info.model_ref == ref
        mock_hf_api.create_inference_endpoint.assert_called_once()

    def test_deploy_requires_repository(self, hf_backend, mock_hf_api):
        ref = make_model_ref()
        with pytest.raises(ValueError, match="repository"):
            hf_backend.deploy(ref, "ep", config={})

    def test_deploy_no_config(self, hf_backend, mock_hf_api):
        ref = make_model_ref()
        with pytest.raises(ValueError, match="repository"):
            hf_backend.deploy(ref, "ep")

    def test_deploy_passes_optional_fields(self, hf_backend, mock_hf_api):
        mock_endpoint = MagicMock()
        mock_endpoint.name = "ep"
        mock_endpoint.url = "https://hf.co/ep"
        mock_endpoint.status = "pending"
        mock_hf_api.create_inference_endpoint.return_value = mock_endpoint

        ref = make_model_ref()
        config = {
            "repository": "user/model",
            "namespace": "my-org",
            "min_replica": 0,
            "max_replica": 2,
            "revision": "main",
        }
        hf_backend.deploy(ref, "ep", config=config)

        call_kwargs = mock_hf_api.create_inference_endpoint.call_args[1]
        assert call_kwargs["namespace"] == "my-org"
        assert call_kwargs["min_replica"] == 0
        assert call_kwargs["max_replica"] == 2
        assert call_kwargs["revision"] == "main"


class TestHuggingFaceGetStatus:
    def test_get_status_running(self, hf_backend, mock_hf_api):
        mock_endpoint = MagicMock()
        mock_endpoint.status = "running"
        mock_hf_api.get_inference_endpoint.return_value = mock_endpoint

        from conftest import make_endpoint_info

        info = make_endpoint_info(backend="huggingface")

        status = hf_backend.get_status(info)
        assert status == EndpointStatus.RUNNING

    def test_get_status_unknown(self, hf_backend, mock_hf_api):
        mock_endpoint = MagicMock()
        mock_endpoint.status = "some_new_status"
        mock_hf_api.get_inference_endpoint.return_value = mock_endpoint

        from conftest import make_endpoint_info

        info = make_endpoint_info(backend="huggingface")

        status = hf_backend.get_status(info)
        assert status == EndpointStatus.UNKNOWN


class TestHuggingFaceDelete:
    def test_delete(self, hf_backend, mock_hf_api):
        from conftest import make_endpoint_info

        info = make_endpoint_info(backend="huggingface")

        hf_backend.delete(info)
        mock_hf_api.delete_inference_endpoint.assert_called_once()


class TestHuggingFaceWaitForReady:
    def test_wait_delegates_to_hf(self, hf_backend, mock_hf_api):
        mock_endpoint = MagicMock()
        mock_endpoint.status = "running"
        mock_endpoint.url = "https://hf.co/ep"
        mock_hf_api.get_inference_endpoint.return_value = mock_endpoint

        from conftest import make_endpoint_info

        info = make_endpoint_info(backend="huggingface")

        result = hf_backend.wait_for_ready(info, timeout=30)
        assert result.status == EndpointStatus.RUNNING
        mock_endpoint.wait.assert_called_once_with(timeout=30)
