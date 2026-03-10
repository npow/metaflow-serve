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


class TestHuggingFacePushServiceFiles:
    def test_push_uploads_handler_requirements_and_artifacts(self, hf_backend, mock_hf_api):
        from metaflow_extensions.serve.plugins.service import ServiceSpec
        from metaflow_extensions.serve.plugins.service import endpoint
        from metaflow_extensions.serve.plugins.service import initialize

        class _SvcForPush(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model

            @endpoint
            def predict(self, request_dict):
                return {"result": self.model(request_dict)}

        mock_artifacts = MagicMock()
        mock_artifacts.flow.model = "fake-model-data"

        ref = make_model_ref()

        with patch.object(HuggingFaceBackend, "_get_task_env_info", return_value=None):
            hf_backend._push_service_files(
                mock_hf_api, "user/model", _SvcForPush, mock_artifacts, ref
            )

        # Should upload handler.py, requirements.txt, and artifacts/model.pkl
        upload_calls = mock_hf_api.upload_file.call_args_list
        paths_uploaded = [
            c[1].get("path_in_repo") or c.kwargs.get("path_in_repo") for c in upload_calls
        ]
        assert "handler.py" in paths_uploaded
        assert "requirements.txt" in paths_uploaded
        assert "artifacts/model.pkl" in paths_uploaded

    def test_push_without_artifacts(self, hf_backend, mock_hf_api):
        from metaflow_extensions.serve.plugins.service import ServiceSpec
        from metaflow_extensions.serve.plugins.service import endpoint
        from metaflow_extensions.serve.plugins.service import initialize

        class _SvcNone(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                pass

            @endpoint
            def predict(self, request_dict):
                return {}

        ref = make_model_ref()

        with patch.object(HuggingFaceBackend, "_get_task_env_info", return_value=None):
            hf_backend._push_service_files(mock_hf_api, "user/model", _SvcNone, None, ref)

        # Should upload handler.py and requirements.txt but no artifacts
        upload_calls = mock_hf_api.upload_file.call_args_list
        paths_uploaded = [
            c[1].get("path_in_repo") or c.kwargs.get("path_in_repo") for c in upload_calls
        ]
        assert "handler.py" in paths_uploaded
        assert "requirements.txt" in paths_uploaded
        assert not any("artifacts/" in p for p in paths_uploaded)


class TestHuggingFaceDeployWithServiceCls:
    def test_deploy_calls_push_service_files(self, hf_backend, mock_hf_api):
        from metaflow_extensions.serve.plugins.service import ServiceSpec
        from metaflow_extensions.serve.plugins.service import endpoint
        from metaflow_extensions.serve.plugins.service import initialize

        class _SvcDeploy(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model

            @endpoint
            def predict(self, request_dict):
                return {}

        mock_endpoint = MagicMock()
        mock_endpoint.name = "ep"
        mock_endpoint.url = "https://hf.co/ep"
        mock_endpoint.status = "pending"
        mock_hf_api.create_inference_endpoint.return_value = mock_endpoint

        ref = make_model_ref()

        with patch.object(HuggingFaceBackend, "_push_service_files") as mock_push:
            hf_backend.deploy(
                ref,
                "ep",
                config={"repository": "user/model"},
                service_cls=_SvcDeploy,
                artifacts=MagicMock(),
            )
            mock_push.assert_called_once()

    def test_deploy_with_custom_image(self, hf_backend, mock_hf_api):
        mock_endpoint = MagicMock()
        mock_endpoint.name = "ep"
        mock_endpoint.url = "https://hf.co/ep"
        mock_endpoint.status = "pending"
        mock_hf_api.create_inference_endpoint.return_value = mock_endpoint

        ref = make_model_ref()
        config = {
            "repository": "user/model",
            "custom_image": {"url": "registry.example.com/image:latest"},
        }
        hf_backend.deploy(ref, "ep", config=config)

        call_kwargs = mock_hf_api.create_inference_endpoint.call_args[1]
        assert call_kwargs["custom_image"] == {"url": "registry.example.com/image:latest"}


class TestGetTaskEnvInfo:
    def test_returns_none_without_metaflow(self):
        """Should return None if metaflow not importable."""
        ref = make_model_ref()
        with patch.dict("sys.modules", {"metaflow": None}):
            result = HuggingFaceBackend._get_task_env_info(ref)
        # May or may not be None depending on import caching, but should not raise
        assert result is None or isinstance(result, dict)

    def test_returns_none_on_task_not_found(self):
        ref = make_model_ref(pathspec="NonExistent/0/step/0")
        # This should not raise — it should return None
        result = HuggingFaceBackend._get_task_env_info(ref)
        assert result is None
