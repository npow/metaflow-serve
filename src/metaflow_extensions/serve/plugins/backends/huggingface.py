"""HuggingFace Inference Endpoints backend."""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from datetime import timezone
from typing import Any

from .backend import EndpointInfo
from .backend import EndpointStatus
from .backend import ModelReference
from .backend import ServingBackend

# Map HF status strings to our enum.
_HF_STATUS_MAP = {
    "pending": EndpointStatus.PENDING,
    "initializing": EndpointStatus.INITIALIZING,
    "running": EndpointStatus.RUNNING,
    "updating": EndpointStatus.UPDATING,
    "failed": EndpointStatus.FAILED,
    "scaledToZero": EndpointStatus.SCALED_TO_ZERO,
}


class HuggingFaceBackend(ServingBackend):
    """Deploy models to HuggingFace Inference Endpoints."""

    name = "huggingface"

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("HF_TOKEN")

    def _get_api(self) -> Any:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for the HuggingFace backend. "
                "Install it with: pip install metaflow-serve[huggingface]"
            ) from exc
        return HfApi(token=self._token)

    def deploy(
        self,
        model_ref: ModelReference,
        endpoint_name: str,
        *,
        config: dict[str, Any] | None = None,
        service_cls: type | None = None,
        artifacts: Any | None = None,
    ) -> EndpointInfo:
        config = config or {}
        api = self._get_api()

        repository = config.get("repository")
        if not repository:
            raise ValueError(
                "HuggingFace backend requires 'repository' in config (e.g. 'user/model-name')."
            )

        if service_cls is not None:
            self._push_service_files(api, repository, service_cls, artifacts, model_ref)

        # Build kwargs for create_inference_endpoint.
        create_kwargs: dict[str, Any] = {
            "name": endpoint_name,
            "repository": repository,
            "framework": config.get("framework", "pytorch"),
            "task": config.get("task", model_ref.task_type or "text-generation"),
            "accelerator": config.get("accelerator", "gpu"),
            "instance_type": config.get("instance_type", "nvidia-a10g.xlarge"),
            "instance_size": config.get("instance_size", "x1"),
            "region": config.get("region", "us-east-1"),
            "vendor": config.get("vendor", "aws"),
            "type": config.get("endpoint_type", "protected"),
        }

        # Pass through optional fields.
        if "namespace" in config:
            create_kwargs["namespace"] = config["namespace"]
        if "min_replica" in config:
            create_kwargs["min_replica"] = config["min_replica"]
        if "max_replica" in config:
            create_kwargs["max_replica"] = config["max_replica"]
        if "revision" in config:
            create_kwargs["revision"] = config["revision"]
        if "custom_image" in config:
            create_kwargs["custom_image"] = config["custom_image"]

        endpoint = api.create_inference_endpoint(**create_kwargs)

        return EndpointInfo(
            name=endpoint.name,
            url=endpoint.url or "",
            backend=self.name,
            status=_HF_STATUS_MAP.get(endpoint.status, EndpointStatus.UNKNOWN),
            model_ref=model_ref,
            deploy_pathspec="",  # filled by decorator
            backend_id=endpoint.name,
            backend_metadata={
                "repository": repository,
                "region": create_kwargs["region"],
                "instance_type": create_kwargs["instance_type"],
                "accelerator": create_kwargs["accelerator"],
                "namespace": getattr(endpoint, "namespace", None),
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_status(self, endpoint_info: EndpointInfo) -> EndpointStatus:
        api = self._get_api()
        namespace = endpoint_info.backend_metadata.get("namespace")
        endpoint = api.get_inference_endpoint(
            endpoint_info.name,
            namespace=namespace,
        )
        return _HF_STATUS_MAP.get(endpoint.status, EndpointStatus.UNKNOWN)

    def delete(self, endpoint_info: EndpointInfo) -> None:
        api = self._get_api()
        namespace = endpoint_info.backend_metadata.get("namespace")
        api.delete_inference_endpoint(
            endpoint_info.name,
            namespace=namespace,
        )

    def _push_service_files(
        self,
        api: Any,
        repo_id: str,
        service_cls: type,
        artifacts: Any | None,
        model_ref: ModelReference,
    ) -> None:
        """Upload handler.py, requirements.txt, and pickled artifacts to the HF repo."""
        from ..codegen import generate_handler
        from ..codegen import generate_requirements
        from ..codegen import get_artifact_names

        artifact_names = get_artifact_names(service_cls)

        # Upload handler.py
        handler_src = generate_handler(service_cls, artifact_names)
        api.upload_file(
            path_or_fileobj=handler_src.encode(),
            path_in_repo="handler.py",
            repo_id=repo_id,
            repo_type="model",
        )

        # Upload requirements.txt — pull resolved deps from the Metaflow task's
        # conda/pypi environment so the HF endpoint gets the same packages.
        env_info = self._get_task_env_info(model_ref)
        reqs = generate_requirements(env_info=env_info)
        api.upload_file(
            path_or_fileobj=reqs.encode(),
            path_in_repo="requirements.txt",
            repo_id=repo_id,
            repo_type="model",
        )

        # Upload pickled artifacts — loaded from the Artifacts object
        # that Deployment already resolved from the Metaflow step.
        if artifacts is not None:
            flow_source = getattr(artifacts, "flow", None)
            for name in artifact_names:
                if flow_source is not None:
                    data = getattr(flow_source, name)
                    api.upload_file(
                        path_or_fileobj=pickle.dumps(data),
                        path_in_repo=f"artifacts/{name}.pkl",
                        repo_id=repo_id,
                        repo_type="model",
                    )

    @staticmethod
    def _get_task_env_info(model_ref: ModelReference) -> dict | None:
        """Load environment info from the Metaflow task that produced the model."""
        try:
            from metaflow import Task
        except ImportError:
            return None
        try:
            task = Task(model_ref.pathspec)
            return task.environment_info
        except Exception:
            # Task may not exist (tests, unknown pathspec, no conda env)
            return None

    def wait_for_ready(
        self,
        endpoint_info: EndpointInfo,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> EndpointInfo:
        """Use HF's built-in wait mechanism."""
        api = self._get_api()
        namespace = endpoint_info.backend_metadata.get("namespace")
        endpoint = api.get_inference_endpoint(
            endpoint_info.name,
            namespace=namespace,
        )
        endpoint.wait(timeout=timeout)
        endpoint_info.status = _HF_STATUS_MAP.get(endpoint.status, EndpointStatus.UNKNOWN)
        endpoint_info.url = endpoint.url or endpoint_info.url
        return endpoint_info
