"""HuggingFace Inference Endpoints backend."""

from __future__ import annotations

import os
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
    ) -> EndpointInfo:
        config = config or {}
        api = self._get_api()

        repository = config.get("repository")
        if not repository:
            raise ValueError(
                "HuggingFace backend requires 'repository' in config (e.g. 'user/model-name')."
            )

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
