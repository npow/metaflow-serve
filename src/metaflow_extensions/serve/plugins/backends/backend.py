"""Serving backend ABC and core data types."""

from __future__ import annotations

import dataclasses
import enum
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any


class EndpointStatus(enum.Enum):
    """Status of a deployed endpoint."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    UPDATING = "updating"
    FAILED = "failed"
    SCALED_TO_ZERO = "scaled_to_zero"
    DELETED = "deleted"
    UNKNOWN = "unknown"


@dataclass
class ModelReference:
    """Reference to the Metaflow artifact that produced the model."""

    flow_name: str
    run_id: str
    step_name: str
    task_id: str
    artifact_name: str
    pathspec: str  # "Flow/Run/Step/Task"
    framework: str | None = None  # "transformers", "sklearn"
    task_type: str | None = None  # "text-generation", etc.


@dataclass
class EndpointInfo:
    """Full endpoint descriptor with bidirectional lineage."""

    name: str
    url: str
    backend: str  # "huggingface", "modal"
    status: EndpointStatus
    model_ref: ModelReference
    deploy_pathspec: str  # pathspec of the deployment task
    backend_id: str | None = None
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


def asdict_with_enums(obj: Any) -> Any:
    """Convert dataclass to dict, serializing enums to their values."""
    return _convert_enums(dataclasses.asdict(obj))


def _convert_enums(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _convert_enums(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_enums(v) for v in obj]
    return obj


class ServingBackend(ABC):
    """Abstract base class for serving backends."""

    name: str = ""

    @abstractmethod
    def deploy(
        self,
        model_ref: ModelReference,
        endpoint_name: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> EndpointInfo:
        """Deploy a model and return endpoint info."""

    @abstractmethod
    def get_status(self, endpoint_info: EndpointInfo) -> EndpointStatus:
        """Query the current status of an endpoint."""

    @abstractmethod
    def delete(self, endpoint_info: EndpointInfo) -> None:
        """Delete a deployed endpoint."""

    def wait_for_ready(
        self,
        endpoint_info: EndpointInfo,
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> EndpointInfo:
        """Poll until endpoint is running or timeout is reached."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_status(endpoint_info)
            endpoint_info.status = status
            if status == EndpointStatus.RUNNING:
                return endpoint_info
            if status == EndpointStatus.FAILED:
                raise RuntimeError(f"Endpoint {endpoint_info.name!r} failed to start.")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Endpoint {endpoint_info.name!r} did not become ready within {timeout}s."
        )
