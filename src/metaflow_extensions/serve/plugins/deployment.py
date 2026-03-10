"""Deployment class for deploying ServiceSpec services.

Provides a fluent API for deploying, auditing, and promoting services::

    Deployment(MyService, step=self.train, config={...})
        .audit("predict", payload={"input": [1, 2, 3]})
        .promote()
"""

from __future__ import annotations

import json
from typing import Any
from typing import Callable

from .artifacts import Artifacts
from .backends import get_backend
from .backends.backend import EndpointInfo
from .backends.backend import EndpointStatus
from .backends.backend import ModelReference
from .backends.backend import asdict_with_enums
from .service import ServiceSpec
from .service import _get_endpoints
from .service import _get_init_config


class DeploymentError(Exception):
    """Raised when deployment fails."""


class AuditError(Exception):
    """Raised when an audit check fails."""


class Deployment:
    """Deploy a ServiceSpec to a serving backend.

    Parameters
    ----------
    service_cls : type[ServiceSpec]
        The service class to deploy.
    step : object, optional
        A Metaflow step reference to load artifacts from.
    flows : dict, optional
        Named flow/step references for multi-flow artifact access.
    config : dict, optional
        Backend-specific configuration (merged with @initialize config).
    timeout : int
        Seconds to wait for the endpoint to become ready (default 300).
    """

    def __init__(
        self,
        service_cls: type[ServiceSpec],
        step: Any = None,
        flows: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> None:
        self._service_cls = service_cls
        self._promoted = False

        # Resolve artifacts
        if step is not None:
            self._artifacts = Artifacts.from_step(step)
        elif flows is not None:
            self._artifacts = Artifacts.from_flows(flows)
        else:
            self._artifacts = Artifacts()

        # Get backend config from @initialize, merge with explicit config
        init_config = _get_init_config(service_cls)
        merged_config = {**init_config, **(config or {})}

        # Extract backend name before passing config to backend
        backend_name = merged_config.pop("backend", "huggingface")
        # Remove non-backend keys
        merged_config.pop("cpu", None)
        merged_config.pop("memory", None)
        merged_config.pop("packages", None)

        # Build model reference from step context
        self._model_ref = self._build_model_ref(step)

        # Deploy via backend
        backend = get_backend(backend_name)
        self._backend = backend
        self._backend_name = backend_name
        self._config = merged_config

        endpoint_name = merged_config.pop("endpoint_name", None) or service_cls.__name__
        self._endpoint_info = backend.deploy(
            self._model_ref,
            endpoint_name,
            config=merged_config if merged_config else None,
            service_cls=self._service_cls,
            artifacts=self._artifacts,
        )

        # Wait for ready
        if timeout > 0:
            self._endpoint_info = backend.wait_for_ready(self._endpoint_info, timeout=timeout)

    def audit(
        self,
        endpoint_name: str,
        payload: dict[str, Any] | None = None,
        expected: Any | None = None,
        check_func: Callable[[Any], bool] | None = None,
    ) -> Deployment:
        """Run an audit against a deployed endpoint.

        Parameters
        ----------
        endpoint_name : str
            Name of the endpoint to test.
        payload : dict, optional
            Request payload to send.
        expected : Any, optional
            Expected response (compared with ==).
        check_func : callable, optional
            Custom check function ``(response) -> bool``.

        Returns
        -------
        Deployment
            Self, for fluent chaining.
        """
        if self._endpoint_info.status != EndpointStatus.RUNNING:
            raise AuditError(f"Cannot audit: endpoint status is {self._endpoint_info.status.value}")

        # Verify the endpoint exists on the service class
        endpoints = _get_endpoints(self._service_cls)
        endpoint_names = [ep["name"] for ep in endpoints]
        if endpoint_name not in endpoint_names:
            raise AuditError(
                f"Endpoint '{endpoint_name}' not found on {self._service_cls.__name__}. "
                f"Available: {endpoint_names}"
            )

        # Call the endpoint URL
        url = self._endpoint_info.url
        if url:
            try:
                import urllib.request

                req_data = json.dumps(payload or {}).encode()
                headers: dict[str, str] = {"Content-Type": "application/json"}

                # Add auth headers for backends that need them (e.g. HF protected endpoints)
                token = self._get_backend_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                req = urllib.request.Request(  # noqa: S310
                    url,
                    data=req_data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                    response = json.loads(resp.read().decode())
            except Exception as exc:
                raise AuditError(f"Audit request failed: {exc}") from exc

            if expected is not None and response != expected:
                raise AuditError(f"Audit failed: expected {expected!r}, got {response!r}")
            if check_func is not None and not check_func(response):
                raise AuditError(f"Audit check function returned False for response: {response!r}")

        return self

    def promote(self) -> Deployment:
        """Mark the deployment as promoted/production.

        Returns
        -------
        Deployment
            Self, for fluent chaining.
        """
        self._promoted = True
        return self

    @property
    def version(self) -> EndpointInfo:
        """Return the endpoint info for this deployment."""
        return self._endpoint_info

    @property
    def endpoint_url(self) -> str:
        """Return the deployed endpoint URL."""
        return self._endpoint_info.url

    def as_dict(self) -> dict[str, Any]:
        """Serialize the deployment info for storage as a Metaflow artifact."""
        result: dict[str, Any] = asdict_with_enums(self._endpoint_info)
        return result

    def _get_backend_token(self) -> str | None:
        """Return an auth token for the backend, if available."""
        import os

        # Check backend instance for a token attribute (e.g. HuggingFaceBackend._token)
        token = getattr(self._backend, "_token", None)
        if token:
            return str(token)
        # Fallback to well-known env vars per backend
        env_vars = {
            "huggingface": "HF_TOKEN",
        }
        env_var = env_vars.get(self._backend_name)
        if env_var:
            return os.environ.get(env_var)
        return None

    def _build_model_ref(self, step: Any) -> ModelReference:
        """Build a ModelReference from a step context."""
        if step is None:
            return ModelReference(
                flow_name="unknown",
                run_id="0",
                step_name="unknown",
                task_id="0",
                artifact_name="unknown",
                pathspec="unknown/0/unknown/0",
            )

        # Try to extract pathspec info from the step
        flow_name = str(
            getattr(step, "flow_name", None)
            or getattr(getattr(step, "__class__", None), "__name__", "unknown")
        )
        run_id = str(getattr(step, "run_id", "0"))
        step_name = str(getattr(step, "step_name", None) or getattr(step, "_name", "unknown"))
        task_id = str(getattr(step, "task_id", "0"))

        # Handle Metaflow Step objects
        if hasattr(step, "pathspec"):
            pathspec = step.pathspec
        else:
            pathspec = f"{flow_name}/{run_id}/{step_name}/{task_id}"

        return ModelReference(
            flow_name=flow_name,
            run_id=run_id,
            step_name=step_name,
            task_id=task_id,
            artifact_name="model",
            pathspec=pathspec,
        )

    def __repr__(self) -> str:
        return (
            f"Deployment(service={self._service_cls.__name__!r}, "
            f"endpoint={self._endpoint_info.name!r}, "
            f"status={self._endpoint_info.status.value!r})"
        )
