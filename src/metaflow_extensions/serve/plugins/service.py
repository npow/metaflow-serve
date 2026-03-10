"""ServiceSpec base class and decorators for defining serving endpoints.

Users subclass ``ServiceSpec`` and decorate methods with ``@initialize``
and ``@endpoint`` to define a service that can be deployed via ``Deployment``.
"""

from __future__ import annotations

from typing import Any
from typing import Callable

from .artifacts import Artifacts

_INIT_TAG = "_serve_initialize"
_ENDPOINT_TAG = "_serve_endpoint"


def initialize(
    backend: str = "huggingface",
    cpu: int | None = None,
    memory: int | None = None,
    packages: dict[str, str] | None = None,
    **config: Any,
) -> Callable[..., Any]:
    """Mark a method as the service initializer.

    Parameters
    ----------
    backend : str
        Serving backend name (default ``"huggingface"``).
    cpu : int, optional
        Number of CPUs to request.
    memory : int, optional
        Memory in MB to request.
    packages : dict[str, str], optional
        Python packages required by the service, as ``{"name": "version_spec"}``.
        For example ``{"torch": ">=2.0", "numpy": ">=1.24"}``.
        These are merged with any packages resolved from the Metaflow step
        environment.
    **config
        Additional backend-specific configuration.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._serve_tag = _INIT_TAG  # type: ignore[attr-defined]
        func._serve_config = {  # type: ignore[attr-defined]
            "backend": backend,
            "cpu": cpu,
            "memory": memory,
            "packages": packages,
            **config,
        }
        return func

    return decorator


def endpoint(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Mark a method as a service endpoint handler.

    Can be used with or without parentheses::

        @endpoint
        def predict(self, request_dict): ...

        @endpoint(name="predict", description="Run inference")
        def predict(self, request_dict): ...

    Parameters
    ----------
    name : str, optional
        Endpoint name (defaults to the method name).
    description : str, optional
        Human-readable description (defaults to the docstring).
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        f._serve_tag = _ENDPOINT_TAG  # type: ignore[attr-defined]
        f._serve_endpoint_name = name or f.__name__  # type: ignore[attr-defined]
        f._serve_description = description or (f.__doc__ or "").strip()  # type: ignore[attr-defined]
        return f

    if func is not None:
        # @endpoint without parentheses
        return decorator(func)
    return decorator


def _find_tagged_methods(cls_or_instance: Any, tag: str) -> list[Callable[..., Any]]:
    """Find all methods on a class/instance tagged with a given _serve_tag."""
    results = []
    for attr_name in dir(cls_or_instance):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(cls_or_instance, attr_name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_serve_tag", None) == tag:
            results.append(attr)
    return results


def _get_init_config(service_cls: type[ServiceSpec]) -> dict[str, Any]:
    """Extract the @initialize config from a ServiceSpec class."""
    for attr_name in dir(service_cls):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(service_cls, attr_name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_serve_tag", None) == _INIT_TAG:
            return dict(getattr(attr, "_serve_config", {}))
    return {}


def _get_endpoints(service_cls: type[ServiceSpec]) -> list[dict[str, Any]]:
    """Extract endpoint metadata from a ServiceSpec class."""
    endpoints = []
    for attr_name in dir(service_cls):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(service_cls, attr_name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_serve_tag", None) == _ENDPOINT_TAG:
            endpoints.append(
                {
                    "name": getattr(attr, "_serve_endpoint_name", attr_name),
                    "description": getattr(attr, "_serve_description", ""),
                    "method": attr_name,
                }
            )
    return endpoints


class ServiceSpec:
    """Base class for defining a serving service.

    Subclass and add ``@initialize`` and ``@endpoint`` methods::

        class MyService(ServiceSpec):
            @initialize(backend="huggingface")
            def init(self):
                self.model = self.artifacts.flow.model

            @endpoint
            def predict(self, request_dict):
                return {"prediction": self.model.predict(request_dict["input"])}

    Specify Python dependencies via ``@initialize`` so the generated handler
    includes the right packages::

        class MyService(ServiceSpec):
            @initialize(
                backend="huggingface",
                packages={"torch": ">=2.0", "numpy": ">=1.24"},
            )
            def init(self):
                self.model = self.artifacts.flow.model

            @endpoint
            def predict(self, data):
                import torch
                tensor = torch.tensor(data["input"])
                return {"result": self.model(tensor).tolist()}

    When deploying from a Metaflow step that uses ``@conda``/``@pypi``,
    the step's resolved environment is used automatically.
    """

    def __init__(self, artifacts: Artifacts | None = None) -> None:
        self.artifacts = artifacts or Artifacts()
        # Auto-call the @initialize method if present
        init_methods = _find_tagged_methods(self, _INIT_TAG)
        if init_methods:
            init_methods[0]()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Validate that subclasses have at least one endpoint
        # (deferred to instantiation/deployment time for flexibility)
