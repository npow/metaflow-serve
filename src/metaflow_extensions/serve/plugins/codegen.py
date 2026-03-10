"""Generate HuggingFace-compatible handler.py from a ServiceSpec class."""

from __future__ import annotations

import ast
import inspect
import textwrap


def get_artifact_names(service_cls: type) -> list[str]:
    """Extract artifact names referenced via ``self.artifacts.flow.<name>`` in @initialize."""

    from .service import _INIT_TAG

    # Find the @initialize method
    init_method = None
    for attr_name in dir(service_cls):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(service_cls, attr_name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_serve_tag", None) == _INIT_TAG:
            init_method = attr
            break

    if init_method is None:
        return []

    source = textwrap.dedent(inspect.getsource(init_method))
    tree = ast.parse(source)
    names: list[str] = []

    for node in ast.walk(tree):
        # Match: self.artifacts.flow.<name>
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Attribute)
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == "self"
            and node.value.value.attr == "artifacts"
            and node.value.attr == "flow"
        ) and node.attr not in names:
            names.append(node.attr)

    return names


def generate_handler(service_cls: type, artifact_names: list[str]) -> str:
    """Generate a ``handler.py`` string with an ``EndpointHandler`` class.

    The generated module embeds the user's ServiceSpec source and wraps it
    with the HuggingFace custom handler contract.
    """
    from .service import _ENDPOINT_TAG

    # Get the user's ServiceSpec source
    service_source = textwrap.dedent(inspect.getsource(service_cls))

    # Find the first @endpoint method name
    endpoint_method = "predict"
    for attr_name in dir(service_cls):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(service_cls, attr_name)
        except AttributeError:
            continue
        if callable(attr) and getattr(attr, "_serve_tag", None) == _ENDPOINT_TAG:
            endpoint_method = attr_name
            break

    artifact_names_repr = repr(artifact_names)

    return f'''\
"""Auto-generated HuggingFace custom handler."""

import pickle
from pathlib import Path
from types import SimpleNamespace


# --- Decorators (standalone stubs so the ServiceSpec class can be defined) ---

def initialize(**kwargs):
    """Stub @initialize decorator."""
    def decorator(func):
        func._serve_tag = "_serve_initialize"
        func._serve_config = kwargs
        return func
    return decorator


def endpoint(func=None, *, name=None, description=None):
    """Stub @endpoint decorator."""
    def decorator(f):
        f._serve_tag = "_serve_endpoint"
        f._serve_endpoint_name = name or f.__name__
        f._serve_description = description or ""
        return f
    if func is not None:
        return decorator(func)
    return decorator


# --- ServiceSpec base (minimal, for standalone use) ---

class ServiceSpec:
    """Minimal base class for standalone handler execution."""

    def __init__(self, artifacts=None):
        self.artifacts = artifacts or SimpleNamespace()
        # Auto-call @initialize
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(self, attr_name)
            except AttributeError:
                continue
            if callable(attr) and getattr(attr, "_serve_tag", None) == "_serve_initialize":
                attr()
                break


# --- User ServiceSpec (embedded source) ---

{service_source}

# --- HuggingFace EndpointHandler ---

class EndpointHandler:
    def __init__(self, path=""):
        artifacts_dir = Path(path) / "artifacts" if path else Path("artifacts")
        loaded = {{}}
        for name in {artifact_names_repr}:
            p = artifacts_dir / f"{{name}}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    loaded[name] = pickle.load(f)
        flow_ns = SimpleNamespace(**loaded)
        artifacts_obj = SimpleNamespace(flow=flow_ns)
        self.service = {service_cls.__name__}(artifacts=artifacts_obj)

    def __call__(self, data):
        return self.service.{endpoint_method}(data)
'''


def extract_requirements_from_env_info(env_info: dict) -> list[str]:
    """Extract pip package specs from a Metaflow task's ``environment_info``.

    The environment info (from ``task.environment_info``) has the structure::

        {"conda": [...], "pypi": [{"url": "https://.../<pkg>.whl"}, ...]}

    We extract package names and versions from the wheel URLs in the ``"pypi"``
    key.  Wheel filenames follow PEP 427: ``{name}-{ver}(-...)-....whl``.
    """
    deps: list[str] = []
    pypi_packages = env_info.get("pypi", [])
    for pkg in pypi_packages:
        url = pkg.get("url", "") if isinstance(pkg, dict) else ""
        if not url:
            continue
        # Extract filename from URL
        filename = url.rsplit("/", 1)[-1]
        if filename.endswith(".whl"):
            # PEP 427: {name}-{version}(-{build})-{python}-{abi}-{platform}.whl
            parts = filename.split("-")
            if len(parts) >= 2:
                name = parts[0].replace("_", "-")
                version = parts[1]
                deps.append(f"{name}=={version}")
        elif filename.endswith(".tar.gz"):
            # sdist: {name}-{version}.tar.gz
            base = filename[: -len(".tar.gz")]
            parts = base.rsplit("-", 1)
            if len(parts) == 2:
                name = parts[0].replace("_", "-")
                version = parts[1]
                deps.append(f"{name}=={version}")
    return deps


def generate_requirements(
    env_info: dict | None = None,
    extra_deps: list[str] | None = None,
    packages: dict[str, str] | None = None,
) -> str:
    """Generate a ``requirements.txt`` string.

    Parameters
    ----------
    env_info : dict, optional
        Metaflow task ``environment_info`` dict.  When provided, the resolved
        pip packages from the training task are included so the HF endpoint
        gets the same environment.
    extra_deps : list[str], optional
        Additional pip requirement strings to append.
    packages : dict[str, str], optional
        Packages from ``@initialize(packages={...})``.  Keys are package
        names, values are version specifiers (e.g. ``{"torch": ">=2.0"}``).
        These take precedence over env_info for the same package name.
    """
    deps: list[str] = []
    if env_info is not None:
        deps.extend(extract_requirements_from_env_info(env_info))
    if not deps:
        # Fallback: at minimum we need metaflow-serve
        deps.append("metaflow-serve")
    if extra_deps:
        deps.extend(extra_deps)
    # Merge explicit packages — override any env_info entry for the same name
    if packages:

        def _pkg_name(dep: str) -> str:
            for sep in ("==", ">=", "<=", "!="):
                dep = dep.split(sep)[0]
            return dep

        existing_names = {_pkg_name(d) for d in deps}
        for name, version in packages.items():
            if name in existing_names:
                # Remove the existing entry so the explicit one wins
                deps = [d for d in deps if not d.startswith(name)]
            if version:
                deps.append(f"{name}{version}")
            else:
                deps.append(name)
    return "\n".join(deps) + "\n"
