"""ServeCard — endpoint status + model lineage visualization."""

from __future__ import annotations

import json
from typing import Any

from metaflow.cards import MetaflowCard


class ServeCard(MetaflowCard):
    """Render endpoint deployment status and model lineage."""

    type = "serve_card"
    ALLOW_USER_COMPONENTS = False
    RUNTIME_UPDATABLE = False

    def render(self, task: Any) -> str:
        raw = getattr(task.data, "endpoint", None)
        if raw is None:
            return _error_html(
                "No endpoint data found.",
                "The <code>@serve</code> decorator did not produce an "
                "<code>endpoint</code> artifact for this task.",
            )

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return _error_html("Invalid endpoint data.", "Could not parse endpoint JSON.")

        status = raw.get("status", "unknown")
        if isinstance(status, dict):
            status = status.get("value", status.get("_value_", "unknown"))
        name = raw.get("name", "unknown")
        url = raw.get("url", "")
        backend = raw.get("backend", "unknown")
        deploy_pathspec = raw.get("deploy_pathspec", "")
        created_at = raw.get("created_at", "")
        metadata = raw.get("backend_metadata", {})
        model_ref = raw.get("model_ref", {})

        status_color = _status_color(status)

        model_html = ""
        if model_ref:
            model_html = f"""
            <div class="section">
                <h3>Model Lineage</h3>
                <table>
                    <tr><td class="label">Flow</td><td>{_esc(model_ref.get("flow_name", ""))}</td></tr>
                    <tr><td class="label">Run</td><td>{_esc(model_ref.get("run_id", ""))}</td></tr>
                    <tr><td class="label">Step</td><td>{_esc(model_ref.get("step_name", ""))}</td></tr>
                    <tr><td class="label">Task</td><td>{_esc(model_ref.get("task_id", ""))}</td></tr>
                    <tr><td class="label">Artifact</td><td><code>{_esc(model_ref.get("artifact_name", ""))}</code></td></tr>
                    <tr><td class="label">Pathspec</td><td><code>{_esc(model_ref.get("pathspec", ""))}</code></td></tr>
                    {_framework_row(model_ref)}
                    {_task_type_row(model_ref)}
                </table>
            </div>
            """

        metadata_html = ""
        if metadata and not metadata.get("error"):
            rows = "".join(
                f'<tr><td class="label">{_esc(str(k))}</td><td>{_esc(str(v))}</td></tr>'
                for k, v in metadata.items()
                if v is not None
            )
            if rows:
                metadata_html = f"""
                <div class="section">
                    <h3>Backend Details</h3>
                    <table>{rows}</table>
                </div>
                """

        error_html = ""
        if metadata.get("error"):
            error_html = f"""
            <div class="section error-section">
                <h3>Error</h3>
                <pre>{_esc(str(metadata["error"]))}</pre>
                {_traceback_details(metadata)}
            </div>
            """

        return f"""<!DOCTYPE html>
<html>
<head><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #fafafa; color: #333; }}
.card {{ max-width: 700px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 24px; }}
h2 {{ margin: 0 0 16px 0; font-size: 20px; }}
h3 {{ margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase; color: #666; letter-spacing: 0.5px; }}
.header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; color: white; }}
.url {{ font-size: 13px; color: #0066cc; word-break: break-all; }}
.section {{ margin-bottom: 20px; padding: 16px; background: #f8f9fa; border-radius: 6px; }}
.error-section {{ background: #fff5f5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
td {{ padding: 4px 8px; }}
td.label {{ font-weight: 600; color: #555; width: 120px; white-space: nowrap; }}
pre {{ font-size: 12px; overflow-x: auto; margin: 8px 0; white-space: pre-wrap; }}
code {{ background: #eee; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
.meta {{ font-size: 12px; color: #888; margin-top: 16px; }}
</style></head>
<body>
<div class="card">
    <div class="header">
        <h2>{_esc(name)}</h2>
        <span class="badge" style="background:{status_color}">{_esc(status)}</span>
    </div>
    {f'<div class="url"><a href="{_esc(url)}">{_esc(url)}</a></div>' if url else ""}
    {model_html}
    {metadata_html}
    {error_html}
    <div class="meta">
        Backend: {_esc(backend)}
        {f" &middot; Deployed: {_esc(created_at)}" if created_at else ""}
        {f" &middot; Deploy pathspec: <code>{_esc(deploy_pathspec)}</code>" if deploy_pathspec else ""}
    </div>
</div>
</body>
</html>"""


def _framework_row(model_ref: dict[str, Any]) -> str:
    fw = model_ref.get("framework")
    if not fw:
        return ""
    return f'<tr><td class="label">Framework</td><td>{_esc(fw)}</td></tr>'


def _task_type_row(model_ref: dict[str, Any]) -> str:
    tt = model_ref.get("task_type")
    if not tt:
        return ""
    return f'<tr><td class="label">Task Type</td><td>{_esc(tt)}</td></tr>'


def _traceback_details(metadata: dict[str, Any]) -> str:
    tb = metadata.get("traceback")
    if not tb:
        return ""
    return f"<details><summary>Traceback</summary><pre>{_esc(tb)}</pre></details>"


def _status_color(status: str) -> str:
    s = status.lower()
    if s == "running":
        return "#22c55e"
    if s in ("pending", "initializing", "updating", "scaled_to_zero"):
        return "#f59e0b"
    if s == "failed":
        return "#ef4444"
    return "#6b7280"


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _error_html(title: str, detail: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }}
.error {{ max-width: 500px; margin: 40px auto; text-align: center; color: #666; }}
h2 {{ color: #333; }}
</style></head>
<body>
<div class="error"><h2>{title}</h2><p>{detail}</p></div>
</body>
</html>"""
