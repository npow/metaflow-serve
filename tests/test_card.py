"""Tests for ServeCard rendering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import make_endpoint_info
from conftest import make_model_ref

from metaflow_extensions.serve.plugins.backends.backend import EndpointStatus
from metaflow_extensions.serve.plugins.backends.backend import asdict_with_enums
from metaflow_extensions.serve.plugins.cards.serve_card.card import ServeCard


@pytest.fixture
def card():
    return ServeCard()


def _make_task(endpoint_dict):
    """Create a mock task with endpoint data accessible via task.data.endpoint."""
    task = MagicMock()
    task.data = MagicMock()
    task.data.endpoint = endpoint_dict
    return task


class TestServeCard:
    def test_type(self, card):
        assert card.type == "serve_card"

    def test_render_no_data(self, card):
        task = MagicMock()
        task.data = MagicMock(spec=[])  # no endpoint attr
        html = card.render(task)
        assert "No endpoint data found" in html

    def test_render_running_endpoint(self, card):
        info = make_endpoint_info()
        task = _make_task(asdict_with_enums(info))

        html = card.render(task)
        assert "test-endpoint" in html
        assert "running" in html
        assert "https://api.example.com/v1/test-endpoint" in html
        assert "TestFlow" in html  # model lineage
        assert "#22c55e" in html  # green color for running

    def test_render_failed_endpoint(self, card):
        info = make_endpoint_info(
            status=EndpointStatus.FAILED,
            url="",
            backend_metadata={"error": "Something broke"},
        )
        task = _make_task(asdict_with_enums(info))

        html = card.render(task)
        assert "failed" in html
        assert "Something broke" in html
        assert "#ef4444" in html  # red color for failed

    def test_render_pending_endpoint(self, card):
        info = make_endpoint_info(status=EndpointStatus.PENDING)
        task = _make_task(asdict_with_enums(info))

        html = card.render(task)
        assert "pending" in html
        assert "#f59e0b" in html  # yellow color

    def test_render_with_framework(self, card):
        ref = make_model_ref(framework="sklearn", task_type="classification")
        info = make_endpoint_info(model_ref=ref)
        task = _make_task(asdict_with_enums(info))

        html = card.render(task)
        assert "sklearn" in html
        assert "classification" in html

    def test_render_backend_metadata(self, card):
        info = make_endpoint_info(backend_metadata={"region": "us-east-1", "instance_type": "a10g"})
        task = _make_task(asdict_with_enums(info))

        html = card.render(task)
        assert "us-east-1" in html
        assert "a10g" in html

    def test_html_escaping(self, card):
        info = make_endpoint_info(name="<script>alert(1)</script>")
        task = _make_task(asdict_with_enums(info))

        html = card.render(task)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
