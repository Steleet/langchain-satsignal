"""Offline tests for SatsignalCallbackHandler.

Uses the ``transport=`` mock plumbed through SatsignalConfig so the entire
chain start -> decide -> chain end flow runs without the network.
"""
from __future__ import annotations

import json
from typing import List, Tuple
from uuid import uuid4

import pytest

from langchain_satsignal import (
    APIError,
    SatsignalCallbackHandler,
    SatsignalConfig,
)


def make_mock_transport(canned_status: int = 200, canned_body: bytes = None):
    """Returns (transport, captured_calls). Each call recorded as a tuple
    of (method, url, headers, body_dict, timeout)."""
    captured: List[Tuple] = []
    if canned_body is None:
        canned_body = (
            b'{"txid":"deadbeefcafebabe","bundle_id":"abcd1234",'
            b'"root":"f00dface","leaf_count":1}'
        )

    def transport(method, url, headers, body, timeout):
        try:
            body_dict = json.loads(body.decode("utf-8"))
        except Exception:
            body_dict = None
        captured.append((method, url, dict(headers), body_dict, timeout))
        return canned_status, canned_body

    return transport, captured


class FakeAgentAction:
    def __init__(self, tool, tool_input, log=""):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log


# ---- happy path: chain_start -> agent_action x N -> chain_end ----


def test_full_chain_anchors_policy_decisions_and_manifest():
    transport, captured = make_mock_transport()
    cfg = SatsignalConfig(
        api_key="sk_test",
        matter_slug="agent-runs",
        transport=transport,
        write_handoff=False,
    )
    h = SatsignalCallbackHandler(cfg)
    run_id = uuid4()

    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "what is 2+2?"},
        run_id=run_id,
    )

    nested_run_id = uuid4()
    h.on_agent_action(
        FakeAgentAction("calculator", {"expr": "2+2"}, log="thought 1"),
        run_id=nested_run_id,
        parent_run_id=run_id,
    )
    h.on_agent_action(
        FakeAgentAction("calculator", {"expr": "validate"}, log="thought 2"),
        run_id=uuid4(),
        parent_run_id=run_id,
    )

    h.on_chain_end(
        outputs={"output": "4"}, run_id=run_id,
    )

    # 1 policy + 2 decisions + 1 manifest = 4 anchor POSTs
    assert len(captured) == 4
    categories = [c[3]["category"] for c in captured]
    assert categories == [
        "policy_snapshot",
        "commitment",
        "commitment",
        "evidence_bundle",
    ]
    # All four hit /api/v1/anchors on the configured base_url
    assert all(c[1].endswith("/api/v1/anchors") for c in captured)
    # session_id propagates from the run_id
    assert captured[0][3]["session_id"] == str(run_id)


def test_nested_chains_dont_double_anchor():
    transport, captured = make_mock_transport()
    cfg = SatsignalConfig(
        api_key="sk_test", matter_slug="m", transport=transport,
    )
    h = SatsignalCallbackHandler(cfg)
    top = uuid4()
    nested = uuid4()

    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "go"},
        run_id=top,
    )
    # Nested chain start (LangChain wraps tool calls in sub-chains).
    h.on_chain_start(
        serialized={"name": "Tool"},
        inputs={"input": "nested"},
        run_id=nested,
        parent_run_id=top,
    )
    # One real decision between the nested start/end.
    h.on_agent_action(
        FakeAgentAction("calc", "2+2"),
        run_id=uuid4(), parent_run_id=top,
    )
    # Nested chain end — should NOT anchor a manifest.
    h.on_chain_end(outputs={"x": 1}, run_id=nested, parent_run_id=top)
    # Top-level chain end — anchors the manifest.
    h.on_chain_end(outputs={"output": "done"}, run_id=top)

    # Expected exactly: 1 policy + 1 decision + 1 manifest = 3 anchors.
    # If nested start/end were emitting their own anchors, we'd see > 3.
    cats = [c[3]["category"] for c in captured]
    assert cats == ["policy_snapshot", "commitment", "evidence_bundle"]


def test_decide_on_tool_start():
    transport, captured = make_mock_transport()
    cfg = SatsignalConfig(
        api_key="sk_test", matter_slug="m",
        transport=transport, decide_on="tool_start",
    )
    h = SatsignalCallbackHandler(cfg)
    top = uuid4()

    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "go"},
        run_id=top,
    )
    # agent_action should NOT fire decide when decide_on=tool_start
    h.on_agent_action(
        FakeAgentAction("t", "x"), run_id=uuid4(), parent_run_id=top,
    )
    h.on_tool_start(
        serialized={"name": "calc"}, input_str="2+2",
        run_id=uuid4(), parent_run_id=top,
    )
    h.on_chain_end(outputs={"output": "4"}, run_id=top)

    cats = [c[3]["category"] for c in captured]
    assert cats == ["policy_snapshot", "commitment", "evidence_bundle"]
    # The one commitment was for the tool, not the action
    assert captured[1][3]["label"].startswith("tool:calc")


def test_decide_on_chain_end_emits_one_decision():
    transport, captured = make_mock_transport()
    cfg = SatsignalConfig(
        api_key="sk_test", matter_slug="m",
        transport=transport, decide_on="chain_end",
    )
    h = SatsignalCallbackHandler(cfg)
    top = uuid4()

    h.on_chain_start(
        serialized={"name": "AgentExecutor"},
        inputs={"input": "x"}, run_id=top,
    )
    h.on_agent_action(
        FakeAgentAction("t", "x"), run_id=uuid4(), parent_run_id=top,
    )
    h.on_chain_end(outputs={"output": "done"}, run_id=top)

    cats = [c[3]["category"] for c in captured]
    assert cats == ["policy_snapshot", "commitment", "evidence_bundle"]
    assert captured[1][3]["label"] == "chain:final"


# ---- failure modes ----


def test_fail_open_swallows_api_errors():
    """API returns 401 throughout; agent execution should still complete."""
    def transport_401(method, url, headers, body, timeout):
        return 401, b'{"error":"invalid_key"}'

    cfg = SatsignalConfig(
        api_key="sk_test", matter_slug="m",
        transport=transport_401, fail_open=True,
    )
    h = SatsignalCallbackHandler(cfg)
    top = uuid4()

    # No exception should escape any of these calls
    h.on_chain_start(
        serialized={"name": "x"}, inputs={"input": "y"}, run_id=top,
    )
    h.on_agent_action(
        FakeAgentAction("t", "x"), run_id=uuid4(), parent_run_id=top,
    )
    h.on_chain_end(outputs={"output": "done"}, run_id=top)


def test_fail_closed_propagates_api_errors():
    def transport_429(method, url, headers, body, timeout):
        return 429, b'{"error":"quota_exceeded"}'

    cfg = SatsignalConfig(
        api_key="sk_test", matter_slug="m",
        transport=transport_429, fail_open=False,
    )
    h = SatsignalCallbackHandler(cfg)
    top = uuid4()

    with pytest.raises(APIError) as excinfo:
        h.on_chain_start(
            serialized={"name": "x"}, inputs={"input": "y"}, run_id=top,
        )
    assert excinfo.value.status == 429


# ---- config validation ----


def test_invalid_decide_on_rejected():
    with pytest.raises(ValueError, match="decide_on"):
        SatsignalConfig(api_key="sk", matter_slug="m", decide_on="bogus")


# ---- error path during chain ----


def test_on_chain_error_skips_manifest_but_cleans_up():
    transport, captured = make_mock_transport()
    cfg = SatsignalConfig(
        api_key="sk_test", matter_slug="m", transport=transport,
    )
    h = SatsignalCallbackHandler(cfg)
    top = uuid4()

    h.on_chain_start(
        serialized={"name": "x"}, inputs={"input": "y"}, run_id=top,
    )
    h.on_agent_action(
        FakeAgentAction("t", "x"), run_id=uuid4(), parent_run_id=top,
    )
    h.on_chain_error(RuntimeError("kaboom"), run_id=top)

    # policy + decision but NO manifest (chain failed)
    cats = [c[3]["category"] for c in captured]
    assert cats == ["policy_snapshot", "commitment"]
    # session was cleaned up — re-using the run_id shouldn't anchor again
    h.on_chain_end(outputs={}, run_id=top)
    cats_after = [c[3]["category"] for c in captured]
    assert cats_after == cats  # no new anchors after the error
