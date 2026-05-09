"""LangChain callback that anchors agent decisions on BSV via Satsignal.

Top-level chain start  -> policy snapshot anchor.
Per-decision (configurable: agent_action / tool_start / chain_end) -> commit-reveal anchor.
Top-level chain end    -> evidence-bundle manifest anchor + handoff.json.

Nested chains (parent_run_id != None) don't double-anchor; they're absorbed
into the top-level session keyed by the top-level run_id.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from . import _anchor


@dataclass
class SatsignalConfig:
    """Configuration for SatsignalCallbackHandler.

    api_key:      Bearer key from a Satsignal workspace. A human must
                  mint this once with anchors:create + receipts:read scopes.
    matter_slug:  Workspace matter to anchor under. Must exist server-side.
    base_url:     API base URL. Default is the production endpoint.
    fail_open:    If True (default), API errors are logged to stderr and
                  agent execution continues; the audit trail will have gaps
                  but the agent isn't blocked. If False, APIError is
                  re-raised — right for regulated workflows where
                  unanchored decisions are unacceptable.
    decide_on:    "agent_action" (default, one anchor per ReAct step),
                  "tool_start"   (one anchor per tool call), or
                  "chain_end"    (one anchor per top-level chain — the
                                  manifest still binds the set, but only
                                  the final outputs get a per-decision
                                  commitment).
    transport:    Optional callable for unit tests. Signature:
                  (method, url, headers, body_bytes, timeout) -> (status, response_bytes).
                  Default uses urllib; tests pass a mock so CI runs offline.
    session_id:   Override the auto-generated session_id (defaults to the
                  LangChain top-level run_id stringified).
    write_handoff: If True, writes handoff.json on top-level chain end.
                  Default False because handoff.json contains decision
                  envelopes (i.e. tool inputs / outputs) which may include
                  user-supplied secrets.
    handoff_path: Path for handoff.json when write_handoff=True.
    agent_name / agent_version: Optional metadata baked into the policy
                  snapshot's agent block.
    """

    api_key: str
    matter_slug: str
    base_url: str = "https://app.satsignal.cloud"
    fail_open: bool = True
    decide_on: str = "agent_action"
    transport: Optional[Any] = None
    session_id: Optional[str] = None
    write_handoff: bool = False
    handoff_path: Optional[str] = "handoff.json"
    agent_name: Optional[str] = None
    agent_version: Optional[str] = None

    def __post_init__(self):
        if self.decide_on not in ("agent_action", "tool_start", "chain_end"):
            raise ValueError(
                f"decide_on must be one of agent_action/tool_start/chain_end; "
                f"got {self.decide_on!r}"
            )


class SatsignalCallbackHandler(BaseCallbackHandler):
    """LangChain BaseCallbackHandler that anchors decisions via Satsignal.

    One handler instance can be reused across multiple top-level chain
    runs; sessions are keyed per top-level ``run_id`` so concurrent
    chains don't share state. Within a single chain, callbacks fire
    serially by LangChain's contract.

    Sync-only: LangChain dispatches sync callbacks in a thread for
    async chains, so this works for both. An async-native handler is
    deferred to a later release.
    """

    def __init__(self, config: SatsignalConfig):
        self.config = config
        self._sessions: Dict[UUID, _anchor.Session] = {}
        self._lock = threading.Lock()

    # ---- internal helpers ----

    def _is_top_level(self, parent_run_id: Optional[UUID]) -> bool:
        return parent_run_id is None

    def _top_session_for(self, run_id: UUID) -> Optional[_anchor.Session]:
        # Within a single chain, the top-level run_id is the only key
        # in self._sessions for that chain. Decisions fire under nested
        # run_ids; we anchor against whichever session is active.
        # Since one handler may serve concurrent chains, we walk by
        # parent chain — but LangChain doesn't pass us the chain root
        # in on_agent_action, so we maintain the lookup here.
        with self._lock:
            return self._sessions.get(run_id)

    def _try(self, fn, *args, **kwargs):
        """Run fn with fail_open semantics. Returns the fn result, or
        None if fail_open swallowed an APIError."""
        try:
            return fn(*args, **kwargs)
        except _anchor.APIError as e:
            if self.config.fail_open:
                sys.stderr.write(
                    f"[satsignal] anchor failed: {e}; "
                    f"continuing (fail_open=True)\n"
                )
                return None
            raise

    def _build_policy_kwargs(
        self, serialized: Any, inputs: Any
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if isinstance(inputs, dict):
            text = inputs.get("input") or inputs.get("question")
            if text is None and inputs:
                text = json.dumps(inputs, sort_keys=True, default=str)
            if text:
                kwargs["user_instruction_text"] = str(text)
        elif inputs is not None:
            kwargs["user_instruction_text"] = str(inputs)
        if serialized:
            try:
                kwargs["system_policy_text"] = json.dumps(
                    serialized, sort_keys=True, default=str
                )
            except (TypeError, ValueError):
                kwargs["system_policy_text"] = repr(serialized)
        return kwargs

    # ---- LangChain hooks ----

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._is_top_level(parent_run_id):
            return
        session = _anchor.Session(
            api_key=self.config.api_key,
            matter_slug=self.config.matter_slug,
            base_url=self.config.base_url,
            session_id=self.config.session_id or str(run_id),
            handoff_path=(
                self.config.handoff_path
                if self.config.write_handoff
                else None
            ),
            agent_name=self.config.agent_name,
            agent_version=self.config.agent_version,
            commit_each=True,
            write_handoff=self.config.write_handoff,
            transport=self.config.transport,
        )
        session.__enter__()
        with self._lock:
            self._sessions[run_id] = session

        policy_kwargs = self._build_policy_kwargs(serialized, inputs)
        if policy_kwargs:
            self._try(session.policy, **policy_kwargs)

    def on_agent_action(
        self,
        action,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self.config.decide_on != "agent_action":
            return
        session = self._lookup_top_session(run_id, parent_run_id)
        if session is None:
            return
        tool = getattr(action, "tool", None) or "unknown"
        payload = {
            "tool": tool,
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self._try(session.decide, f"action:{tool}", payload)

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if self.config.decide_on != "tool_start":
            return
        session = self._lookup_top_session(run_id, parent_run_id)
        if session is None:
            return
        tool_name = "unknown"
        if isinstance(serialized, dict):
            tool_name = serialized.get("name") or "unknown"
        payload = {"tool": tool_name, "input": input_str}
        self._try(session.decide, f"tool:{tool_name}", payload)

    def on_chain_end(
        self,
        outputs,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._is_top_level(parent_run_id):
            return
        with self._lock:
            session = self._sessions.pop(run_id, None)
        if session is None:
            return

        if self.config.decide_on == "chain_end":
            payload = (
                {"outputs": outputs}
                if isinstance(outputs, dict)
                else {"output": str(outputs)}
            )
            self._try(session.decide, "chain:final", payload)

        try:
            session.__exit__(None, None, None)
        except _anchor.APIError as e:
            if not self.config.fail_open:
                raise
            sys.stderr.write(
                f"[satsignal] manifest anchor failed: {e}; "
                f"continuing (fail_open=True)\n"
            )

    def on_chain_error(
        self,
        error,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self._is_top_level(parent_run_id):
            return
        with self._lock:
            session = self._sessions.pop(run_id, None)
        if session is None:
            return
        # Pass the exc to __exit__ so it skips the manifest anchor (the
        # decisions still get persisted to handoff.json if configured).
        try:
            session.__exit__(type(error), error, None)
        except _anchor.APIError as e:
            if not self.config.fail_open:
                raise
            sys.stderr.write(
                f"[satsignal] handoff write failed: {e}; "
                f"continuing (fail_open=True)\n"
            )

    # ---- session lookup ----

    def _lookup_top_session(
        self, run_id: UUID, parent_run_id: Optional[UUID]
    ) -> Optional[_anchor.Session]:
        # The decision callback fires under a nested run_id; the active
        # session is keyed under the chain's TOP-LEVEL run_id, which we
        # don't get directly from LangChain at decision time. Walk the
        # ancestry: agent_action's parent_run_id is the chain root for
        # AgentExecutor flows. For deeper nesting we still resolve
        # via the sessions dict — the top-level run_id is the only key
        # present, so any non-None lookup wins.
        with self._lock:
            # Direct hit (rare but possible if decide fires at top level)
            session = self._sessions.get(run_id)
            if session is not None:
                return session
            # Otherwise, the parent_run_id is the chain root we want
            if parent_run_id is not None:
                session = self._sessions.get(parent_run_id)
                if session is not None:
                    return session
            # Fallback: if exactly one session is active, use it. Common
            # case for single-chain tests; ambiguous for concurrent runs
            # where it returns None and the decision is silently dropped.
            if len(self._sessions) == 1:
                return next(iter(self._sessions.values()))
            return None
