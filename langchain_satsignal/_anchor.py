#!/usr/bin/env python3
"""agent_anchor.py — Satsignal session helper for autonomous AI agents.

Bundles the four-anchor pattern (policy snapshot at start, per-decision
commitments, evidence-bundle manifest at end) into a single context
manager. Writes a ``handoff.json`` so an auditor receiving the file plus
the chain anchors can reconstruct the agent's session offline.

Stdlib only. No SDK install. Drop into your agent's runtime alongside
your existing tool-use loop.

Quick start:

    from agent_anchor import Session

    with Session(
        api_key="sk_live_...",
        matter_slug="agent-runs",
        agent_name="my-evaluator-bot",
        agent_version="1.4.2",
    ) as s:
        s.policy(
            system_policy_text=SYSTEM_PROMPT,
            user_instruction_text=user_request,
            tool_permissions={"max_calls": 50, "tools": [...]},
            budget_limits={"max_usd": 5.0},
            model_config={"model": "claude-opus-4-7", "temperature": 0.2},
        )

        for step in plan:
            decision = run_step(step)
            s.decide(step.label, decision)

        # __exit__ anchors the evidence-bundle manifest + writes
        # handoff.json with policy_anchor + decisions[] + manifest_anchor.

Why this shape:

    - **Policy snapshot at start.** Binds "what the agent was allowed
      to do" before any decision happens. Auditor can prove with the
      original system prompt that this anchor came from this prompt.

    - **Commit-reveal per decision.** Each decision payload is wrapped
      with a random nonce before hashing. Pre-discloses TIMING (the
      anchor exists) without pre-disclosing CONTENT (the payload is
      hidden until you reveal). Auditor receives the nonce + payload
      via handoff.json, recomputes, checks against the chain.

    - **Evidence-bundle manifest at end.** Cryptographic "these
      decisions belong together" binding via Merkle root. This is the
      part that ties the session — NOT session_id, which is a
      workspace-side query key, not a chain-bound proof.

Anchor-per-decision vs batch-only:

    - **Use Session() as written** (default) when the relative ORDER /
      TIMING of decisions matters to your auditor. Each decision is
      individually chain-timestamped, so disclosure timing is provable.

    - **Use Session(commit_each=False)** when only end-state integrity
      matters. Decisions are kept locally and only the final manifest
      anchors. One on-chain anchor per session. Cheaper, weaker timing
      claim. The reviewer cold-start used commit_each=True (default).

Costs (2026-05): a 5-anchor session ≈ 595 sats ≈ $0.0004 at $60/BSV.
A "commit_each=False" session is one anchor.

Honest onboarding note:

    This helper requires an API key from a Satsignal workspace. The
    workspace itself requires a magic-link sign-in to a real email
    inbox — there is no agent-only signup path. A human must claim
    the magic link once, mint a key with anchors:create + receipts:read
    scopes, and hand the bearer string to the agent. From there the
    agent self-integrates; before that, you need a person.

Testing without the network:

    Pass ``transport=`` to ``Session(...)`` (or the module-level
    ``_post``) to inject a callable with signature
    ``(method, url, headers, body_bytes, timeout) -> (status, response_bytes)``.
    Default is urllib; framework CI passes a callable that returns
    canned responses, so the wrapper is exercisable without the
    network or burning chain fees. ``APIError`` is still raised on
    non-2xx, so 401/429 branches stay testable.

    Example::

        def fake(method, url, headers, body, timeout):
            return 200, b'{"txid":"deadbeef","bundle_id":"abcd"}'

        with Session(api_key="sk_test", matter_slug="m",
                     transport=fake, write_handoff=False) as s:
            s.decide("step-1", {"x": 1})
        assert s.decisions[0]["txid"] == "deadbeef"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Optional


WIRE_VERSION_HANDOFF = "satsignal-agent-handoff-v1"
WIRE_VERSION_POLICY = "satsignal-policy-snapshot-v1"
WIRE_VERSION_DECISION = "satsignal-agent-decision-v1"


# ---- JCS-style canonicalize (matches notary.canonicalize lenience for floats) ----

def _nfc_deep(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise ValueError(f"NaN/Infinity not canonicalizable: {value!r}")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_deep(v) for v in value]
    if isinstance(value, dict):
        return {
            unicodedata.normalize("NFC", k): _nfc_deep(v)
            for k, v in value.items()
        }
    raise TypeError(f"non-canonicalizable: {type(value).__name__}")


def canonicalize(doc: Any) -> bytes:
    """JCS-style canonical bytes. Identical output to the matching
    helper in scripts/policy_snapshot.py."""
    return json.dumps(
        _nfc_deep(doc), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(s: str) -> str:
    return _sha256_hex(s.encode("utf-8"))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- HTTP helpers (stdlib urllib; no requests dep) ----

class APIError(RuntimeError):
    """Non-2xx response from /api/v1/anchors. ``status`` + ``body``
    carry the wire-level error so a caller can branch on 429 (quota)
    vs 400 (validation) vs 401 (bad key)."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:512]}")
        self.status = status
        self.body = body


# Transport hook: callable with signature
#   (method, url, headers, body_bytes, timeout) -> (status, response_bytes)
# Default is _urllib_transport. Tests pass a closure returning canned
# (status, bytes) so the wrapper is unit-testable without the network.
# Kept as Any in annotations so the file's typing surface stays minimal.

def _urllib_transport(
    method: str, url: str, headers: dict, body: bytes, timeout: float,
) -> tuple:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"")


def _post(
    url: str, *, bearer: str, body: dict, timeout: float = 30.0,
    transport: Optional[Any] = None,
) -> dict:
    headers = {
        "Authorization": f"Bearer {bearer}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8")
    send = transport or _urllib_transport
    status, resp_bytes = send("POST", url, headers, body_bytes, timeout)
    if 200 <= status < 300:
        return json.loads(resp_bytes.decode("utf-8"))
    raise APIError(status, resp_bytes.decode("utf-8", errors="replace"))


# ---- Policy snapshot (matches policy_snapshot.py wire format) ----

def build_policy_snapshot(
    *,
    agent_name: Optional[str] = None,
    agent_version: Optional[str] = None,
    system_policy_text: Optional[str] = None,
    system_policy_hash: Optional[str] = None,
    user_instruction_text: Optional[str] = None,
    user_instruction_hash: Optional[str] = None,
    tool_permissions: Optional[dict] = None,
    tool_permissions_hash: Optional[str] = None,
    budget_limits: Optional[dict] = None,
    budget_limits_hash: Optional[str] = None,
    model_config: Optional[dict] = None,
    model_config_hash: Optional[str] = None,
    extra: Optional[dict] = None,
    snapshot_at_utc: Optional[str] = None,
) -> dict:
    """Assemble + hash a policy snapshot. Same wire shape as
    scripts/policy_snapshot.py — pass either the raw artifact (text /
    dict) and we'll hash, or pass the sha256 directly."""
    snap: dict = {
        "version": WIRE_VERSION_POLICY,
        "snapshot_at_utc": snapshot_at_utc or _utc_now(),
    }
    agent_obj: dict = {}
    if agent_name:
        agent_obj["name"] = agent_name
    if agent_version:
        agent_obj["version"] = agent_version
    if agent_obj:
        snap["agent"] = agent_obj

    pairs = (
        ("system_policy_hash",
         system_policy_hash or (_hash_text(system_policy_text)
                                if system_policy_text is not None else None)),
        ("user_instruction_hash",
         user_instruction_hash or (_hash_text(user_instruction_text)
                                   if user_instruction_text is not None else None)),
        ("tool_permissions_hash",
         tool_permissions_hash or (_sha256_hex(canonicalize(tool_permissions))
                                   if tool_permissions is not None else None)),
        ("budget_limits_hash",
         budget_limits_hash or (_sha256_hex(canonicalize(budget_limits))
                                if budget_limits is not None else None)),
        ("model_config_hash",
         model_config_hash or (_sha256_hex(canonicalize(model_config))
                               if model_config is not None else None)),
    )
    for k, v in pairs:
        if v is None:
            continue
        v = v.strip().lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"{k}: must be 64-char lowercase sha256 hex")
        snap[k] = v
    if extra:
        snap["extra"] = extra
    return snap


# ---- Commit-reveal wrapper for a decision payload ----

def commit_decision(payload: Any, *, nonce_hex: Optional[str] = None) -> dict:
    """Wrap ``payload`` in a fresh-nonce commit-reveal envelope. Returns
    the canonical envelope + its sha256 — what gets POSTed to
    /api/v1/anchors with category=commitment.

    Disclosure path: hand the auditor the envelope + nonce_hex, they
    canonicalize again, recompute sha256, check against the receipt.
    """
    if nonce_hex is None:
        nonce_hex = secrets.token_hex(16)  # 128-bit nonce
    if len(nonce_hex) % 2 != 0 or any(c not in "0123456789abcdef" for c in nonce_hex):
        raise ValueError("nonce_hex must be even-length lowercase hex")
    envelope = {
        "version": WIRE_VERSION_DECISION,
        "nonce_hex": nonce_hex,
        "payload": payload,
    }
    canonical = canonicalize(envelope)
    return {
        "envelope": envelope,
        "canonical_bytes": canonical,
        "decision_sha256_hex": _sha256_hex(canonical),
        "nonce_hex": nonce_hex,
    }


# ---- Session context manager ----

class Session:
    """Bundles policy + decisions + manifest into one object. Use as a
    context manager; ``__exit__`` anchors the manifest and writes
    ``handoff.json``. Errors during the session do NOT anchor a manifest
    (the partial handoff is still written so an auditor can see what
    completed before the failure)."""

    def __init__(
        self,
        api_key: str,
        *,
        matter_slug: str,
        base_url: str = "https://app.satsignal.cloud",
        session_id: Optional[str] = None,
        handoff_path: Optional[str] = "handoff.json",
        agent_name: Optional[str] = None,
        agent_version: Optional[str] = None,
        commit_each: bool = True,
        write_handoff: bool = True,
        transport: Optional[Any] = None,
    ):
        if not api_key.startswith("sk_") and not api_key.startswith("sat_"):
            # Soft warning, not a block — local dev keys may use other
            # prefixes. Fail at the API on a real misuse.
            sys.stderr.write(
                "[agent_anchor] note: api_key does not match the usual "
                "sk_/sat_ prefix; continuing\n"
            )
        self.api_key = api_key
        self.matter_slug = matter_slug
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or secrets.token_hex(8)
        self.handoff_path = handoff_path
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.commit_each = commit_each
        self.write_handoff = write_handoff
        self.transport = transport

        self.started_at_utc: Optional[str] = None
        self.ended_at_utc: Optional[str] = None
        self.policy_anchor: Optional[dict] = None
        self.policy_snapshot: Optional[dict] = None
        self.decisions: list[dict] = []
        self.manifest_anchor: Optional[dict] = None

    # ---- context-manager protocol ----

    def __enter__(self) -> "Session":
        self.started_at_utc = _utc_now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.ended_at_utc = _utc_now()
        try:
            if exc_type is None and self.decisions:
                self._anchor_manifest()
        finally:
            if self.write_handoff and self.handoff_path:
                self._write_handoff(self.handoff_path)
        return False  # re-raise any exception

    # ---- anchoring primitives ----

    def policy(self, **snapshot_kwargs) -> dict:
        """Build + anchor a policy snapshot. Accepts the same kwargs as
        ``build_policy_snapshot`` (raw text + dicts get hashed for you).
        Anchors with category=policy_snapshot. Returns the anchor
        response."""
        if "agent_name" not in snapshot_kwargs and self.agent_name:
            snapshot_kwargs["agent_name"] = self.agent_name
        if "agent_version" not in snapshot_kwargs and self.agent_version:
            snapshot_kwargs["agent_version"] = self.agent_version
        snap = build_policy_snapshot(**snapshot_kwargs)
        canonical = canonicalize(snap)
        sha = _sha256_hex(canonical)
        size = len(canonical)
        anchor = self._post_anchor({
            "matter_slug": self.matter_slug,
            "sha256_hex": sha,
            "file_size": size,
            "category": "policy_snapshot",
            "label": f"policy snapshot {snap.get('snapshot_at_utc', '')}".strip(),
            "session_id": self.session_id,
        })
        self.policy_snapshot = snap
        self.policy_anchor = {
            "txid": anchor.get("txid"),
            "bundle_id": anchor.get("bundle_id"),
            "snapshot_sha256_hex": sha,
            "anchored_at_utc": _utc_now(),
        }
        return anchor

    def decide(self, label: str, payload: Any) -> dict:
        """Anchor one decision. Wraps ``payload`` in a fresh-nonce
        commit-reveal envelope, hashes it, and (when commit_each=True,
        the default) anchors with category=commitment. The label is the
        operator's tag for the decision in the eventual manifest.

        Returns the anchor response (or, when commit_each=False, just a
        dict with the local commit so callers can keep flow uniform).
        """
        commit = commit_decision(payload)
        decision_record = {
            "label": label,
            "decision_sha256_hex": commit["decision_sha256_hex"],
            "nonce_hex": commit["nonce_hex"],
            "envelope": commit["envelope"],
            "anchored_at_utc": _utc_now(),
        }
        if self.commit_each:
            anchor = self._post_anchor({
                "matter_slug": self.matter_slug,
                "sha256_hex": commit["decision_sha256_hex"],
                "file_size": len(commit["canonical_bytes"]),
                "category": "commitment",
                "label": label,
                "session_id": self.session_id,
            })
            decision_record["txid"] = anchor.get("txid")
            decision_record["bundle_id"] = anchor.get("bundle_id")
            self.decisions.append(decision_record)
            return anchor
        # batch path: hold the local commit for the end-of-session manifest.
        self.decisions.append(decision_record)
        return {"local_only": True, **decision_record}

    # ---- manifest assembly ----

    def _anchor_manifest(self) -> dict:
        """End-of-session manifest. Items[] is one (label, decision_sha)
        per decision. Server computes the Merkle root + binds it on
        chain via mode=manifest."""
        items = [
            {
                "label": d["label"],
                "sha256_hex": d["decision_sha256_hex"],
            }
            for d in self.decisions
        ]
        anchor = self._post_anchor({
            "matter_slug": self.matter_slug,
            "items": items,
            "category": "evidence_bundle",
            "label": (
                f"agent session {self.session_id} "
                f"({len(items)} decisions)"
            ),
            "session_id": self.session_id,
        })
        self.manifest_anchor = {
            "txid": anchor.get("txid"),
            "bundle_id": anchor.get("bundle_id"),
            "root": anchor.get("root"),
            "leaf_count": anchor.get("leaf_count"),
            "anchored_at_utc": _utc_now(),
        }
        return anchor

    # ---- HTTP wiring ----

    def _post_anchor(self, body: dict) -> dict:
        return _post(
            f"{self.base_url}/api/v1/anchors",
            bearer=self.api_key, body=body, transport=self.transport,
        )

    # ---- handoff.json ----

    def to_handoff(self) -> dict:
        return {
            "version": WIRE_VERSION_HANDOFF,
            "session_id": self.session_id,
            "matter_slug": self.matter_slug,
            "agent": {
                k: v for k, v in
                (("name", self.agent_name), ("version", self.agent_version))
                if v
            } or None,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "policy_anchor": self.policy_anchor,
            "policy_snapshot": self.policy_snapshot,
            "decisions": self.decisions,
            "manifest_anchor": self.manifest_anchor,
        }

    def _write_handoff(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_handoff(), f, indent=2, sort_keys=True)
            f.write("\n")


# ---- CLI: demo subcommand mirrors the reviewer cold-start flow ----

def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a 5-anchor probe (policy + 3 decisions + manifest) and write
    a handoff.json. Mirrors the reviewer's cold-start integration."""
    api_key = args.api_key or os.environ.get("SATSIGNAL_API_KEY", "")
    if not api_key:
        sys.stderr.write(
            "[demo] need --api-key or SATSIGNAL_API_KEY env var\n"
        )
        return 2
    with Session(
        api_key=api_key,
        matter_slug=args.matter,
        base_url=args.base_url,
        agent_name="agent_anchor-demo",
        agent_version="1",
        handoff_path=args.handoff,
    ) as s:
        sys.stderr.write(f"[demo] session_id={s.session_id}\n")
        s.policy(
            system_policy_text="You are a demo agent. Make 3 toy decisions.",
            user_instruction_text="run a satsignal cold-start probe",
            tool_permissions={"max_calls": 10, "tools": ["echo"]},
            budget_limits={"max_usd": 0.01},
            model_config={"model": "demo", "temperature": 0.0},
        )
        sys.stderr.write(f"[demo] policy txid={s.policy_anchor['txid']}\n")
        for i in range(3):
            r = s.decide(
                f"step-{i+1}",
                {"step": i + 1, "decision": "demo", "ts": _utc_now()},
            )
            sys.stderr.write(f"[demo] decision-{i+1} txid={r.get('txid')}\n")
    sys.stderr.write(f"[demo] manifest txid={s.manifest_anchor['txid']}\n")
    sys.stderr.write(f"[demo] wrote {args.handoff}\n")
    return 0


def _cmd_verify_handoff(args: argparse.Namespace) -> int:
    """Local-only verifier: recompute every decision_sha256 from the
    envelope shipped in handoff.json and compare against the recorded
    sha. This catches handoff-file tampering but does NOT chain-confirm
    — for that, fetch /api/v1/lookup/<txid> and compare against the
    payload_hex commitment."""
    with open(args.handoff, "r", encoding="utf-8") as f:
        handoff = json.load(f)
    decisions = handoff.get("decisions") or []
    rows = []
    for d in decisions:
        env = d.get("envelope")
        if env is None:
            rows.append({"label": d.get("label"), "verified": False,
                         "reason": "no envelope shipped (revealed-on-demand)"})
            continue
        recomputed = _sha256_hex(canonicalize(env))
        ok = recomputed == d.get("decision_sha256_hex")
        rows.append({
            "label": d.get("label"),
            "verified": ok,
            "recomputed_sha256_hex": recomputed,
            "declared_sha256_hex": d.get("decision_sha256_hex"),
            "txid": d.get("txid"),
        })
    print(json.dumps({
        "session_id": handoff.get("session_id"),
        "decision_count": len(rows),
        "all_verified": bool(rows) and all(r["verified"] for r in rows),
        "decisions": rows,
        "note": (
            "Local-only check. To chain-confirm, fetch each txid via "
            "/api/v1/lookup/<txid> and compare payload_hex against the "
            "OP_RETURN of the commitment doc you'd reconstruct from "
            "(version, nonce_hex, payload)."
        ),
    }, indent=2))
    return 0 if (rows and all(r["verified"] for r in rows)) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="agent_anchor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser(
        "demo",
        help=(
            "anchor a 5-anchor probe session "
            "(policy + 3 decisions + manifest)"
        ),
    )
    pd.add_argument("--api-key",
                    help="bearer key (or set SATSIGNAL_API_KEY env var)")
    pd.add_argument("--matter", default="agent-runs",
                    help="matter_slug to anchor under (default agent-runs)")
    pd.add_argument("--base-url", default="https://app.satsignal.cloud",
                    help="API base URL")
    pd.add_argument("--handoff", default="handoff.json",
                    help="output path for handoff.json")
    pd.set_defaults(func=_cmd_demo)

    pv = sub.add_parser(
        "verify-handoff",
        help="local-only sanity check on a handoff.json (no chain hit)",
    )
    pv.add_argument("--handoff", required=True,
                    help="path to a handoff.json produced by Session()")
    pv.set_defaults(func=_cmd_verify_handoff)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
