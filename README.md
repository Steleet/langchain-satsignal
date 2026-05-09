# langchain-satsignal

LangChain callback that anchors agent decisions on Bitcoin SV via [Satsignal](https://satsignal.cloud).

**Why this exists.** When an autonomous agent decides something — picks a tool, runs a query, signs off on an output — there's no native record proving *what was decided* or *when*. `langchain-satsignal` wires a `BaseCallbackHandler` into the four-anchor pattern Satsignal uses for agent runs: a policy snapshot at the start, a commit-reveal anchor per decision, and an evidence-bundle manifest at the end. Auditors get a tamper-evident, offline-verifiable record without needing access to your agent's runtime.

## Install

```bash
pip install langchain-satsignal
```

Only requires `langchain-core`. No SDK. The Satsignal helper is vendored in (stdlib-only) so the dependency tree stays flat.

## Quickstart

```python
from langchain_satsignal import SatsignalCallbackHandler, SatsignalConfig

handler = SatsignalCallbackHandler(SatsignalConfig(
    api_key="sk_live_...",     # mint in your Satsignal workspace
    matter_slug="agent-runs",  # workspace matter to anchor under
    agent_name="my-evaluator",
    agent_version="1.0.0",
))

# Pass to any LangChain agent / chain via the callbacks config:
result = executor.invoke(
    {"input": "..."},
    config={"callbacks": [handler]},
)
```

That's it. The handler:
- anchors a **policy snapshot** at the top-level chain start (system prompt + user input hashes),
- anchors a **commit-reveal commitment** per ReAct step (default; configurable),
- anchors an **evidence-bundle manifest** at the top-level chain end (Merkle root over all decisions).

See [`examples/react_agent.py`](examples/react_agent.py) for a complete runnable example.

## Config

| Field           | Default                              | Notes                                                                                  |
|-----------------|--------------------------------------|----------------------------------------------------------------------------------------|
| `api_key`       | (required)                           | Bearer key minted from your Satsignal workspace                                        |
| `matter_slug`   | (required)                           | Server-side matter; must exist before the first anchor                                 |
| `base_url`      | `https://app.satsignal.cloud`        | Override for self-hosted / staging                                                     |
| `fail_open`     | `True`                               | API errors logged + swallowed; agent execution continues                               |
| `decide_on`     | `"agent_action"`                     | `"agent_action"` (per ReAct step) / `"tool_start"` (per tool call) / `"chain_end"` (final outputs only) |
| `transport`     | `None`                               | Test hook. Inject a callable to mock HTTP — see "Testing without the network"          |
| `session_id`    | top-level run_id                     | Override only if you need a custom correlation key                                     |
| `write_handoff` | `False`                              | If True, writes `handoff.json` on chain end (contains tool inputs/outputs — review for secrets first) |
| `agent_name`    | `None`                               | Optional metadata baked into the policy snapshot                                       |
| `agent_version` | `None`                               | Optional metadata baked into the policy snapshot                                       |

## `decide_on` granularity

| Value           | One anchor per…                     | When to use                                                                            |
|-----------------|-------------------------------------|----------------------------------------------------------------------------------------|
| `agent_action`  | ReAct step (LLM decision to act)    | Default. Strongest "what did the agent decide" claim per anchor                        |
| `tool_start`    | Each individual tool call           | Tool-heavy agents where each external effect needs its own timestamp                   |
| `chain_end`     | The final chain output only         | Cheapest mode. Manifest still binds the run, but per-decision timing isn't anchored    |

Cost: ≈ 119 sats / anchor at 0.1 sat/byte BSV fee floor (≈ $0.0001 at $60/BSV in May 2026). A 5-step agent with `decide_on="agent_action"` runs about $0.0007 in chain fees. See the [Satsignal `/agents` doc](https://proof.satsignal.cloud/agents) for full cost analysis.

## Fail modes

`fail_open=True` (default) is right for most agents — an audit-trail gap is preferable to a halted agent. Errors go to stderr:

```
[satsignal] anchor failed: HTTP 429: ...; continuing (fail_open=True)
```

`fail_open=False` is right for **regulated workflows** where unanchored decisions are unacceptable. The first API error halts the agent.

A future release will add `fail_pending` (queue locally, anchor when network recovers); for now that flow needs a custom transport.

## Testing without the network

Frameworks that integrate this need to test their wrapper without burning chain fees. The `transport=` config takes any callable with the signature:

```python
transport(method, url, headers, body_bytes, timeout) -> (status_int, response_bytes)
```

so your test suite can substitute any HTTP mocking library. Example:

```python
def fake_transport(method, url, headers, body, timeout):
    return 200, b'{"txid":"deadbeef","bundle_id":"abc","root":"f00d","leaf_count":1}'

handler = SatsignalCallbackHandler(SatsignalConfig(
    api_key="sk_test", matter_slug="m", transport=fake_transport,
))
```

Status codes 4xx/5xx still raise `APIError`, so the handler's `fail_open` / `fail_closed` branches are testable. See [`tests/test_callback.py`](tests/test_callback.py) for the full pattern (8 tests, run offline in 0.04s).

## How the handler maps to LangChain hooks

| LangChain callback                                      | Satsignal action                                                          |
|---------------------------------------------------------|---------------------------------------------------------------------------|
| `on_chain_start`  (top-level: `parent_run_id is None`)  | Build policy snapshot from `serialized` + `inputs`, anchor it             |
| `on_chain_start`  (nested)                              | No-op (no double anchoring)                                               |
| `on_agent_action` (with `decide_on="agent_action"`)     | Wrap action in commit-reveal envelope, anchor commitment                  |
| `on_tool_start`   (with `decide_on="tool_start"`)       | Wrap tool input in commit-reveal envelope, anchor commitment              |
| `on_chain_end`    (top-level)                           | (If `decide_on="chain_end"`: anchor final outputs.) Anchor manifest.      |
| `on_chain_error`  (top-level)                           | Skip manifest (chain failed); clean up the session                        |

Concurrency: one handler instance can serve multiple top-level chain runs concurrently — each gets its own session keyed by the top-level `run_id`. Within a single chain, callbacks fire serially per LangChain's contract.

## Limitations (v0.1)

- **Sync handler only.** LangChain dispatches sync callbacks in a thread for async chains, so this works for both, but a native async handler is deferred.
- **No streaming token anchors.** Per-token anchoring would dominate chain fees and add no audit value.
- **No fail-pending queue mode.** See "Fail modes" above.
- **Policy snapshot is heuristic.** `serialized` from LangChain is hashed wholesale; for a tighter snapshot, build one yourself with the underlying `_anchor.build_policy_snapshot(...)` and anchor it before invoking the chain.

## Links

- [Satsignal](https://satsignal.cloud)
- [Satsignal `/agents` integration guide](https://proof.satsignal.cloud/agents)
- [Vendored helper source](https://satsignal.cloud/agent_anchor.py) (stdlib-only)

## License

MIT. See [LICENSE](LICENSE).
