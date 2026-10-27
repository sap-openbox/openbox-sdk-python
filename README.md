# openbox-sdk-python

OpenBox base SDK (`import openbox_core`) — the standalone core every OpenBox
framework SDK builds on. It owns:

- **Contracts** — event and result dataclasses (`EventEnvelope`,
  `EvaluationResult`, `ApprovalResult`, `Verdict`).
- **Always-strict governance gate** — malformed event/runtime contracts raise
  `ContractError` before send; there are no configurable gate modes.
- **Identity & signing** — v1 OpenBox DID: AIP DID validation + Ed25519 request
  signing, byte-compatible with the Temporal SDK signing contract. v2 Okta AI
  Agent: tagged identity configuration + RS256 JWT assertion signing
  (`openbox_core.identity_okta`, `openbox_core.identity_types`).
- **Evaluate client** — sync + async calls to OpenBox Core, routed by
  configured identity: `/api/v1/governance/evaluate` (`/approval`,
  `/auth/validate`, `/handoffs`) for OpenBox DID / inferred unsigned mode, or
  the `/api/v2/*` equivalents for Okta AI Agent — no cross-version retry.
- **Context & runtime** — `ActivityContext`, `ContextStore`, trace
  correlation, `FrameworkAdapter` protocol, `OpenBoxRuntime`.
- **OTel span wire serialization** — OTel spans are the internal source of
  truth, projected to the current Core `SpanData` wire shape before send.
- **Generic instrumentation** — HTTP/DB/file/function wrappers with a
  started/completed hook runtime enforcing preflight before real operations.
- **Conformance kit** — reusable fixtures framework SDKs run to prove parity.

Framework SDKs (Temporal, LangGraph, LangChain, DeepAgent, CrewAI) are thin
adapters: they bind framework lifecycle into `ActivityContext`, emit lifecycle
events, and translate `EvaluationResult` into native behavior.

## Framework SDK integration

New framework SDK developers and agents should start with
[the OpenBox Python Framework SDK Integration guide](.github/instructions/openbox-sdk-python.instructions.md).
It describes how to build a framework SDK on top of this base package without
reimplementing config, signing, governance gates, hook wire payloads, or
conformance fixtures.

## Install

```bash
uv sync                 # core
uv sync --extra http    # + HTTP instrumentation targets
uv sync --extra db      # + DB instrumentation targets
```

HTTP instrumentation also captures `httpx2.Client` and `httpx2.AsyncClient`
requests when `httpx2` is installed by the host application. It uses the same
HTTP toggle and started/completed span payloads as `httpx`, including body
capture and credential-header redaction.

For streamed `httpx`/`httpx2` responses, completed telemetry is emitted when the
caller finishes reading the decoded body or closes the response. Text chunks are
copied as they pass through; the SDK does not drain or buffer the entire stream.
SSE responses retain their raw event text in `response_body`. Capture respects
`privacy.max_body_size` (characters); when that limit is disabled, streaming
capture still uses a 65,536-character bound. Binary bodies are not captured.
A response closed before transport EOF records the consumed prefix and sets
`attributes["openbox.http.response_body.partial"]` to `true`. This includes clients
that stop reading after an SSE end marker and then close the HTTP response.
Stream read failures retain their original exception and report partial evidence.

## OpenRouter routing provenance

With the `http` extra installed, a governed `httpx` call to `openrouter.ai` made
inside an activity also reports which upstream provider served it. After the
response arrives, the SDK sends Core an `llm_provenance` span carrying
`gen_ai.generation.id`, the upstream provider and served model, cost and token
usage, finish reason, and whether the requested model and `provider.only`
routing were honoured (`gen_ai.*`, `openbox.routing.*`, `openbox.model.*`).
The dashboard's Provenance tab, shown for agents whose workflow engine is
OpenRouter, aggregates these spans.

- The lookup runs on a small pool of background (daemon) threads and never
  delays or fails the model call. It reuses the call's own `Authorization`
  header; no extra key is needed.
- OpenRouter's generation record (which adds data region, latency and fallback
  attempts) is published late, so it is tried within a 12-second budget
  (`LOOKUP_BUDGET_SECONDS`), counted from the end of the model call. When it is
  not ready, the provenance in the completion response itself is sent instead.
  A lookup rejected by OpenRouter (401/403/5xx) is logged with its status.
- `WorkflowCompleted` / `WorkflowFailed` wait for that session's pending
  lookups only (at most `DRAIN_TIMEOUT_SECONDS`, longer than the budget),
  because Core does not record spans that arrive after the session closes.
  `OpenBoxRuntime.close()` waits for any that remain.
- Set `OPENBOX_OPENROUTER_PROVENANCE=false` to turn it off.

## Okta AI Agent identity (`okta_ai_agent`)

An agent whose OpenBox identity is verified against an Okta AI Agent credential
signs every request with an RS256 assertion. That assertion binds seven values
OpenBox Core already owns, so the runtime does **not** configure them — it fetches
them from Core after authenticating with its API key:

```dotenv
OPENBOX_API_URL=https://core.example.com
OPENBOX_API_KEY=obx_live_...
OPENBOX_OKTA_AGENT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----"
```

That is the complete configuration. The private key is the only value Core can
never supply — **OpenBox never receives, stores, or returns it.** Everything else
(agent id, organization id, deployment id, assertion audience, external Okta agent
id, credential `kid`, algorithm) comes from `GET /api/v2/auth/bootstrap` on first
use and is cached in memory for the lifetime of the client.

Before the first governed request the SDK derives its private key's RFC 7638
public-key thumbprint and compares it with the one Core reports for the agent's
selected credential. A mismatch raises `OpenBoxConfigError` immediately, rather
than sending an assertion that could only be rejected:

> The configured private key does not match the selected Okta credential for this
> OpenBox agent. Export the private key associated with the selected credential,
> or rotate the agent credential.

**Credential rotation.** Long-running agents can refresh explicitly:

```python
document = client.refresh_identity_metadata()   # async: await client.arefresh_identity_metadata()
print(document.okta.credential_kid)
print(document.authority.activation_version)
```

The document also carries required, non-secret assignment, provider-generation,
activation, credential, and projection identifiers. The SDK fails closed when an
IAM-v3 Core deployment omits or returns malformed authority metadata.
The refresh re-runs the thumbprint check *before* replacing cached metadata, so a
credential that rotated to a key this process does not hold fails loudly and leaves
the client on its previous identity. The SDK never refreshes automatically after an
auth failure: rotation may have selected a new public key while the process still
holds the old private key, and a silent retry would hide that rather than fix it.

**Requirements.** The key must be a PKCS8 PEM RSA key of at least 2048 bits, and
its public half must already be registered in Okta for the selected credential.

**Explicit configuration** (every metadata field set locally) remains supported for
compatibility. The two styles cannot be mixed: a configuration carrying only *some*
metadata fields raises `OpenBoxConfigError` naming the offending fields, rather than
quietly merging stale local values over what Core would have supplied. If Core
answers `404`, the SDK reports that the deployment predates bootstrap and asks you
to upgrade Core or supply the complete explicit configuration — it never downgrades
to an unsigned request or to a different identity method.

## Import safety

`openbox_core.__init__` and all `openbox_core.contracts.*` modules import
without pulling in `httpx`, `cryptography`, `requests`, or OTel
instrumentation — safe for constrained framework paths such as the Temporal
workflow sandbox. `tests/test_import_safety.py` enforces this in CI.

`openbox_core.bootstrap` and `openbox_core.jwk_thumbprint` follow the same rule —
both keep `httpx`/`cryptography` behind function-local imports.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

## License

MIT — see [`LICENSE`](LICENSE).
