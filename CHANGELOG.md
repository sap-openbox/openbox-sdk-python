# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2026-10-26

### Added
- OpenRouter routing provenance. After a governed `httpx` call to `openrouter.ai`
  completes inside an activity, the SDK sends an `llm_provenance` span carrying
  the generation id, upstream provider, served model, cost, token usage, finish
  reason and requested-vs-honored routing (`gen_ai.*`, `openbox.routing.*`,
  `openbox.model.*`). It looks up OpenRouter's generation record within a fixed
  12-second budget (`LOOKUP_BUDGET_SECONDS`, retries included, counted from the
  end of the model call) and otherwise sends the provenance in the completion
  response itself. A rejected lookup (401/403/5xx) is logged with its status.
  Lookups run on a small pool of daemon threads, so process exit never waits on
  one. Disable with `OPENBOX_OPENROUTER_PROVENANCE=false`.
- `HookType.LLM_PROVENANCE`.

### Changed
- `EvaluationClient.evaluate` / `aevaluate` wait for the closing session's own
  pending provenance lookups before sending `WorkflowCompleted` or
  `WorkflowFailed` (bounded by `DRAIN_TIMEOUT_SECONDS`, longer than the lookup
  budget), since Core drops spans that arrive after the session closes. Other
  sessions' lookups are not waited on. `OpenBoxRuntime.close()` drains the rest.

### Fixed
- `CoreAdapter.handle_approval` and the sync HTTP preflight no longer require
  an `approval_id`. Core's REQUIRE_APPROVAL verdict does not carry one, so every
  approval was rejected without a wait. The approval is polled by
  workflow/run/activity id, and both still fail safe when those are missing.
- Capture streamed text response bodies for sync/async `httpx` and `httpx2`
  requests without consuming the stream ahead of the caller. Emit completed
  telemetry once on EOF, close, or read failure, with bounded body capture and
  the original span/activity identity preserved.

## [1.4.0] - 2026-10-02

### Added
- HTTP span capture for `httpx2.Client` and `httpx2.AsyncClient` when `httpx2`
  is installed, alongside `httpx`, using the existing HTTP toggle and OpenBox
  started/completed payload contract.
- **Okta identity bootstrap** — an `okta_ai_agent` runtime now needs only
  `OPENBOX_API_URL`, `OPENBOX_API_KEY`, and `OPENBOX_OKTA_AGENT_PRIVATE_KEY`. The SDK
  fetches the remaining identity metadata (agent id, organization id, deployment id,
  assertion audience, external Okta agent id, credential `kid`, algorithm) from
  `GET /api/v2/auth/bootstrap` on first use and caches it in memory for the lifetime of
  the client. The signed assertion and its verification are unchanged — only how the SDK
  obtains its non-secret configuration.
- Before the first governed request the SDK derives its private key's RFC 7638 public-key
  thumbprint and compares it (constant-time) with the one Core reports for the agent's
  selected credential. A mismatch raises `OpenBoxConfigError` with actionable guidance;
  no governed request is sent.
- `EvaluationClient.refresh_identity_metadata()` / `arefresh_identity_metadata()` for
  long-running agents whose selected credential rotated. The thumbprint is re-verified
  *before* cached metadata is replaced, so a credential rotated to a key this process
  does not hold fails loudly and leaves the client on its previous identity. The client
  never refreshes automatically after an auth failure — a blind retry cannot repair a key
  mismatch and would hide it.
- `EvaluationClient.identity_metadata()` exposes the validated bootstrap document
  (non-secret).
- New modules `openbox_core.bootstrap` and `openbox_core.jwk_thumbprint`.

### Changed
- Okta configuration now resolves into exactly one of three modes, never a blend:
  **bootstrap** (private key only), **legacy explicit** (every metadata field set, works
  unchanged), or **invalid mixed** (only some metadata fields set) — which raises
  `OpenBoxConfigError` naming the offending fields rather than merging stale local values
  over what Core would have supplied.
- `OPENBOX_DEPLOYMENT_ID` is no longer required in bootstrap mode. Core returns its own
  deployment id, which prevents a runtime from signing for one deployment while calling
  another.
- No failure path downgrades to v1, to an unsigned request, or to a different identity
  method. A Core that predates the endpoint (`404`) produces actionable upgrade guidance.

## [1.3.1] - 2026-09-17

### Security
- Require `cryptography>=50,<51` to address CVE-2026-69247. No OpenBox SDK API or
  runtime behavior changes are included in this patch release.

## [1.3.0] - 2026-07-30

### Added
- v2 (Okta AI Agent) identity verification support, alongside existing v1 (OpenBox DID)
  and inferred `legacy_unsigned` support: tagged identity configuration
  (`OpenBoxDidIdentityConfig` / `OktaAiAgentIdentityConfig` in
  `openbox_core.identity_types`), RS256 assertion signing
  (`openbox_core.identity_okta`), and version-aware endpoint selection —
  `okta_ai_agent` identities route to `/api/v2/governance/evaluate`,
  `/api/v2/governance/approval`, `GET /api/v2/auth/validate`, and
  `/api/v2/handoffs`; there is no cross-version retry.
- New env vars: `OPENBOX_AGENT_IDENTITY_METHOD`, `OPENBOX_OKTA_AGENT_ID`,
  `OPENBOX_OKTA_AGENT_KEY_ID`, `OPENBOX_OKTA_AGENT_PRIVATE_KEY`,
  `OPENBOX_OKTA_AGENT_ALGORITHM`, `OPENBOX_AGENT_ID`, `OPENBOX_ORGANIZATION_ID`,
  `OPENBOX_DEPLOYMENT_ID`, `OPENBOX_AGENT_PROOF_AUDIENCE`. DID and Okta identity
  fields are mutually exclusive; the canonical Okta private-key encoding is PKCS8 PEM.
- `EvaluationClient.emit_handoff` / `aemit_handoff` — source-authenticated handoff
  (`POST /api/v1/handoffs` for openbox_did, `/api/v2/handoffs` for okta_ai_agent).
  Inferred unsigned mode raises `OpenBoxConfigError` (no source identity to prove)
  instead of silently falling back to the legacy receiver-authenticated event.
- `EvaluationClient.validate_okta_identity_transition` /
  `avalidate_okta_identity_transition` and `.validate_openbox_did_identity_transition` /
  `avalidate_openbox_did_identity_transition` — transition-preflight helpers that sign
  with an **explicit candidate identity** and never fall back to the client's active
  signer, even if the client happens to already use the same identity method.

### Fixed
- **BREAKING (security fix):** `EvaluationClient.evaluate` / `aevaluate` and
  `.poll_approval` / `.apoll_approval` now raise an actionable
  `OpenBoxAuthError`/`OpenBoxSigningError` on HTTP 401/403, regardless of
  `on_api_error`. Previously, evaluate routed every `>= 400` response — including
  401/403 — through the network-failure path, which under the default `fail_open`
  produced a fallback `ALLOW`; an agent whose key was revoked kept operating,
  ungoverned. Approval polling treated 401/403 the same as a transient poll failure
  (returned `None`, read by callers as "still pending"), so a revoked agent's
  approval wait never terminated. Both are now hard authentication failures: they
  fail closed unconditionally and are never retried against the other API version.
  A genuine network error or 5xx is unaffected and still respects `on_api_error`.

## [1.2.0] - 2026-07-23

### Changed
- **BREAKING:** Renamed the public BLOCK remediation contract from `retry_plan` to `patch`.
  `RetryPlan` is now `Patch`, `RetryDirective` is now `PatchDirective`, and
  `handle_retryable_block()` is now `handle_patch()`. `EvaluationResult.retry_plan` and
  `ApprovalResult.retry_plan` are now `.patch`. The parser now reads only the canonical outer
  wire key `patch`; the old `retry_plan` key is no longer recognized.
- The verdict gate is unchanged: a directive is still surfaced only for an exact `BLOCK` verdict
  carrying a valid patch — never for HALT/`should_stop()`, and never for an expired approval
  result.

### Notes
- This is a breaking release: the old public names (`RetryPlan`, `RetryDirective`,
  `handle_retryable_block`, `.retry_plan`) are removed, not aliased. Consumers still pinned to
  `1.1.0` remain fail-safe — they ignore the unknown `patch` field but still enforce the `BLOCK`
  verdict.

## [1.1.0] - 2026-07-21

### Added
- `RetryPlan` and `RetryDirective` dataclasses in `openbox_core.contracts.results`.
- Optional `retry_plan` directive parsing on both `EvaluationResult` and `ApprovalResult`.
  A `_MISSING` sentinel keeps a present `new_input: null` distinct from an absent field; falsy
  values (`null`, `""`, `0`, `[]`, `{}`) are preserved; a boolean `new_input` is rejected; and every
  number (recursively) must be finite and, if integral, a JS-safe integer (`|n| <= 2^53 - 1`).
- `handle_retryable_block(result)` — an opt-in, pure inspector that returns a `RetryDirective` only
  for a `BLOCK` verdict carrying a valid plan. Returns `None` for a plain BLOCK, every non-BLOCK
  verdict (including HALT), a pending verdict, and an expired `ApprovalResult`.

### Notes
- Default enforcement is unchanged: a `BLOCK` verdict still raises `GovernanceBlockedError`. The new
  helper is opt-in and never triggers an automatic retry; malformed or ineligible plans are treated
  as absent (never fail open).
