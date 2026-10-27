"""HookRuntime — the SINGLE decision point for hook verdicts.

One path only: wrapper -> hook runtime -> adapter. Wrappers never interpret
verdicts; this runtime validates via the gate and delegates every native
effect to the FrameworkAdapter:

- started BLOCK/HALT  -> mark abort (+halt flag) -> ``adapter.raise_hook_blocked``
- started REQUIRE_APPROVAL -> approval flow; rejected/unavailable -> blocked
- completed verdicts  -> ``adapter.on_completed_hook_result`` + abort/halt
  flags for FUTURE execution (the operation already ran; never undone)
- prior abort         -> fail fast without another network call
- no bound context    -> skip silently (not an error)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, NoReturn

from ..adapters.base import adapter_accepts_context
from ..approvals import ApprovalPoller
from ..contracts.events import EventEnvelope
from ..contracts.otel_spans import HookType, Stage
from ..contracts.results import EvaluationResult, Verdict
from ..errors import ContractError, GovernanceAPIError, GovernanceBlockedError
from ..hooks.events import build_hook_event, resolve_context
from ..runtime import OpenBoxRuntime

logger = logging.getLogger(__name__)

__all__ = ["HookRuntime"]


class HookRuntime:
    """Drives preflight/completed hook evaluation for one OpenBoxRuntime."""

    def __init__(self, runtime: OpenBoxRuntime):
        self._runtime = runtime
        self._store = runtime.context_store
        self._gate = runtime.gate
        self._adapter = runtime.adapter
        # Decide ONCE whether the adapter's callbacks take ``context``, by
        # inspecting their signatures — so a genuine TypeError raised inside a
        # callback body is never mistaken for an arity mismatch and swallowed.
        self._completed_accepts_context = adapter_accepts_context(
            self._adapter.on_completed_hook_result
        )
        self._approval_accepts_context = adapter_accepts_context(self._adapter.handle_approval)
        hitl = runtime.config.hitl
        self._sync_poller: ApprovalPoller | None = None
        if hitl.enabled:
            self._sync_poller = ApprovalPoller(
                runtime.client,
                poll_interval_seconds=hitl.poll_interval_ms / 1000.0,
                max_wait_seconds=(hitl.max_wait_ms / 1000.0) if hitl.max_wait_ms else None,
            )

    # ── Preflight (started stage) ─────────────────────────────────────────

    def preflight(
        self,
        span: Any,
        *,
        hook_type: HookType,
        identifier: str = "",
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        """Evaluate BEFORE the real operation. True ⇒ proceed.

        Blocking outcomes never return — the adapter raises. Skipped hooks
        (disabled preflight / no context) return True.
        """
        event = self._pre_gate(span, hook_type, fields)
        if event is None:
            return True
        try:
            result = self._gate.preflight(event)
        except (ContractError, GovernanceAPIError) as e:
            self._fail_closed_started(e, span)
        return self._decide_started(result, span, sync=True)

    async def apreflight(
        self,
        span: Any,
        *,
        hook_type: HookType,
        identifier: str = "",
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        """Async :meth:`preflight` — approval delegates to the adapter."""
        event = self._pre_gate(span, hook_type, fields)
        if event is None:
            return True
        try:
            result = await self._gate.apreflight(event)
        except (ContractError, GovernanceAPIError) as e:
            self._fail_closed_started(e, span)
        return await self._adecide_started(result, span)

    def _pre_gate(
        self, span: Any, hook_type: HookType, fields: Mapping[str, Any] | None
    ) -> EventEnvelope | None:
        if not self._runtime.config.instrumentation.preflight_enabled:
            return None
        ctx = resolve_context(self._store, span)
        if ctx is not None:
            # Abort short-circuit: a prior hook already stopped this activity.
            if self._store.is_activity_aborted(ctx.workflow_id, ctx.activity_id):
                self._adapter.raise_hook_blocked(
                    EvaluationResult(
                        verdict=Verdict.BLOCK,
                        reason="Activity aborted by a prior hook verdict",
                    )
                )
        return build_hook_event(
            self._store, span, stage=Stage.STARTED, hook_type=hook_type, fields=fields
        )

    def _fail_closed_started(self, error: Exception, span: Any) -> NoReturn:
        """Map a started-hook evaluation failure to a framework-native HALT.

        Reached only when the failure must stop the operation: the client
        raises ``GovernanceAPIError`` solely under ``on_api_error=fail_closed``
        (fail-open returns an allow-shaped fallback instead), and started-hook
        ``ContractError``s always fail closed — a payload we cannot express to
        Core must not let the operation run ungoverned. Raising the raw error
        would leave frameworks treating it as a generic (often retryable)
        failure; routing a HALT-shaped result through the adapter preserves
        the non-retryable halt semantics.
        """
        halt = EvaluationResult(
            verdict=Verdict.HALT,
            reason=f"Governance evaluation failed closed: {error}",
            fallback_used=True,
            raw={"fail_closed_error": str(error), "error_type": type(error).__name__},
        )
        self._mark_stopped(halt, span)
        self._adapter.raise_hook_blocked(halt)  # NoReturn by contract
        raise GovernanceBlockedError(
            halt.verdict, halt.reason or "Blocked (adapter returned)"
        )

    def _decide_started(self, result: EvaluationResult, span: Any, *, sync: bool) -> bool:
        verdict = result.verdict
        if verdict.should_stop():
            self._mark_stopped(result, span)
            self._adapter.raise_hook_blocked(result)  # NoReturn by contract
            # Defense in depth: a misbehaving adapter that RETURNS from its
            # NoReturn callback must not fall through to run the operation.
            raise GovernanceBlockedError(
                result.verdict, result.reason or "Blocked (adapter returned)"
            )
        if verdict.requires_approval():
            return self._sync_approval(result, span)
        return True

    async def _adecide_started(self, result: EvaluationResult, span: Any) -> bool:
        verdict = result.verdict
        if verdict.should_stop():
            self._mark_stopped(result, span)
            self._adapter.raise_hook_blocked(result)  # NoReturn by contract
            # Defense in depth: a misbehaving adapter that RETURNS from its
            # NoReturn callback must not fall through to run the operation.
            raise GovernanceBlockedError(
                result.verdict, result.reason or "Blocked (adapter returned)"
            )
        if verdict.requires_approval():
            # Adapter drives its native approval flow; returning ⇒ approved.
            # Core omits the workflow/run/activity IDs from the evaluate
            # response, so hand the span-resolved context for the poll.
            if self._approval_accepts_context:
                await self._adapter.handle_approval(
                    result, context=resolve_context(self._store, span)
                )
            else:
                await self._adapter.handle_approval(result)
            return True
        return True

    def _sync_approval(self, result: EvaluationResult, span: Any) -> bool:
        """Sync approval: adapter-native flow first, core poller fallback.

        An adapter exposing ``handle_approval_sync`` owns the flow. Returning
        normally means approved. Without that seam, drive the core poller,
        keyed by the activity — Core's verdict carries no approval_id. No
        poller / no activity context ⇒ fail safe: blocked (the operation must
        not run on an unresolved approval).
        """
        ctx = resolve_context(self._store, span)
        adapter_sync = getattr(self._adapter, "handle_approval_sync", None)
        if adapter_sync is not None:
            # Pass the span-resolved context: ambient ContextVar lookup can
            # miss in user-spawned threads, and the adapter needs the context
            # for skip-HITL decisions and framework buffer correlation.
            adapter_sync(result, context=ctx)
            return True
        if self._sync_poller is None or ctx is None:
            self._mark_stopped(result, span)
            self._adapter.raise_hook_blocked(result)  # NoReturn by contract
            # Defense in depth: a misbehaving adapter that RETURNS from its
            # NoReturn callback must not fall through to run the operation.
            raise GovernanceBlockedError(
                result.verdict, result.reason or "Blocked (adapter returned)"
            )
        approval = self._sync_poller.wait_for_decision(
            ctx.workflow_id or "", ctx.run_id or "", ctx.activity_id or ""
        )
        if approval.allow_shaped:
            return True
        self._mark_stopped(result, span)
        self._adapter.raise_hook_blocked(result)  # NoReturn by contract
        raise GovernanceBlockedError(
            result.verdict, result.reason or "Blocked (adapter returned)"
        )

    def _mark_stopped(self, result: EvaluationResult, span: Any) -> None:
        ctx = resolve_context(self._store, span)
        if ctx is not None:
            self._store.mark_activity_aborted(ctx.workflow_id, ctx.activity_id)
        if result.verdict is Verdict.HALT:
            # Expose the halt request; the framework adapter decides how to
            # stop future work.
            self._store.request_halt()

    # ── Completed (telemetry stage) ───────────────────────────────────────

    def completed(
        self,
        span: Any,
        *,
        hook_type: HookType,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        """Evaluate AFTER the operation ran. Never raises to the caller and
        never undoes the operation — stop verdicts only mark FUTURE execution
        blocked (abort/halt flags + adapter callback)."""
        if not self._runtime.config.instrumentation.completed_telemetry_enabled:
            return
        event = build_hook_event(
            self._store, span, stage=Stage.COMPLETED, hook_type=hook_type, fields=fields
        )
        if event is None:
            return
        try:
            result = self._gate.completed(event)
        except Exception:
            logger.warning("completed-hook telemetry failed", exc_info=True)
            return
        self._after_completed(result, span)

    async def acompleted(
        self,
        span: Any,
        *,
        hook_type: HookType,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        """Async :meth:`completed`."""
        if not self._runtime.config.instrumentation.completed_telemetry_enabled:
            return
        event = build_hook_event(
            self._store, span, stage=Stage.COMPLETED, hook_type=hook_type, fields=fields
        )
        if event is None:
            return
        try:
            result = await self._gate.acompleted(event)
        except Exception:
            logger.warning("completed-hook telemetry failed", exc_info=True)
            return
        self._after_completed(result, span)

    def _after_completed(self, result: EvaluationResult, span: Any) -> None:
        if result.verdict.should_stop():
            self._mark_stopped(result, span)  # future execution only
        # Hand the span-resolved context so adapters can bridge a completed
        # BLOCK/HALT to native effects on the correct run/activity.
        ctx = resolve_context(self._store, span)
        try:
            if self._completed_accepts_context:
                self._adapter.on_completed_hook_result(result, context=ctx)
            else:
                self._adapter.on_completed_hook_result(result)
        except Exception:
            logger.warning("adapter.on_completed_hook_result failed", exc_info=True)
