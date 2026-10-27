"""OpenBoxRuntime — owns config, adapter, gate, client, context store, and
(from the instrumentation phase) the instrumentation manager.

The runtime is the composition root for non-sandbox code:

    config  = OpenBoxConfig.resolve(...)
    runtime = OpenBoxRuntime(config, adapter=MyFrameworkAdapter())
    runtime.install_instrumentation()
    ...
    runtime.close()

Lifecycle helpers evaluate through the strict gate and delegate native
enforcement (block/halt/approval) to the adapter callbacks.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .adapters.base import CoreAdapter, FrameworkAdapter, adapter_accepts_context
from .client import EvaluationClient
from .config import OpenBoxConfig
from .context import ContextStore, default_context_store
from .contracts.context import ActivityContext
from .contracts.events import EventEnvelope
from .contracts.results import EvaluationResult, Verdict
from .errors import GuardrailsValidationError
from .gate import GovernanceGate

__all__ = ["OpenBoxRuntime"]


def _default_payload_builder(config: OpenBoxConfig) -> Any:
    """Wire the hook evaluate-body assembler (the single body-shape owner),
    binding the runtime's privacy config into the gate's one-argument seam."""
    from .wire.evaluate_payload import make_payload_builder

    return make_payload_builder(config.privacy)


class OpenBoxRuntime:
    """Composition root wiring config → identity → client → gate → adapter."""

    def __init__(
        self,
        config: OpenBoxConfig,
        adapter: FrameworkAdapter | None = None,
        *,
        client: EvaluationClient | None = None,
        context_store: ContextStore | None = None,
        payload_builder: Any = None,
    ):
        self.config = config
        self.adapter: FrameworkAdapter = adapter if adapter is not None else CoreAdapter()
        # Decide ONCE whether the adapter's handle_approval accepts ``context``
        # (older adapters take only ``result``) — see adapter_accepts_context.
        self._approval_accepts_context = adapter_accepts_context(self.adapter.handle_approval)
        self.context_store = context_store if context_store is not None else default_context_store()
        self.client = client if client is not None else EvaluationClient(
            config.api_url,
            config.api_key,
            timeout_seconds=config.timeout_seconds,
            on_api_error=config.on_api_error,
            # Mutually exclusive by config validation — at most one is non-None.
            identity=config.load_okta_identity() or config.load_identity(),
            # Non-None only in bootstrap mode, where load_okta_identity() returns
            # None because Core has not supplied the metadata yet.
            okta_bootstrap_private_key=config.okta_bootstrap_private_key(),
            workload_private_key=config.keycloak_workload_private_key(),
            sdk_version=config.sdk_version,
            sdk_engine=config.sdk_engine,
            sdk_language=config.sdk_language,
        )
        self.gate = GovernanceGate(
            self.client,
            config,
            payload_builder=payload_builder if payload_builder is not None else _default_payload_builder(config),
        )
        self._instrumentation_manager: Any = None  # set by install_instrumentation

    # ── Instrumentation lifecycle (real body lands with the manager) ──────

    def install_instrumentation(self) -> None:
        """Install generic instrumentation per InstrumentationConfig."""
        if not self.config.instrumentation.enabled:
            return
        if self._instrumentation_manager is None:
            from .instrumentation.manager import InstrumentationManager

            self._instrumentation_manager = InstrumentationManager(self)
        self._instrumentation_manager.install()

    def uninstall_instrumentation(self) -> None:
        """Uninstall instrumentation and restore originals (idempotent)."""
        if self._instrumentation_manager is not None:
            self._instrumentation_manager.uninstall()

    # ── Lifecycle evaluation helpers ──────────────────────────────────────

    def evaluate_lifecycle(self, event: EventEnvelope) -> EvaluationResult:
        """Evaluate + enforce a lifecycle event (sync).

        BLOCK/HALT delegate to ``adapter.raise_lifecycle_blocked``; a
        guardrails failure raises before any approval flow. REQUIRE_APPROVAL
        is returned to the caller on the sync path (approval flows are async —
        use :meth:`aevaluate_lifecycle` to drive the adapter's approval flow).
        """
        result = self.gate.evaluate(event)
        return self._enforce_lifecycle(result, drive_approval=False)

    async def aevaluate_lifecycle(self, event: EventEnvelope) -> EvaluationResult:
        """Async evaluate + enforce, including the adapter approval flow."""
        result = await self.gate.aevaluate(event)
        if result.verdict.requires_approval():
            self._check_guardrails(result)
            # Core omits workflow/run/activity IDs from the evaluate response, so
            # build the approval context from the originating event for the poll.
            if self._approval_accepts_context:
                await self.adapter.handle_approval(
                    result, context=_approval_context_from_event(event)
                )
            else:
                await self.adapter.handle_approval(result)
            return result
        return self._enforce_lifecycle(result, drive_approval=False)

    def _enforce_lifecycle(self, result: EvaluationResult, *, drive_approval: bool) -> EvaluationResult:
        if result.verdict.should_stop():
            if result.verdict is Verdict.HALT:
                self.context_store.request_halt()
            self.adapter.raise_lifecycle_blocked(result)
        self._check_guardrails(result)
        return result

    @staticmethod
    def _check_guardrails(result: EvaluationResult) -> None:
        # Guardrails failure outranks approval so it is never swallowed by HITL.
        if result.guardrails and not result.guardrails.validation_passed:
            reasons = result.guardrails.get_reason_strings()
            raise GuardrailsValidationError(reasons or ["Guardrails validation failed"])

    # ── Shutdown ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Uninstall instrumentation, clear correlation state, close transports.

        Pending OpenRouter provenance lookups are drained first (bounded); they
        send through this runtime's client.
        """
        from .instrumentation.openrouter_provenance import drain_openrouter_provenance

        drain_openrouter_provenance()
        self.uninstall_instrumentation()
        self.context_store.clear()
        self.client.close()

    async def aclose(self) -> None:
        """Async :meth:`close` (also closes the async transport)."""
        from .instrumentation.openrouter_provenance import drain_openrouter_provenance

        await asyncio.to_thread(drain_openrouter_provenance)
        self.uninstall_instrumentation()
        self.context_store.clear()
        await self.client.aclose()


def _approval_context_from_event(event: EventEnvelope) -> ActivityContext:
    """Build the approval context for a lifecycle event.

    ``workflow_id`` / ``run_id`` live in the flat wire ``payload``;
    ``activity_id`` is a first-class envelope field (a workflow-level approval
    legitimately has none). Core's evaluate response omits all three, so the
    poll must be built from the originating event — see
    ``CoreAdapter.handle_approval``.
    """
    payload = event.payload

    def _s(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    activity_id = event.activity_id if event.activity_id is not None else _s(payload.get("activity_id"))
    return ActivityContext(
        workflow_id=_s(payload.get("workflow_id")),
        run_id=_s(payload.get("run_id")),
        activity_id=activity_id,
    )
