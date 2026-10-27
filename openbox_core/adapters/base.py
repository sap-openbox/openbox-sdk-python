"""FrameworkAdapter protocol + the core default adapter.

The adapter is the ONE seam where governance verdicts become framework-native
effects. There is exactly one path: wrapper -> hook runtime -> adapter.
Framework SDKs override the callbacks; the default ``CoreAdapter`` raises the
core error types directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..contracts.results import EvaluationResult
from ..errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    GovernanceBlockedError,
    GovernanceHaltError,
)

if TYPE_CHECKING:
    from typing import NoReturn

    from ..approvals import ApprovalPoller
    from ..contracts.context import ActivityContext

__all__ = ["FrameworkAdapter", "CoreAdapter", "adapter_accepts_context"]


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Native-enforcement callbacks implemented by framework SDKs."""

    name: str

    async def handle_approval(
        self, result: EvaluationResult, context: ActivityContext | None = None
    ) -> None:
        """Drive the framework's approval flow for REQUIRE_APPROVAL.

        Return normally when approved; raise the framework's native rejection/
        expiry error otherwise. Called BEFORE the real operation runs.

        ``context`` carries the workflow/run/activity IDs the approval poll
        needs: Core's evaluate response does NOT echo them, so they cannot be
        recovered from ``result.raw``. The runtime passes the originating
        ``ActivityContext`` (lifecycle events build one from the event).
        Optional — an adapter defined as ``handle_approval(self, result)`` still
        conforms; the runtime passes ``context`` only when the signature accepts
        it (see :func:`adapter_accepts_context`).

        Adapters may ALSO define a plain-sync ``handle_approval_sync(result,
        context)`` (not part of the required protocol): when present, sync hook
        paths delegate to it instead of driving the core inline poller.
        Frameworks with retry-based HITL can raise their native pending error
        there.
        """
        ...

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> NoReturn:
        """Produce the framework-native effect for a BLOCK/HALT lifecycle verdict."""
        ...

    def raise_hook_blocked(self, result: EvaluationResult) -> NoReturn:
        """Produce the framework-native effect for a BLOCK/HALT started-hook
        verdict. The real operation has NOT run."""
        ...

    def on_completed_hook_result(
        self, result: EvaluationResult, context: ActivityContext | None = None
    ) -> None:
        """React to a completed-hook verdict. The operation ALREADY ran —
        implementations may only affect FUTURE execution (e.g. mark the
        activity/session blocked); they must never pretend to undo work.

        ``context`` is the span-resolved ActivityContext (may be None). Frameworks
        that must bridge a completed BLOCK/HALT to native effects on the correct
        run/activity read the workflow/run/activity keys from it."""
        ...


class CoreAdapter:
    """Default adapter — raises core error types (framework-agnostic).

    Args:
        approval_poller: Optional ApprovalPoller enabling a real HITL wait.
            Without one, REQUIRE_APPROVAL is fail-safe: rejected (the operation
            does not run) rather than silently allowed.
    """

    name = "core"

    def __init__(self, approval_poller: ApprovalPoller | None = None):
        self._poller = approval_poller

    async def handle_approval(
        self, result: EvaluationResult, context: ActivityContext | None = None
    ) -> None:
        if self._poller is None:
            raise ApprovalRejectedError(
                "REQUIRE_APPROVAL verdict but no approval flow is configured — "
                "failing safe (operation not run)"
            )
        # The approval is polled by workflow/run/activity id. Core's verdict
        # carries no approval_id, so that is not required here.
        workflow_id, run_id, activity_id = _approval_poll_ids(result, context)
        if not (workflow_id and run_id and activity_id):
            raise ApprovalRejectedError(
                "REQUIRE_APPROVAL verdict without the workflow/run/activity ids to "
                "poll — failing safe (operation not run)"
            )
        approval = await self._poller.await_decision(workflow_id, run_id, activity_id)
        if approval.allow_shaped:
            return
        if approval.expired:
            raise ApprovalExpiredError(approval.reason or "Approval window expired")
        raise ApprovalRejectedError(approval.reason or "Approval rejected")

    def raise_lifecycle_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_stop(result)

    def raise_hook_blocked(self, result: EvaluationResult) -> NoReturn:
        self._raise_stop(result)

    def on_completed_hook_result(
        self, result: EvaluationResult, context: ActivityContext | None = None
    ) -> None:
        # Completed telemetry never undoes the operation; the runtime records
        # abort/halt flags for FUTURE execution — nothing to do here.
        return None

    @staticmethod
    def _raise_stop(result: EvaluationResult) -> NoReturn:
        from ..contracts.results import Verdict

        if result.verdict is Verdict.HALT:
            raise GovernanceHaltError(result.reason or "Halted by governance policy")
        raise GovernanceBlockedError(
            result.verdict, result.reason or "Blocked by governance policy"
        )


def _approval_poll_ids(
    result: EvaluationResult, context: ActivityContext | None
) -> tuple[str, str, str]:
    """Resolve the (workflow_id, run_id, activity_id) the approval poll sends.

    Core's evaluate response does not echo them, so the originating
    ``ActivityContext`` is authoritative; ``result.raw`` is only a fallback for
    a caller that predates the ``context`` argument (it is empty in real Core
    traffic).
    """
    raw = result.raw
    if context is None:
        return (
            raw.get("workflow_id", ""),
            raw.get("run_id", ""),
            raw.get("activity_id", ""),
        )
    return (
        context.workflow_id or raw.get("workflow_id", ""),
        context.run_id or raw.get("run_id", ""),
        context.activity_id or raw.get("activity_id", ""),
    )


def adapter_accepts_context(callback: Callable[..., Any] | None) -> bool:
    """True when an adapter callback accepts a ``context`` argument (an explicit
    parameter or ``**kwargs``).

    Checked by signature so a genuine ``TypeError`` raised inside the callback
    body is never mistaken for an arity mismatch and silently dropped. Lets the
    runtime stay backward-compatible with adapters written against the older
    ``handle_approval(self, result)`` / ``on_completed_hook_result(self,
    result)`` signatures.
    """
    import inspect

    if callback is None:
        return False
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    return "context" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
