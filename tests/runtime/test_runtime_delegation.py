"""OpenBoxRuntime delegation tests — block/halt/approval route to the adapter."""

import httpx
import pytest

from openbox_core.adapters.base import CoreAdapter, FrameworkAdapter
from openbox_core.approvals import ApprovalPoller
from openbox_core.client import EvaluationClient
from openbox_core.config import OpenBoxConfig
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.events import workflow_started
from openbox_core.contracts.results import ApprovalResult, EvaluationResult, Verdict
from openbox_core.errors import (
    ApprovalRejectedError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
)
from openbox_core.runtime import OpenBoxRuntime

WF = dict(workflow_id="wf-1", run_id="r-1", workflow_type="W")


class RecordingAdapter:
    """Fake FrameworkAdapter recording every delegation."""

    name = "recording"

    def __init__(self):
        self.lifecycle_blocked: list[EvaluationResult] = []
        self.hook_blocked: list[EvaluationResult] = []
        self.approvals: list[EvaluationResult] = []
        self.completed: list[EvaluationResult] = []

    async def handle_approval(self, result):
        self.approvals.append(result)

    def raise_lifecycle_blocked(self, result):
        self.lifecycle_blocked.append(result)
        raise GovernanceBlockedError(result.verdict, result.reason or "blocked")

    def raise_hook_blocked(self, result):
        self.hook_blocked.append(result)
        raise GovernanceBlockedError(result.verdict, result.reason or "blocked")

    def on_completed_hook_result(self, result):
        self.completed.append(result)


class _IdRecordingClient:
    """Fake client recording the (workflow_id, run_id, activity_id) each async
    poll is called with, then returning an allow-shaped decision."""

    def __init__(self, seen: list[tuple[str, str, str]]):
        self._seen = seen

    async def apoll_approval(self, workflow_id, run_id, activity_id):
        self._seen.append((workflow_id, run_id, activity_id))
        return ApprovalResult.from_dict({"action": "allow"})


class _ContextRecordingAdapter(RecordingAdapter):
    """RecordingAdapter whose handle_approval ACCEPTS and records ``context``."""

    def __init__(self):
        super().__init__()
        self.approval_contexts: list[ActivityContext | None] = []

    async def handle_approval(self, result, context=None):
        self.approvals.append(result)
        self.approval_contexts.append(context)


def make_runtime(response_json, adapter=None):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=response_json))
    client = EvaluationClient(
        "https://core.test", "obx_test_x", transport=transport, async_transport=transport
    )
    config = OpenBoxConfig(api_url="https://core.test", api_key="obx_test_x")
    return OpenBoxRuntime(config, adapter, client=client)


class TestLifecycleDelegation:
    def test_allow_returns_result(self):
        runtime = make_runtime({"verdict": "allow"}, RecordingAdapter())
        result = runtime.evaluate_lifecycle(workflow_started(**WF))
        assert result.verdict is Verdict.ALLOW

    def test_block_delegates_to_adapter(self):
        adapter = RecordingAdapter()
        runtime = make_runtime({"verdict": "block", "reason": "no"}, adapter)
        with pytest.raises(GovernanceBlockedError):
            runtime.evaluate_lifecycle(workflow_started(**WF))
        assert len(adapter.lifecycle_blocked) == 1
        assert adapter.lifecycle_blocked[0].verdict is Verdict.BLOCK

    def test_halt_delegates_and_sets_halt_flag(self):
        adapter = RecordingAdapter()
        runtime = make_runtime({"verdict": "halt", "reason": "kill"}, adapter)
        with pytest.raises(GovernanceBlockedError):  # adapter's chosen error
            runtime.evaluate_lifecycle(workflow_started(**WF))
        assert runtime.context_store.halt_requested is True

    def test_guardrails_failure_raises(self):
        response = {
            "verdict": "allow",
            "guardrails_result": {"validation_passed": False, "reasons": [{"reason": "pii"}]},
        }
        runtime = make_runtime(response, RecordingAdapter())
        with pytest.raises(GuardrailsValidationError, match="pii"):
            runtime.evaluate_lifecycle(workflow_started(**WF))

    async def test_require_approval_drives_adapter_async(self):
        adapter = RecordingAdapter()
        runtime = make_runtime({"verdict": "require_approval", "approval_id": "app-1"}, adapter)
        result = await runtime.aevaluate_lifecycle(workflow_started(**WF))
        assert result.verdict is Verdict.REQUIRE_APPROVAL
        assert len(adapter.approvals) == 1

    async def test_require_approval_threads_event_context(self):
        # A context-accepting adapter receives the event's workflow/run IDs —
        # Core's evaluate response never echoes them, so the runtime must supply
        # them from the originating event.
        adapter = _ContextRecordingAdapter()
        runtime = make_runtime({"verdict": "require_approval", "approval_id": "app-1"}, adapter)
        await runtime.aevaluate_lifecycle(workflow_started(**WF))
        ctx = adapter.approval_contexts[0]
        assert ctx is not None
        assert (ctx.workflow_id, ctx.run_id) == ("wf-1", "r-1")

    def test_sync_require_approval_returned_to_caller(self):
        adapter = RecordingAdapter()
        runtime = make_runtime({"verdict": "require_approval"}, adapter)
        result = runtime.evaluate_lifecycle(workflow_started(**WF))
        assert result.verdict is Verdict.REQUIRE_APPROVAL
        assert adapter.approvals == []  # sync path never drives async approval


class TestCoreAdapterDefaults:
    def test_core_adapter_raises_block(self):
        adapter = CoreAdapter()
        with pytest.raises(GovernanceBlockedError):
            adapter.raise_hook_blocked(EvaluationResult(verdict=Verdict.BLOCK, reason="x"))

    def test_core_adapter_raises_halt(self):
        adapter = CoreAdapter()
        with pytest.raises(GovernanceHaltError):
            adapter.raise_lifecycle_blocked(EvaluationResult(verdict=Verdict.HALT))

    async def test_core_adapter_approval_fails_safe_without_poller(self):
        adapter = CoreAdapter()
        with pytest.raises(ApprovalRejectedError, match="failing safe"):
            await adapter.handle_approval(
                EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, approval_id="a")
            )

    async def test_core_adapter_polls_with_context_ids_not_raw(self):
        # raw is EMPTY, exactly as a real Core evaluate response is — the poll
        # IDs must come from the context the runtime threads in.
        seen: list[tuple[str, str, str]] = []
        poller = ApprovalPoller(_IdRecordingClient(seen), poll_interval_seconds=0.001)
        adapter = CoreAdapter(approval_poller=poller)
        result = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL, approval_id="a")
        ctx = ActivityContext(workflow_id="wf-9", run_id="run-9", activity_id="act-9")
        await adapter.handle_approval(result, context=ctx)
        assert seen == [("wf-9", "run-9", "act-9")]

    async def test_core_adapter_approval_falls_back_to_raw_without_context(self):
        seen: list[tuple[str, str, str]] = []
        poller = ApprovalPoller(_IdRecordingClient(seen), poll_interval_seconds=0.001)
        adapter = CoreAdapter(approval_poller=poller)
        result = EvaluationResult(
            verdict=Verdict.REQUIRE_APPROVAL,
            approval_id="a",
            raw={"workflow_id": "wf-raw", "run_id": "run-raw", "activity_id": "act-raw"},
        )
        await adapter.handle_approval(result)
        assert seen == [("wf-raw", "run-raw", "act-raw")]

    async def test_core_adapter_polls_without_approval_id(self):
        # Core's REQUIRE_APPROVAL verdict carries no approval_id; the poll is
        # keyed by the activity, so the wait must still happen.
        seen: list[tuple[str, str, str]] = []
        poller = ApprovalPoller(_IdRecordingClient(seen), poll_interval_seconds=0.001)
        adapter = CoreAdapter(approval_poller=poller)
        ctx = ActivityContext(workflow_id="wf-9", run_id="run-9", activity_id="act-9")
        await adapter.handle_approval(EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL), context=ctx)
        assert seen == [("wf-9", "run-9", "act-9")]

    async def test_core_adapter_approval_fails_safe_without_poll_ids(self):
        seen: list[tuple[str, str, str]] = []
        poller = ApprovalPoller(_IdRecordingClient(seen), poll_interval_seconds=0.001)
        adapter = CoreAdapter(approval_poller=poller)
        with pytest.raises(ApprovalRejectedError, match="failing safe"):
            await adapter.handle_approval(EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL))
        assert seen == []

    def test_protocol_conformance(self):
        assert isinstance(CoreAdapter(), FrameworkAdapter)
        assert isinstance(RecordingAdapter(), FrameworkAdapter)


class TestRuntimeLifecycleManagement:
    def test_close_clears_context_and_transport(self):
        from openbox_core.context import ContextStore
        from openbox_core.contracts.context import ActivityContext

        store = ContextStore()
        runtime = make_runtime({"verdict": "allow"})
        runtime.context_store = store
        store.register_trace(1, ActivityContext(workflow_id="wf"))
        runtime.close()
        assert store.trace_map_size() == 0

    async def test_aclose(self):
        runtime = make_runtime({"verdict": "allow"})
        await runtime.aclose()

    def test_uninstall_without_install_is_noop(self):
        make_runtime({"verdict": "allow"}).uninstall_instrumentation()
