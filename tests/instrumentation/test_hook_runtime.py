"""HookRuntime decision-point tests — the single wrapper->runtime->adapter path."""

import sys

import pytest
from conftest import ACTIVITY_CTX, FakeCore, build_runtime

from openbox_core.contracts.otel_spans import HookType
from openbox_core.contracts.results import Verdict
from openbox_core.errors import GovernanceBlockedError, GovernanceHaltError
from openbox_core.hooks.preflight import HookRuntime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "wire"))
from span_fixtures import FakeSpan  # noqa: E402


class TestSkipSemantics:
    def test_no_bound_context_skips_not_errors(self, hook_runtime, fake_core):
        proceed = hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        assert proceed is True
        assert fake_core.payloads == []  # nothing sent

    def test_preflight_disabled_skips(self, fake_core, adapter, store):
        runtime = build_runtime(fake_core, adapter, store, preflight_enabled=False)
        hook_runtime = HookRuntime(runtime)
        token = store.bind(ACTIVITY_CTX)
        try:
            assert hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST) is True
            assert fake_core.payloads == []
        finally:
            store.reset(token)

    def test_context_without_activity_binding_skips(self, hook_runtime, fake_core, store):
        from openbox_core.contracts.context import ActivityContext

        token = store.bind(ActivityContext(workflow_id="wf-only"))
        try:
            assert hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST) is True
            assert fake_core.payloads == []
        finally:
            store.reset(token)


class TestStartedVerdicts:
    def _bound(self, store):
        return store.bind(ACTIVITY_CTX)

    def test_allow_proceeds_and_sends_started(self, fake_core, adapter, store):
        fake_core.queue = [{"verdict": "allow"}]
        runtime = build_runtime(fake_core, adapter, store)
        hook_runtime = HookRuntime(runtime)
        token = self._bound(store)
        try:
            assert hook_runtime.preflight(
                FakeSpan(), hook_type=HookType.HTTP_REQUEST,
                fields={"http_method": "GET", "http_url": "https://x"},
            ) is True
        finally:
            store.reset(token)
        payload = fake_core.payloads[0]
        assert payload["event_type"] == "ActivityStarted"
        assert payload["hook_trigger"] is True
        assert payload["span_count"] == 1
        assert payload["spans"][0]["stage"] == "started"
        assert payload["spans"][0]["hook_type"] == "http_request"
        assert "otel" not in payload["spans"][0]

    def test_block_delegates_and_marks_abort(self, fake_core, adapter, store):
        fake_core.queue = [{"verdict": "block", "reason": "denied"}]
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = self._bound(store)
        try:
            with pytest.raises(GovernanceBlockedError, match="denied"):
                hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert adapter.hook_blocked[0].verdict is Verdict.BLOCK
        assert store.is_activity_aborted(ACTIVITY_CTX.workflow_id, ACTIVITY_CTX.activity_id)

    def test_halt_sets_halt_flag(self, fake_core, adapter, store):
        fake_core.queue = [{"verdict": "halt", "reason": "kill"}]
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = self._bound(store)
        try:
            with pytest.raises(GovernanceHaltError):
                hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert store.halt_requested is True

    def test_abort_short_circuit_no_network(self, fake_core, adapter, store):
        store.mark_activity_aborted(ACTIVITY_CTX.workflow_id, ACTIVITY_CTX.activity_id)
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = self._bound(store)
        try:
            with pytest.raises(GovernanceBlockedError, match="prior hook"):
                hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert fake_core.payloads == []  # short-circuited BEFORE any send


class TestApprovalFlows:
    def test_sync_approval_allowed_proceeds(self, store, adapter):
        fake_core = FakeCore(
            {"verdict": "require_approval", "approval_id": "app-1"},  # evaluate
            {"action": "allow"},  # approval poll
        )
        runtime = build_runtime(fake_core, adapter, store)
        runtime.config.hitl.poll_interval_ms = 0
        hook_runtime = HookRuntime(runtime)
        token = store.bind(ACTIVITY_CTX)
        try:
            assert hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST) is True
        finally:
            store.reset(token)

    def test_sync_approval_rejected_blocks(self, store, adapter):
        fake_core = FakeCore(
            {"verdict": "require_approval", "approval_id": "app-1"},
            {"action": "block", "reason": "human said no"},
        )
        runtime = build_runtime(fake_core, adapter, store)
        runtime.config.hitl.poll_interval_ms = 0
        hook_runtime = HookRuntime(runtime)
        token = store.bind(ACTIVITY_CTX)
        try:
            with pytest.raises(GovernanceBlockedError):
                hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert store.is_activity_aborted(ACTIVITY_CTX.workflow_id, ACTIVITY_CTX.activity_id)

    def test_sync_approval_without_hitl_fails_safe(self, store, adapter):
        fake_core = FakeCore({"verdict": "require_approval"})
        runtime = build_runtime(fake_core, adapter, store)
        runtime.config.hitl.enabled = False
        hook_runtime = HookRuntime(runtime)
        token = store.bind(ACTIVITY_CTX)
        try:
            with pytest.raises(GovernanceBlockedError):
                hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)

    def test_sync_approval_polls_without_approval_id(self, store, adapter):
        # Core's REQUIRE_APPROVAL verdict carries no approval_id; the poll is
        # keyed by the bound activity.
        fake_core = FakeCore({"verdict": "require_approval"}, {"action": "allow"})
        runtime = build_runtime(fake_core, adapter, store)
        runtime.config.hitl.poll_interval_ms = 0
        hook_runtime = HookRuntime(runtime)
        token = store.bind(ACTIVITY_CTX)
        try:
            assert hook_runtime.preflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST) is True
        finally:
            store.reset(token)
        (poll,) = fake_core.approval_requests
        assert poll["activity_id"] == ACTIVITY_CTX.activity_id

    async def test_async_approval_delegates_to_adapter(self, store, adapter):
        fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = store.bind(ACTIVITY_CTX)
        try:
            assert await hook_runtime.apreflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST) is True
        finally:
            store.reset(token)
        assert len(adapter.approvals) == 1

    async def test_async_approval_rejection_propagates(self, store, adapter):
        from openbox_core.errors import ApprovalRejectedError

        adapter.approve_next = False
        fake_core = FakeCore({"verdict": "require_approval", "approval_id": "app-1"})
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = store.bind(ACTIVITY_CTX)
        try:
            with pytest.raises(ApprovalRejectedError):
                await hook_runtime.apreflight(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)


class TestCompletedTelemetry:
    def test_completed_sends_and_never_raises(self, fake_core, adapter, store):
        fake_core.queue = [{"verdict": "allow"}]
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = store.bind(ACTIVITY_CTX)
        try:
            hook_runtime.completed(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert fake_core.payloads[0]["spans"][0]["stage"] == "completed"
        assert adapter.completed_results[0].verdict is Verdict.ALLOW

    def test_completed_block_marks_future_only(self, fake_core, adapter, store):
        fake_core.queue = [{"verdict": "block", "reason": "post-hoc"}]
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = store.bind(ACTIVITY_CTX)
        try:
            hook_runtime.completed(FakeSpan(), hook_type=HookType.HTTP_REQUEST)  # NO raise
        finally:
            store.reset(token)
        assert store.is_activity_aborted(ACTIVITY_CTX.workflow_id, ACTIVITY_CTX.activity_id)  # future execution blocked
        assert adapter.completed_results[0].verdict is Verdict.BLOCK

    async def test_async_completed(self, fake_core, adapter, store):
        fake_core.queue = [{"verdict": "allow"}]
        hook_runtime = HookRuntime(build_runtime(fake_core, adapter, store))
        token = store.bind(ACTIVITY_CTX)
        try:
            await hook_runtime.acompleted(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert len(fake_core.completed_payloads) == 1

    def test_completed_disabled_by_config(self, fake_core, adapter, store):
        runtime = build_runtime(fake_core, adapter, store, completed_telemetry_enabled=False)
        hook_runtime = HookRuntime(runtime)
        token = store.bind(ACTIVITY_CTX)
        try:
            hook_runtime.completed(FakeSpan(), hook_type=HookType.HTTP_REQUEST)
        finally:
            store.reset(token)
        assert fake_core.payloads == []
