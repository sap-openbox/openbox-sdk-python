"""OpenRouter routing provenance — record shaping, scheduling and the session drain."""

import json
import threading
from types import SimpleNamespace

import pytest

from openbox_core import client as client_module
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.otel_spans import HookType
from openbox_core.instrumentation import openrouter_provenance as prov

GEN_ID = "gen-1761-abcDEF_01"
REQUEST_BODY = json.dumps(
    {
        "model": "openai/gpt-4o-mini",
        "provider": {"only": ["openai"], "order": ["openai", "azure"], "allow_fallbacks": False},
    }
)
RESPONSE_BODY = json.dumps(
    {
        "id": GEN_ID,
        "provider": "OpenAI",
        "model": "openai/gpt-4o-mini-2024-07-18",
        "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "cost": 0.0001,
            "is_byok": False,
            "cost_details": {"upstream_inference_cost": 0.00008},
        },
    }
)
GENERATION_RECORD = {
    "data": {
        "id": GEN_ID,
        "provider_name": "OpenAI",
        "model": "openai/gpt-4o-mini-2024-07-18",
        "data_region": "global",
        "total_cost": 0.0001,
        "upstream_inference_cost": 0.00008,
        "tokens_prompt": 12,
        "tokens_completion": 3,
        "latency": 640,
        "finish_reason": "stop",
        "is_byok": False,
        "provider_responses": [
            {"provider_name": "Azure", "status": 429, "latency": 90},
            {"provider_name": "OpenAI", "status": 200, "latency": 550},
        ],
    }
}
CONTEXT = ActivityContext(
    workflow_id="wf-1",
    run_id="run-1",
    workflow_type="support-agent",
    task_queue="q",
    activity_id="act-1",
    activity_type="llm_call",
)


def _request(url="https://openrouter.ai/api/v1/chat/completions", authorization="Bearer sk-or-test"):
    headers = {"authorization": authorization} if authorization else {}
    return SimpleNamespace(url=url, headers=headers)


class _RecordingRuntime:
    def __init__(self):
        self._store = None
        self.calls = []

    def completed(self, span, *, hook_type, fields):
        self.calls.append({"span": span, "hook_type": hook_type, "fields": fields})


@pytest.fixture
def captured(monkeypatch):
    """Run the attestation inline and record what reaches the runtime."""
    runtime = _RecordingRuntime()
    monkeypatch.setattr(prov, "activity_scope", lambda *_a, **_k: _NullScope())
    return runtime


class _NullScope:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRecordShaping:
    def test_generation_id_from_json_and_sse(self):
        assert prov.extract_generation_id(RESPONSE_BODY) == GEN_ID
        sse = f'data: {{"id":"{GEN_ID}","choices":[]}}\n\ndata: [DONE]\n'
        assert prov.extract_generation_id(sse) == GEN_ID
        assert prov.extract_generation_id('{"id":"chatcmpl-1"}') is None
        assert prov.extract_generation_id(None) is None

    def test_requested_routing(self):
        assert prov.extract_requested_routing(REQUEST_BODY) == {
            "model": "openai/gpt-4o-mini",
            "only": ["openai"],
            "order": ["openai", "azure"],
            "allow_fallbacks": False,
        }
        assert prov.extract_requested_routing("not json") == {}

    def test_honored_checks(self):
        requested = prov.extract_requested_routing(REQUEST_BODY)
        assert prov.is_routing_honored("OpenAI", requested) is True
        assert prov.is_routing_honored("Azure", requested) is False
        assert prov.is_routing_honored("OpenAI", {}) is None
        assert prov.is_model_honored("openai/gpt-4o-mini-2024-07-18", requested) is True
        assert prov.is_model_honored("mistralai/mistral-large", requested) is False
        assert prov.is_model_honored("openai/gpt-4o", {"model": "openrouter/auto"}) is None

    def test_generation_record_attributes(self):
        requested = prov.extract_requested_routing(REQUEST_BODY)
        attrs = prov.provenance_attributes(
            prov.normalize_generation_record(GEN_ID, GENERATION_RECORD, requested)
        )
        assert attrs["gen_ai.generation.id"] == GEN_ID
        assert attrs["gen_ai.upstream.provider"] == "OpenAI"
        assert attrs["gen_ai.upstream.data_region"] == "global"
        assert attrs["gen_ai.upstream.latency_ms"] == 640
        assert attrs["gen_ai.routing.fallback_attempts"] == 2
        assert attrs["gen_ai.routing.providers_tried"] == ["Azure", "OpenAI"]
        assert attrs["openbox.routing.honored"] is True
        assert attrs["openbox.routing.allow_fallbacks"] is False
        assert attrs["openbox.model.honored"] is True

    def test_response_record_leaves_record_only_fields_out(self):
        attrs = prov.provenance_attributes(
            prov.record_from_response(GEN_ID, RESPONSE_BODY, prov.extract_requested_routing(REQUEST_BODY))
        )
        assert attrs["gen_ai.upstream.provider"] == "OpenAI"
        assert attrs["gen_ai.usage.total_cost"] == 0.0001
        assert attrs["gen_ai.usage.upstream_cost"] == 0.00008
        assert attrs["gen_ai.usage.tokens_prompt"] == 12
        assert attrs["gen_ai.response.finish_reason"] == "stop"
        assert attrs["gen_ai.upstream.is_byok"] is False
        for absent in ("gen_ai.upstream.data_region", "gen_ai.upstream.latency_ms", "gen_ai.routing.attempts"):
            assert absent not in attrs
        assert None not in attrs.values()

    def test_response_record_from_sse_stream(self):
        sse = "\n".join(
            [
                f'data: {{"id":"{GEN_ID}","provider":"OpenAI","model":"openai/gpt-4o-mini","choices":[]}}',
                f'data: {{"id":"{GEN_ID}","choices":[{{"finish_reason":"tool_calls"}}],'
                '"usage":{"prompt_tokens":5,"completion_tokens":2,"cost":0.00002}}',
                "data: [DONE]",
            ]
        )
        record = prov.record_from_response(GEN_ID, sse, {})
        assert record["provider"] == "OpenAI"
        assert record["finish_reason"] == "tool_calls"
        assert record["tokens_completion"] == 2


class TestAttest:
    def test_prefers_the_generation_record(self, captured, monkeypatch):
        monkeypatch.setattr(prov, "_fetch_generation_record", lambda *_a: GENERATION_RECORD)
        prov._attest(captured, CONTEXT, None, GEN_ID, "Bearer k", "https://openrouter.ai", {}, None)
        (call,) = captured.calls
        assert call["hook_type"] is HookType.LLM_PROVENANCE
        assert call["fields"]["attributes"]["gen_ai.upstream.data_region"] == "global"
        assert call["fields"]["duration_ns"] == 640_000_000
        assert call["fields"]["name"] == "served by OpenAI (openai/gpt-4o-mini-2024-07-18)"

    def test_falls_back_to_the_response_when_the_record_is_late(self, captured, monkeypatch):
        monkeypatch.setattr(prov, "_fetch_generation_record", lambda *_a: None)
        from_response = prov.record_from_response(GEN_ID, RESPONSE_BODY, {})
        prov._attest(captured, CONTEXT, None, GEN_ID, "Bearer k", "https://openrouter.ai", {}, from_response)
        (call,) = captured.calls
        attrs = call["fields"]["attributes"]
        assert attrs["gen_ai.generation.id"] == GEN_ID
        assert attrs["gen_ai.upstream.provider"] == "OpenAI"
        assert "gen_ai.upstream.data_region" not in attrs
        assert call["fields"]["duration_ns"] is None

    def test_sends_nothing_without_either_source(self, captured, monkeypatch):
        monkeypatch.setattr(prov, "_fetch_generation_record", lambda *_a: None)
        prov._attest(captured, CONTEXT, None, GEN_ID, "Bearer k", "https://openrouter.ai", {}, None)
        assert captured.calls == []


class TestSchedule:
    @pytest.fixture
    def attested(self, monkeypatch):
        calls = []
        monkeypatch.setattr(prov, "_attest", lambda *args: calls.append(args))
        return calls

    def _schedule(self, *, request=None, context=CONTEXT, response=RESPONSE_BODY):
        prov.schedule_openrouter_provenance(
            object(), context, None, request or _request(), REQUEST_BODY, response
        )
        prov.drain_openrouter_provenance(timeout=5)

    def test_schedules_for_a_governed_openrouter_call(self, attested):
        self._schedule()
        (args,) = attested
        assert args[3] == GEN_ID
        assert args[4] == "Bearer sk-or-test"
        assert args[5] == "https://openrouter.ai"
        assert args[7]["provider"] == "OpenAI"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"request": _request(url="https://api.openai.com/v1/chat/completions")},
            {"request": _request(url=f"https://openrouter.ai/api/v1/generation?id={GEN_ID}")},
            {"request": _request(authorization=None)},
            {"context": None},
            {"context": ActivityContext(workflow_id="wf", run_id="r", workflow_type="t", task_queue="q")},
            {"response": '{"id":"chatcmpl-1"}'},
        ],
        ids=["other-host", "generation-lookup", "no-key", "no-context", "no-activity", "no-gen-id"],
    )
    def test_skips(self, attested, kwargs):
        self._schedule(**kwargs)
        assert attested == []

    def test_env_toggle_disables(self, attested, monkeypatch):
        monkeypatch.setenv("OPENBOX_OPENROUTER_PROVENANCE", "false")
        self._schedule()
        assert attested == []

    def test_attest_errors_never_raise(self, monkeypatch):
        def boom(*_args):
            raise RuntimeError("lookup exploded")

        monkeypatch.setattr(prov, "_attest", boom)
        self._schedule()
        assert not prov._pending


class TestSessionDrain:
    def test_closing_events_wait_for_pending_lookups(self, monkeypatch):
        release = threading.Event()
        finished = []

        def slow_attest(*_args):
            release.wait(5)
            finished.append(True)

        monkeypatch.setattr(prov, "_attest", slow_attest)
        prov.schedule_openrouter_provenance(object(), CONTEXT, None, _request(), REQUEST_BODY, RESPONSE_BODY)
        threading.Timer(0.2, release.set).start()

        client_module._drain_provenance()
        assert finished == [True]

    @pytest.mark.parametrize(
        ("event_type", "closes"),
        [("WorkflowCompleted", True), ("WorkflowFailed", True), ("ActivityCompleted", False)],
    )
    def test_which_events_close_the_session(self, event_type, closes):
        assert client_module._closes_session({"event_type": event_type}) is closes
