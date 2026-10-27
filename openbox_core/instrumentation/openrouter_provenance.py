"""OpenRouter routing provenance — which provider actually served each call.

OpenRouter picks the serving provider per request. Two sources say which one
it was:

- the completion response itself (JSON or SSE): ``provider``, ``model``,
  ``usage.cost``, ``usage.is_byok``, token counts, finish reason — available
  immediately;
- the generation record at ``GET /api/v1/generation?id=<gen-id>``, which adds
  the data region, latency and every fallback attempt — but is published
  late, measured on live anywhere from a few seconds to over a minute.

For each governed model call that goes through OpenRouter, this module reads
the response, tries the generation record for a short, bounded window, and
sends Core one hook span carrying ``gen_ai.generation.id`` plus the
``gen_ai.*`` / ``openbox.*`` attributes below — from the record when it
arrived in time, otherwise from the response alone. Core classifies such a
span as ``llm_provenance`` and the dashboard's Provenance tab aggregates it.

The attribute names and semantics match the OpenRouter TypeScript SDK
(``openbox-openrouter-sdk-ts``, ``src/provenance.ts``) so both SDKs feed the
same panel.

The work runs on a small pool of background threads after the model call has
returned, so it is never on the critical path of the answer. Each lookup has a
fixed time budget counted from the end of the model call; a session's closing
event waits only for that session's lookups, and never longer than the budget.
The lookup reuses the bearer key the model call itself was sent with, so it
needs no extra configuration, and it is best-effort: nothing here ever raises
into the caller.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, wait
from typing import Any
from urllib.parse import quote, urlsplit

from ..context import activity_scope
from ..contracts.otel_spans import HookType

logger = logging.getLogger(__name__)

__all__ = [
    "DRAIN_TIMEOUT_SECONDS",
    "LOOKUP_BUDGET_SECONDS",
    "OPENROUTER_HOSTS",
    "drain_openrouter_provenance",
    "extract_generation_id",
    "extract_requested_routing",
    "is_routing_honored",
    "is_model_honored",
    "normalize_generation_record",
    "provenance_attributes",
    "record_from_response",
    "schedule_openrouter_provenance",
]

OPENROUTER_HOSTS = frozenset({"openrouter.ai"})
GENERATION_PATH = "/api/v1/generation"
# The record is written after the response, not with it: an immediate lookup
# 404s. Seconds to wait before each retry.
_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 3.0, 4.0)
_ATTEMPT_TIMEOUT_SECONDS = 4.0
# Hard cap on one lookup — retries, backoff and request timeouts included —
# counted from the end of the model call, so a backlog only shortens it. When it
# runs out the response-derived record is sent instead.
LOOKUP_BUDGET_SECONDS = 12.0
# A closing event waits this long for its session's lookups: the budget plus
# time to send the span, so the wait never ends before the lookup does.
DRAIN_TIMEOUT_SECONDS = LOOKUP_BUDGET_SECONDS + 5.0
_MAX_WORKERS = 4
_GENERATION_ID = re.compile(r'"id"\s*:\s*"(gen-[A-Za-z0-9_-]+)"')

SessionKey = tuple[str, str]

# Pending lookups per session (workflow_id, run_id).
_pending: dict[SessionKey, set[Future[None]]] = {}
_pending_lock = threading.Lock()
_work: queue.SimpleQueue[tuple[Future[None], Callable[[], None]]] = queue.SimpleQueue()
_workers: list[threading.Thread] = []


def _enabled() -> bool:
    return os.environ.get("OPENBOX_OPENROUTER_PROVENANCE", "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _is_openrouter_call(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return host in OPENROUTER_HOSTS and not parts.path.startswith(GENERATION_PATH)


def extract_generation_id(response_body: str | None) -> str | None:
    """The ``gen-…`` id of a completion, from a JSON body or an SSE stream."""
    if not response_body:
        return None
    try:
        data = json.loads(response_body)
        if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].startswith("gen-"):
            return data["id"]
    except ValueError:
        pass
    match = _GENERATION_ID.search(response_body)
    return match.group(1) if match else None


def extract_requested_routing(request_body: str | None) -> dict[str, Any]:
    """What the caller asked for: model, model fallbacks, provider routing."""
    try:
        data = json.loads(request_body or "")
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    requested: dict[str, Any] = {}
    if isinstance(data.get("model"), str):
        requested["model"] = data["model"]
    if isinstance(data.get("models"), list):
        requested["models"] = [str(m) for m in data["models"]]
    provider = data.get("provider")
    if isinstance(provider, dict):
        for key, name in (("only", "only"), ("order", "order")):
            if isinstance(provider.get(key), list):
                requested[name] = [str(p) for p in provider[key]]
        if isinstance(provider.get("allow_fallbacks"), bool):
            requested["allow_fallbacks"] = provider["allow_fallbacks"]
    return requested


def _same_provider(a: str, b: str) -> bool:
    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    return norm(a) == norm(b)


def is_routing_honored(provider: str | None, requested: dict[str, Any]) -> bool | None:
    """True/False against a ``provider.only`` allowlist; None when none was named."""
    allowed = requested.get("only")
    if not allowed or provider is None:
        return None
    return any(_same_provider(provider, p) for p in allowed)


def is_model_honored(model: str | None, requested: dict[str, Any]) -> bool | None:
    """Did the model that ran match one the caller named? None for auto-routing."""
    asked = requested.get("model")
    if model is None or not asked or asked.startswith("openrouter/auto"):
        return None
    named = [asked, *requested.get("models", [])]

    def base(value: str) -> str:
        return value.lower().split(":", 1)[0]

    # OpenRouter reports dated snapshots (`openai/gpt-4o-mini-2024-07-18`).
    return any(base(model) == base(n) or base(model).startswith(base(n) + "-") for n in named)


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def normalize_generation_record(
    generation_id: str, raw: Any, requested: dict[str, Any]
) -> dict[str, Any]:
    """Flatten OpenRouter's generation record into the fields we attest."""
    root = raw if isinstance(raw, dict) else {}
    record = root.get("data") if isinstance(root.get("data"), dict) else root
    attempts = []
    for entry in record.get("provider_responses") or []:
        if isinstance(entry, dict):
            attempts.append(
                {
                    "provider": _string(entry.get("provider_name")) or _string(entry.get("provider")),
                    "status": entry.get("status"),
                    "latency_ms": _number(entry.get("latency")),
                    "error": _string(entry.get("error")),
                }
            )
    provider = _string(record.get("provider_name"))
    model = _string(record.get("model"))
    return {
        "generation_id": generation_id,
        "provider": provider,
        "model": model,
        "data_region": _string(record.get("data_region")),
        "total_cost": _number(record.get("total_cost")),
        "upstream_cost": _number(record.get("upstream_inference_cost")),
        "tokens_prompt": _number(record.get("tokens_prompt")),
        "tokens_completion": _number(record.get("tokens_completion")),
        "latency_ms": _number(record.get("latency")),
        "finish_reason": _string(record.get("finish_reason")),
        "is_byok": record.get("is_byok") if isinstance(record.get("is_byok"), bool) else None,
        "attempts": attempts,
        "requested": requested,
        "honored": is_routing_honored(provider, requested),
        "model_honored": is_model_honored(model, requested),
    }


def _response_chunks(response_body: str) -> list[dict[str, Any]]:
    """The JSON object(s) of a completion response: one body, or SSE chunks."""
    try:
        data = json.loads(response_body)
        return [data] if isinstance(data, dict) else []
    except ValueError:
        pass
    chunks = []
    for line in response_body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            chunk = json.loads(line[5:].strip())
        except ValueError:
            continue
        if isinstance(chunk, dict):
            chunks.append(chunk)
    return chunks


def record_from_response(
    generation_id: str, response_body: str | None, requested: dict[str, Any]
) -> dict[str, Any] | None:
    """What the completion response itself says about who served it.

    Same shape as :func:`normalize_generation_record`; region, latency and the
    attempt trail are only in the generation record, so they stay None here.
    """
    chunks = _response_chunks(response_body or "")
    if not chunks:
        return None
    provider = model = finish_reason = None
    usage: dict[str, Any] = {}
    for chunk in chunks:
        provider = _string(chunk.get("provider")) or provider
        model = _string(chunk.get("model")) or model
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if isinstance(choice, dict) and _string(choice.get("finish_reason")):
                finish_reason = choice["finish_reason"]
    if provider is None and model is None:
        return None
    cost_details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
    return {
        "generation_id": generation_id,
        "provider": provider,
        "model": model,
        "data_region": None,
        "total_cost": _number(usage.get("cost")),
        "upstream_cost": _number(cost_details.get("upstream_inference_cost")),
        "tokens_prompt": _number(usage.get("prompt_tokens")),
        "tokens_completion": _number(usage.get("completion_tokens")),
        "latency_ms": None,
        "finish_reason": finish_reason,
        "is_byok": usage.get("is_byok") if isinstance(usage.get("is_byok"), bool) else None,
        "attempts": [],
        "requested": requested,
        "honored": is_routing_honored(provider, requested),
        "model_honored": is_model_honored(model, requested),
    }


def provenance_attributes(p: dict[str, Any]) -> dict[str, Any]:
    """Flat, stable span attributes. Nulls are dropped, never sent as null."""
    attrs: dict[str, Any] = {"gen_ai.generation.id": p["generation_id"]}

    def put(key: str, value: Any) -> None:
        if value is not None:
            attrs[key] = value

    put("gen_ai.upstream.provider", p.get("provider"))
    put("gen_ai.response.model", p.get("model"))
    put("gen_ai.upstream.data_region", p.get("data_region"))
    put("gen_ai.usage.total_cost", p.get("total_cost"))
    put("gen_ai.usage.upstream_cost", p.get("upstream_cost"))
    put("gen_ai.usage.tokens_prompt", p.get("tokens_prompt"))
    put("gen_ai.usage.tokens_completion", p.get("tokens_completion"))
    put("gen_ai.upstream.latency_ms", p.get("latency_ms"))
    put("gen_ai.response.finish_reason", p.get("finish_reason"))
    put("gen_ai.upstream.is_byok", p.get("is_byok"))
    attempts = p.get("attempts") or []
    if attempts:
        attrs["gen_ai.routing.fallback_attempts"] = len(attempts)
        attrs["gen_ai.routing.providers_tried"] = [a["provider"] for a in attempts if a.get("provider")]
        attrs["gen_ai.routing.attempts"] = attempts
    requested = p.get("requested") or {}
    put("openbox.routing.requested_only", requested.get("only"))
    put("openbox.routing.requested_order", requested.get("order"))
    put("openbox.routing.allow_fallbacks", requested.get("allow_fallbacks"))
    put("openbox.routing.honored", p.get("honored"))
    put("openbox.model.requested", requested.get("model"))
    put("openbox.model.honored", p.get("model_honored"))
    return attrs


def _fetch_generation_record(
    generation_id: str, authorization: str, base_url: str, deadline: float
) -> tuple[Any, str]:
    """``(record, outcome)``; outcome is ``"found"``, ``"late"`` (not published
    within the budget) or ``"rejected"`` (any other HTTP status)."""
    import httpx

    from .http import suppress_http_instrumentation

    url = f"{base_url}{GENERATION_PATH}?id={quote(generation_id)}"
    with suppress_http_instrumentation(), httpx.Client() as client:
        for attempt in range(len(_BACKOFF_SECONDS) + 1):
            if attempt:
                pause = _BACKOFF_SECONDS[attempt - 1]
                if time.monotonic() + pause >= deadline:
                    break
                time.sleep(pause)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = client.get(
                    url,
                    headers={"Authorization": authorization},
                    timeout=min(_ATTEMPT_TIMEOUT_SECONDS, remaining),
                )
            except httpx.HTTPError as exc:
                logger.debug("generation lookup %s attempt %d failed: %s", generation_id, attempt, exc)
                continue
            logger.debug(
                "generation lookup %s attempt %d: HTTP %d", generation_id, attempt, response.status_code
            )
            if response.status_code == 404:
                continue  # not written yet
            if response.status_code != 200:
                # 401/403 mean the key cannot read generation records; 5xx is
                # OpenRouter's side. Either way, say so rather than "late".
                logger.warning(
                    "OpenRouter generation lookup for %s failed: HTTP %d",
                    generation_id,
                    response.status_code,
                )
                return None, "rejected"
            return response.json(), "found"
    return None, "late"


def _attest(
    runtime: Any,
    context: Any,
    parent_span: Any,
    generation_id: str,
    authorization: str,
    base_url: str,
    requested: dict[str, Any],
    from_response: dict[str, Any] | None,
    deadline: float,
) -> None:
    raw, outcome = _fetch_generation_record(generation_id, authorization, base_url, deadline)
    if raw is not None:
        record = normalize_generation_record(generation_id, raw, requested)
    elif from_response is not None:
        # No record; the response still names provider, model and cost.
        if outcome == "late":
            logger.info(
                "generation record for %s not published in time — sending the response's own provenance",
                generation_id,
            )
        record = from_response
    else:
        logger.warning(
            "routing provenance unavailable for %s — no generation record (%s) and none in the response",
            generation_id,
            outcome,
        )
        return

    from opentelemetry import trace

    name = f"served by {record['provider'] or 'an unreported provider'}"
    if record["model"]:
        name += f" ({record['model']})"
    span = trace.get_tracer("openbox-core.openrouter").start_span(
        name,
        kind=trace.SpanKind.CLIENT,
        context=trace.set_span_in_context(parent_span),
    )
    latency = record.get("latency_ms")
    try:
        with activity_scope(context, store=runtime._store):
            runtime.completed(
                span,
                hook_type=HookType.LLM_PROVENANCE,
                fields={
                    "name": name,
                    # Nested values (the attempt trail) are not legal OTel
                    # attribute values, so the attested set goes in whole here.
                    "attributes": provenance_attributes(record),
                    "duration_ns": int(latency * 1_000_000) if latency is not None else None,
                    "status": {"code": "OK", "description": None},
                },
            )
        logger.info("routing provenance sent for %s (%s)", generation_id, name)
    finally:
        span.end()


def schedule_openrouter_provenance(
    runtime: Any,
    context: Any,
    parent_span: Any,
    request: Any,
    request_body: str | None,
    response_body: str | None,
) -> None:
    """Start collecting provenance for one completed OpenRouter call.

    No-op unless the call went to OpenRouter, inside a governed activity, and
    the response carried a generation id. Returns immediately.
    """
    if not _enabled() or context is None or not context.activity_id or not context.activity_type:
        return
    url = str(getattr(request, "url", "") or "")
    if not _is_openrouter_call(url):
        return
    generation_id = extract_generation_id(response_body)
    authorization = None
    try:
        authorization = request.headers.get("authorization")
    except Exception:
        pass
    if generation_id is None or not authorization:
        return
    parts = urlsplit(url)
    base_url = f"{parts.scheme}://{parts.netloc}"
    requested = extract_requested_routing(request_body)
    from_response = record_from_response(generation_id, response_body, requested)
    deadline = time.monotonic() + LOOKUP_BUDGET_SECONDS
    key: SessionKey = (context.workflow_id or "", context.run_id or "")

    def run() -> None:
        try:
            _attest(
                runtime, context, parent_span, generation_id, authorization, base_url, requested,
                from_response, deadline,
            )
        except Exception:
            logger.warning("routing provenance failed for %s", generation_id, exc_info=True)

    _submit(key, run)


def _submit(key: SessionKey, fn: Callable[[], None]) -> None:
    future: Future[None] = Future()
    with _pending_lock:
        _pending.setdefault(key, set()).add(future)
        # Daemon workers: a process that exits without closing its session is
        # never held open by a lookup. Sessions that do close are covered by
        # the drain in client.py, and runtime.close() drains the rest.
        if len(_workers) < _MAX_WORKERS:
            worker = threading.Thread(target=_worker, name="openbox-provenance", daemon=True)
            _workers.append(worker)
            worker.start()

    def task() -> None:
        try:
            fn()
        finally:
            # Before the future resolves, so a drain that returns never sees a
            # finished lookup as still pending.
            with _pending_lock:
                futures = _pending.get(key)
                if futures is not None:
                    futures.discard(future)
                    if not futures:
                        del _pending[key]

    _work.put((future, task))


def _worker() -> None:
    while True:
        future, fn = _work.get()
        if not future.set_running_or_notify_cancel():
            continue
        try:
            fn()
        except BaseException as exc:  # pragma: no cover - fn already catches
            future.set_exception(exc)
        else:
            future.set_result(None)


def drain_openrouter_provenance(
    workflow_id: str | None = None,
    run_id: str | None = None,
    timeout: float = DRAIN_TIMEOUT_SECONDS,
) -> None:
    """Wait (bounded) for outstanding provenance lookups.

    With ``workflow_id``/``run_id``, only that session's lookups — what a
    session's closing event needs. Without them, every pending lookup, e.g.
    before shutdown.
    """
    with _pending_lock:
        if workflow_id is None and run_id is None:
            futures = [f for group in _pending.values() for f in group]
        else:
            futures = list(_pending.get((workflow_id or "", run_id or ""), ()))
    if futures:
        wait(futures, timeout=timeout)
