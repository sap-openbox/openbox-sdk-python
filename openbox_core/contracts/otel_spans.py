"""Span contracts — Stage, HookType, and flat OpenBox hook spans.

OTel spans are the INTERNAL source of truth (there are no fixed
HttpSpan/DbSpan/... dataclasses), but ``from_otel_span`` returns the flat Core
``SpanData`` shape immediately. There is no SDK-visible
``{"otel": ..., "openbox": ...}`` span envelope.

Pure module: span access is DUCK-TYPED (plain getattr) so importing this never
pulls in opentelemetry — the import-safety harness holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from ..otel.trace_context import format_span_id, format_trace_id

__all__ = [
    "Stage",
    "HookType",
    "serialize_readable_span",
    "from_otel_span",
    "OpenBoxSpanAdapter",
]


class Stage(str, Enum):
    """Hook evaluation stage."""

    STARTED = "started"
    COMPLETED = "completed"


class HookType(str, Enum):
    """Operation category of a hook span (Core root field ``hook_type``)."""

    HTTP_REQUEST = "http_request"
    DB_QUERY = "db_query"
    FILE_OPERATION = "file_operation"
    FUNCTION_CALL = "function_call"
    LLM_CALL = "llm_call"  # reserved; disabled until provider hooks are implemented
    # OpenRouter's record of who served a completion, read back after the call
    # (instrumentation/openrouter_provenance.py).
    LLM_PROVENANCE = "llm_provenance"


def _span_context_of(span: Any) -> Any:
    if hasattr(span, "get_span_context"):
        try:
            return span.get_span_context()
        except Exception:
            return None
    return getattr(span, "context", None)


def _enum_name(value: Any) -> str | None:
    """'SpanKind.CLIENT' -> 'CLIENT'; plain strings pass through."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.split(".")[-1] if text else None


def _attributes_of(obj: Any) -> dict[str, Any]:
    attributes = getattr(obj, "attributes", None)
    if attributes is None:
        return {}
    try:
        return dict(attributes)
    except Exception:
        return {}


# Best-effort semantic attribute mapping: wire field -> OTel keys (new, legacy).
_SEMANTIC_ATTR_MAP: dict[str, tuple[str, ...]] = {
    "http_url": ("url.full", "http.url"),
    "http_method": ("http.request.method", "http.method"),
    "http_status_code": ("http.response.status_code", "http.status_code"),
    "db_system": ("db.system.name", "db.system"),
    "db_name": ("db.namespace", "db.name"),
    "db_operation": ("db.operation.name", "db.operation"),
    "db_statement": ("db.query.text", "db.statement"),
    "server_address": ("server.address", "net.peer.name"),
    "server_port": ("server.port", "net.peer.port"),
    "file_path": ("file.path",),
    "file_mode": ("file.mode",),
    "file_operation": ("file.operation",),
    "function": ("code.function.name", "code.function"),
    "module": ("code.namespace", "code.module"),
}

_DEFAULT_KIND_BY_HOOK: dict[str, str] = {
    "http_request": "CLIENT",
    "db_query": "CLIENT",
    "file_operation": "INTERNAL",
    "function_call": "INTERNAL",
    "llm_call": "CLIENT",
    "llm_provenance": "CLIENT",
}

# Family-specific root fields that must exist, null-valued when unavailable, so
# hook spans have the same flat key contract in memory and on the wire.
_ROOT_FIELDS_BY_HOOK_TYPE: dict[str, tuple[str, ...]] = {
    "http_request": (
        "http_method",
        "http_url",
        "http_status_code",
        "request_headers",
        "response_headers",
        "request_body",
        "response_body",
    ),
    "db_query": (
        "db_system",
        "db_name",
        "db_operation",
        "db_statement",
        "server_address",
        "server_port",
        "rowcount",
    ),
    "file_operation": (
        "file_path",
        "file_mode",
        "file_operation",
        "bytes_read",
        "bytes_written",
    ),
    "function_call": ("function", "module", "args", "result"),
}


def _attr(attributes: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = attributes.get(key)
        if value is not None:
            return value
    return None


def serialize_readable_span(span: Any) -> dict[str, Any]:
    """Preserve the OTel surface the SDK exposes, via duck-typed access.

    Ids stay RAW INTEGERS here (internal representation; the wire projection
    formats hex strings). Missing pieces degrade to None/empty — a span is
    never rejected during serialization.
    """
    span_context = _span_context_of(span)
    context_part = None
    if span_context is not None:
        span_id = getattr(span_context, "span_id", None)
        trace_id = getattr(span_context, "trace_id", None)
        context_part = {
            "span_id": span_id if isinstance(span_id, int) else None,
            "trace_id": trace_id if isinstance(trace_id, int) else None,
        }

    parent = getattr(span, "parent", None)
    parent_part = None
    if parent is not None:
        parent_span_id = getattr(parent, "span_id", None)
        if isinstance(parent_span_id, int):
            parent_part = {"span_id": parent_span_id}

    status = getattr(span, "status", None)
    status_part = None
    if status is not None:
        status_part = {
            "code": _enum_name(getattr(status, "status_code", None)) or "UNSET",
            "description": getattr(status, "description", None),
        }

    events = []
    for event in getattr(span, "events", None) or ():
        events.append(
            {
                "name": getattr(event, "name", None),
                "timestamp": getattr(event, "timestamp", None),
                "attributes": _attributes_of(event),
            }
        )

    links = []
    for link in getattr(span, "links", None) or ():
        link_context = getattr(link, "context", None)
        links.append(
            {
                "context": {
                    "span_id": getattr(link_context, "span_id", None),
                    "trace_id": getattr(link_context, "trace_id", None),
                },
                "attributes": _attributes_of(link),
            }
        )

    resource = getattr(span, "resource", None)
    resource_part = _attributes_of(resource) if resource is not None else None

    scope = getattr(span, "instrumentation_scope", None)
    scope_part = None
    if scope is not None:
        scope_part = {
            "name": getattr(scope, "name", None),
            "version": getattr(scope, "version", None),
        }

    start_time = getattr(span, "start_time", None)
    end_time = getattr(span, "end_time", None)
    return {
        "context": context_part,
        "parent": parent_part,
        "name": getattr(span, "name", None),
        "kind": _enum_name(getattr(span, "kind", None)),
        "start_time": start_time if isinstance(start_time, int) else None,
        "end_time": end_time if isinstance(end_time, int) else None,
        "attributes": _attributes_of(span),
        "events": events,
        "links": links,
        "status": status_part,
        "resource": resource_part,
        "instrumentation_scope": scope_part,
    }


def from_otel_span(
    span: Any,
    *,
    stage: Stage | str,
    hook_type: HookType | str | None = None,
    activity_context: Any = None,
    fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one flat Core ``SpanData`` dict from an OTel span.

    Args:
        span: OTel span (Span/ReadableSpan/NonRecordingSpan — duck-typed).
        stage: started/completed.
        hook_type: Operation category (None for non-hook telemetry spans).
        activity_context: Accepted for backward-compatible call sites; context is
            stored on the surrounding event, not inside the span.
        fields: Wrapper-supplied Core root fields (request_body, rowcount,
            args, ...) merged at the span root. These carry data OTel attributes
            don't (bodies, results).
    """
    _ = activity_context
    stage_value = stage.value if isinstance(stage, Stage) else str(stage)
    hook_value = hook_type.value if isinstance(hook_type, HookType) else hook_type
    otel = serialize_readable_span(span)
    context = otel.get("context") or {}
    parent = otel.get("parent") or {}
    span_id_int = context.get("span_id")
    trace_id_int = context.get("trace_id")
    parent_id_int = parent.get("span_id")
    attributes = dict(otel.get("attributes") or {})

    start_time = otel.get("start_time")
    end_time = otel.get("end_time")
    if stage_value == Stage.STARTED.value:
        end_time = None
        duration_ns = None
    else:
        duration_ns = (
            end_time - start_time
            if isinstance(end_time, int) and isinstance(start_time, int)
            else None
        )

    wire: dict[str, Any] = {
        "span_id": format_span_id(span_id_int) if isinstance(span_id_int, int) else "0" * 16,
        "trace_id": format_trace_id(trace_id_int) if isinstance(trace_id_int, int) else "0" * 32,
        "parent_span_id": format_span_id(parent_id_int) if isinstance(parent_id_int, int) else None,
        "name": otel.get("name") or (hook_value or "span"),
        "kind": otel.get("kind") or _DEFAULT_KIND_BY_HOOK.get(hook_value or "", "INTERNAL"),
        "stage": stage_value,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ns": duration_ns,
        "attributes": attributes,
        "status": otel.get("status") or {"code": "UNSET", "description": None},
        "events": otel.get("events") or [],
        "error": None,
    }
    if hook_value:
        wire["hook_type"] = hook_value

    for wire_field, attr_keys in _SEMANTIC_ATTR_MAP.items():
        value = _attr(attributes, attr_keys)
        if value is not None:
            wire[wire_field] = value

    for field_name, value in dict(fields or {}).items():
        if value is not None:
            wire[field_name] = value

    for field_name in _ROOT_FIELDS_BY_HOOK_TYPE.get(hook_value or "", ()):
        wire.setdefault(field_name, None)

    if stage_value != Stage.STARTED.value and wire.get("end_time") is None:
        measured = wire.get("duration_ns")
        if isinstance(start_time, int) and isinstance(measured, int):
            wire["end_time"] = start_time + measured

    return wire


class OpenBoxSpanAdapter:
    """Adapter object form of :func:`from_otel_span` (protocol-compatible)."""

    @staticmethod
    def from_otel_span(
        span: Any,
        *,
        stage: Stage | str,
        hook_type: HookType | str | None = None,
        activity_context: Any = None,
        fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return from_otel_span(
            span,
            stage=stage,
            hook_type=hook_type,
            activity_context=activity_context,
            fields=fields,
        )
