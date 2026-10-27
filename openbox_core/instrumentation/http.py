"""HTTP wrappers — requests + httpx/httpx2 (sync/async).

Each wrapper creates or receives an OTel span and invokes the hook runtime
BEFORE the real request is sent; raising via the adapter prevents the request.
The wrapper itself never interprets verdicts.

Self-instrumentation guard: URLs under any ignored prefix (always including
the OpenBox ``api_url``) are skipped so evaluate calls never govern
themselves (no recursion).
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import logging
import time
from typing import Any

from ..context import activity_scope
from ..contracts.otel_spans import HookType
from ..hooks.events import resolve_context
from .http_stream import capture_async_response_stream, capture_response_stream
from .openrouter_provenance import schedule_openrouter_provenance
from .shared import get_hook_runtime

logger = logging.getLogger(__name__)

__all__ = [
    "set_ignored_url_prefixes",
    "suppress_http_instrumentation",
    "should_ignore_url",
    "sanitize_headers",
    "install_requests",
    "uninstall_requests",
    "install_httpx",
    "uninstall_httpx",
    "install_httpx_body_capture",
    "uninstall_httpx_body_capture",
    "install_httpx2_body_capture",
    "uninstall_httpx2_body_capture",
    "install_urllib3",
    "uninstall_urllib3",
    "install_urllib",
    "uninstall_urllib",
]

# Content-type markers safe to capture as a text body.
_TEXT_CONTENT_MARKERS = ("json", "text", "xml")


def _is_text_content_type(content_type: str | None) -> bool:
    """True when the content type indicates text (safe to read as a body)."""
    if not content_type:
        return True  # assume text when unspecified
    return any(marker in content_type.lower() for marker in _TEXT_CONTENT_MARKERS)


_ignored_url_prefixes: set[str] = set()
_http_instrumentation_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_http_instrumentation_suppressed", default=False
)

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_url_prefix(url: str) -> str:
    """Canonical prefix form: lowercase scheme/host, explicit default port,
    no trailing slash. httpx normalizes request URLs (lowercased host,
    default port stripped) — comparing RAW config strings against that would
    miss e.g. ``https://Core.example`` vs ``https://core.example`` and let
    the evaluate call govern itself (unbounded recursion)."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return url.rstrip("/")
    if not parts.scheme or not parts.hostname:
        return url.rstrip("/")
    scheme = parts.scheme.lower()
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(scheme)
    netloc = parts.hostname.lower() + (f":{port}" if port is not None else "")
    return f"{scheme}://{netloc}{parts.path}".rstrip("/")


# span_id -> perf_counter at request start (duration for the response hook).
_HOOK_TIMINGS_MAX = 4096
_hook_timings: dict[int, float] = {}


def set_ignored_url_prefixes(prefixes: set[str]) -> None:
    global _ignored_url_prefixes
    _ignored_url_prefixes = {_normalize_url_prefix(p) for p in prefixes if p}


@contextlib.contextmanager
def suppress_http_instrumentation():
    """Exclude a scoped SDK-owned HTTP exchange from governance hooks."""

    token = _http_instrumentation_suppressed.set(True)
    try:
        yield
    finally:
        _http_instrumentation_suppressed.reset(token)


# Headers whose values are credentials/secrets — never ship them into
# governance payloads (they land in Core logs verbatim otherwise).
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "x-amz-security-token",
    }
)


def sanitize_headers(headers: Any) -> dict | None:
    """Copy headers with credential values redacted; bytes decoded."""
    if not headers:
        return None
    sanitized = {}
    for key, value in dict(headers).items():
        if isinstance(key, bytes):
            key = key.decode("latin-1", errors="ignore")
        if isinstance(value, bytes):
            value = value.decode("latin-1", errors="ignore")
        sanitized[key] = "[REDACTED]" if str(key).lower() in _SENSITIVE_HEADERS else value
    return sanitized


def should_ignore_url(url: str | None) -> bool:
    if _http_instrumentation_suppressed.get():
        return True
    if not url:
        return True
    normalized = _normalize_url_prefix(url)
    return any(normalized.startswith(prefix) for prefix in _ignored_url_prefixes)


def _record_timing(span: Any) -> None:
    try:
        span_id = span.get_span_context().span_id
    except Exception:
        return
    if len(_hook_timings) >= _HOOK_TIMINGS_MAX:
        _hook_timings.clear()
    _hook_timings[span_id] = time.perf_counter()


def _pop_duration_ns(span: Any) -> int | None:
    try:
        span_id = span.get_span_context().span_id
    except Exception:
        return None
    started = _hook_timings.pop(span_id, None)
    return int((time.perf_counter() - started) * 1e9) if started is not None else None


# ── requests ─────────────────────────────────────────────────────────────────


def _requests_request_hook(span: Any, request: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    url = str(getattr(request, "url", "") or "")
    if should_ignore_url(url):
        return
    body = None
    try:
        raw = getattr(request, "body", None)
        if raw:
            body = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    headers = sanitize_headers(getattr(request, "headers", None))
    _record_timing(span)
    runtime.preflight(
        span,
        hook_type=HookType.HTTP_REQUEST,
        identifier=url,
        fields={
            "http_method": getattr(request, "method", None) or "UNKNOWN",
            "http_url": url,
            "request_body": body,
            "request_headers": headers,
        },
    )


def _requests_body(request: Any) -> str | None:
    """Best-effort request body from a requests PreparedRequest."""
    try:
        raw = getattr(request, "body", None)
        if not raw:
            return None
        return (
            raw.decode("utf-8", errors="ignore")
            if isinstance(raw, (bytes, bytearray))
            else str(raw)
        )
    except Exception:
        return None


def _requests_response_hook(span: Any, request: Any, response: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    url = str(getattr(request, "url", "") or "")
    if should_ignore_url(url):
        return
    status_code = getattr(response, "status_code", None)
    body = None
    try:
        content_type = response.headers.get("content-type", "") if response.headers else ""
        if _is_text_content_type(content_type):
            body = response.text
    except Exception:
        pass
    # Completed retains the request body/headers alongside the response so the
    # stage carries the full exchange.
    runtime.completed(
        span,
        hook_type=HookType.HTTP_REQUEST,
        fields={
            "http_method": getattr(request, "method", None) or "UNKNOWN",
            "http_url": url,
            "http_status_code": status_code,
            "request_body": _requests_body(request),
            "request_headers": sanitize_headers(getattr(request, "headers", None)),
            "response_body": body,
            "response_headers": sanitize_headers(getattr(response, "headers", None)),
            "duration_ns": _pop_duration_ns(span),
            "error": f"HTTP {status_code}" if status_code and status_code >= 400 else None,
        },
    )


def install_requests() -> bool:
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except ImportError:
        logger.info("requests instrumentation not available (install extra [http]) — deferred")
        return False
    RequestsInstrumentor().instrument(
        request_hook=_requests_request_hook, response_hook=_requests_response_hook
    )
    return True


def uninstall_requests() -> None:
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().uninstrument()
    except Exception:
        logger.debug("requests uninstrument skipped", exc_info=True)


# ── httpx (sync + async) ─────────────────────────────────────────────────────
#
# httpx bodies are NOT available in the OTel hooks: the request hook sees an
# unread stream and the response hook sees a ResponseInfo whose stream cannot be
# consumed safely. The send patch owns both stages so prebuilt/custom clients
# still get deterministic started + completed governance. The OTel request hook
# remains as a compatibility fallback for environments where the send patch is
# not installed.

# Task/thread-local stash: the request hook publishes the httpx CLIENT span here
# so the send patch (running after that span has ended) attaches the completed
# event to the SAME span identity.
_httpx_span_var: contextvars.ContextVar = contextvars.ContextVar("_httpx_span", default=None)
_httpx_patch_span_var: contextvars.ContextVar = contextvars.ContextVar(
    "_httpx_patch_span", default=None
)


def _httpx_url(request_info: Any) -> str:
    return str(getattr(request_info, "url", "") or "")


def _decode_method(method: Any) -> str:
    """httpx methods arrive as bytes (b"POST"); plain str() would mangle them."""
    if isinstance(method, bytes):
        return method.decode("ascii", errors="ignore") or "UNKNOWN"
    return str(method) if method else "UNKNOWN"


def _httpx_request_body(request_info: Any) -> str | None:
    """Best-effort request body from an OTel httpx RequestInfo.

    Reads ONLY already-buffered bytes (stream ``_stream``/``body``/``_body`` or
    a bytes ``_content``) — never triggers a property that could consume a live
    stream.
    """
    try:
        raw: Any = None
        stream = getattr(request_info, "stream", None)
        if stream is not None:
            for attr in ("_stream", "body", "_body"):
                candidate = getattr(stream, attr, None)
                if isinstance(candidate, (bytes, bytearray)):
                    raw = candidate
                    break
            if raw is None and isinstance(stream, (bytes, bytearray)):
                raw = stream
        if raw is None:
            content = getattr(request_info, "_content", None)
            if isinstance(content, (bytes, bytearray)):
                raw = content
        if raw is None:
            return None
        return bytes(raw).decode("utf-8", errors="ignore")
    except Exception:
        return None


def _httpx_started_fields(request_info: Any) -> dict:
    return {
        "http_method": _decode_method(getattr(request_info, "method", b"")),
        "http_url": _httpx_url(request_info),
        "request_headers": sanitize_headers(getattr(request_info, "headers", None)),
        "request_body": _httpx_request_body(request_info),
    }


def _httpx_request_hook(span: Any, request_info: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    if _httpx_patch_span_var.get(None) is not None:
        return
    url = _httpx_url(request_info)
    if should_ignore_url(url):
        return
    # Stash the span for the send patch's completed stage, then preflight.
    _httpx_span_var.set(span)
    runtime.preflight(
        span,
        hook_type=HookType.HTTP_REQUEST,
        identifier=url,
        fields=_httpx_started_fields(request_info),
    )


async def _httpx_async_request_hook(span: Any, request_info: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    if _httpx_patch_span_var.get(None) is not None:
        return
    url = _httpx_url(request_info)
    if should_ignore_url(url):
        return
    _httpx_span_var.set(span)
    await runtime.apreflight(
        span,
        hook_type=HookType.HTTP_REQUEST,
        identifier=url,
        fields=_httpx_started_fields(request_info),
    )


def install_httpx() -> bool:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        logger.info("httpx instrumentation not available (install extra [http]) — deferred")
        return False
    # Only request hooks: completed is emitted by the send patch (see below).
    HTTPXClientInstrumentor().instrument(
        request_hook=_httpx_request_hook,
        async_request_hook=_httpx_async_request_hook,
    )
    return True


def uninstall_httpx() -> None:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().uninstrument()
    except Exception:
        logger.debug("httpx uninstrument skipped", exc_info=True)


# ── httpx/httpx2 body capture (Client.send / AsyncClient.send patch) ──────────
#
# Separate from OTel instrumentation: OTel hooks receive streams that cannot be
# consumed. Patching send lets us read the request body (buffered before send)
# and the response body safely. Unread streams are observed as the caller reads
# them; completed telemetry waits for stream consumption or explicit close.

# Each library has distinct client classes. Keep its originals independent,
# including in the wrapper closures, so installing/restoring one cannot affect
# requests through the other.
_original_httpx_sends: dict[str, tuple[Any, Any, Any]] = {}


def _pop_httpx_span() -> Any:
    """Retrieve + clear the span stashed by the request hook (fallback: current
    span, so completed still resolves context if the stash missed)."""
    span = _httpx_span_var.get(None)
    _httpx_span_var.set(None)
    if span is None:
        from opentelemetry import trace

        span = trace.get_current_span()
    return span


def _capture_httpx_request(request: Any) -> tuple[str | None, dict | None]:
    """(request_body, request_headers) — reads only buffered content."""
    body = None
    headers = None
    try:
        raw = getattr(request, "_content", None)
        if isinstance(raw, (bytes, bytearray)):
            body = bytes(raw).decode("utf-8", errors="ignore")
        headers = sanitize_headers(getattr(request, "headers", None))
    except Exception:
        pass
    return body, headers


def _start_httpx_span(request: Any) -> Any:
    from opentelemetry import trace

    method = _decode_method(getattr(request, "method", None))
    url = str(getattr(request, "url", "") or "")
    span = trace.get_tracer("openbox-core.httpx").start_span(
        f"HTTP {method}", kind=trace.SpanKind.CLIENT
    )
    with contextlib.suppress(Exception):
        span.set_attribute("http.request.method", method)
        span.set_attribute("url.full", url)
    return span


def _capture_httpx_response(response: Any) -> tuple[str | None, dict | None]:
    """(response_body, response_headers) — never consumes a stream.

    ``response.text`` raises ``ResponseNotRead`` for an unread streaming
    response; that is caught, leaving the body None (safe, no consumption).
    """
    body = None
    headers = None
    try:
        headers = sanitize_headers(getattr(response, "headers", None))
        content_type = ""
        try:
            content_type = response.headers.get("content-type", "") if response.headers else ""
        except Exception:
            content_type = ""
        if _is_text_content_type(content_type):
            try:
                body = response.text
            except Exception:
                body = None
    except Exception:
        pass
    return body, headers


def _httpx_completed_fields(
    request: Any,
    response: Any,
    duration_ns: int | None,
    request_body: str | None,
    request_headers: dict | None,
    response_body: str | None,
    response_headers: dict | None,
) -> dict:
    status_code = getattr(response, "status_code", None)
    return {
        "http_method": _decode_method(getattr(request, "method", None)),
        "http_url": str(getattr(request, "url", "") or ""),
        "http_status_code": status_code,
        "request_body": request_body,
        "request_headers": request_headers,
        "response_body": response_body,
        "response_headers": response_headers,
        "duration_ns": duration_ns,
        "error": f"HTTP {status_code}"
        if isinstance(status_code, int) and status_code >= 400
        else None,
    }


def _stream_completed_fields(
    request: Any, response: Any, span: Any, start: float,
    request_body: str | None, request_headers: dict | None,
    body: str | None, error: str | None, partial: bool,
) -> dict:
    with contextlib.suppress(Exception):
        span.set_attribute("openbox.http.response_body.partial", partial)
    fields = _httpx_completed_fields(
        request, response, int((time.perf_counter() - start) * 1e9),
        request_body, request_headers, body, sanitize_headers(response.headers),
    )
    if error is not None:
        fields["error"] = error
    return fields


def install_httpx_body_capture() -> bool:
    """Capture both stages for httpx; install AFTER ``install_httpx``."""
    return _install_httpx_body_capture("httpx")


def install_httpx2_body_capture() -> bool:
    """Capture both stages for httpx2 when installed, using the same wire shape.

    The send wrapper creates its own OTel span, so this does not require a
    newer OTel HTTPX instrumentor with native httpx2 support.
    """
    return _install_httpx_body_capture("httpx2")


def _install_httpx_body_capture(module_name: str) -> bool:
    if module_name in _original_httpx_sends:
        return True
    try:
        library = importlib.import_module(module_name)
    except ImportError:
        logger.info("%s not available for body capture — deferred", module_name)
        return False

    original_send = library.Client.send
    original_async_send = library.AsyncClient.send

    def _patched_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        runtime = get_hook_runtime()
        url = str(getattr(request, "url", "") or "")
        if runtime is None or should_ignore_url(url):
            return original_send(self, request, *args, **kwargs)
        request_body, request_headers = _capture_httpx_request(request)
        span = _start_httpx_span(request)
        context = resolve_context(runtime._store, span)
        patch_token = _httpx_patch_span_var.set(span)
        start = time.perf_counter()
        deferred = False
        try:
            runtime.preflight(
                span,
                hook_type=HookType.HTTP_REQUEST,
                identifier=url,
                fields={
                    "http_method": _decode_method(getattr(request, "method", None)),
                    "http_url": url,
                    "request_headers": request_headers,
                    "request_body": request_body,
                },
            )
            response = original_send(self, request, *args, **kwargs)
            if not response.is_stream_consumed:
                # An unbound request must not acquire another activity's
                # context just because its body is consumed later.
                if context is None or not context.activity_id or not context.activity_type:
                    return response

                def complete(body: str | None, error: str | None, partial: bool) -> None:
                    try:
                        fields = _stream_completed_fields(
                            request, response, span, start, request_body, request_headers,
                            body, error, partial,
                        )
                        with activity_scope(context, store=runtime._store):
                            runtime.completed(span, hook_type=HookType.HTTP_REQUEST, fields=fields)
                        schedule_openrouter_provenance(
                            runtime, context, span, request, request_body, body
                        )
                    finally:
                        span.end()

                capture_response_stream(
                    response,
                    max_chars=runtime._runtime.config.privacy.max_body_size,
                    capture_text=_is_text_content_type(response.headers.get("content-type")),
                    complete=complete,
                )
                deferred = True
                return response
            duration_ns = int((time.perf_counter() - start) * 1e9)
            response_body, response_headers = _capture_httpx_response(response)
            runtime.completed(
                span,
                hook_type=HookType.HTTP_REQUEST,
                fields=_httpx_completed_fields(
                    request,
                    response,
                    duration_ns,
                    request_body,
                    request_headers,
                    response_body,
                    response_headers,
                ),
            )
            schedule_openrouter_provenance(
                runtime, context, span, request, request_body, response_body
            )
            return response
        finally:
            _httpx_patch_span_var.reset(patch_token)
            if not deferred:
                with contextlib.suppress(Exception):
                    span.end()

    async def _patched_async_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        runtime = get_hook_runtime()
        url = str(getattr(request, "url", "") or "")
        if runtime is None or should_ignore_url(url):
            return await original_async_send(self, request, *args, **kwargs)
        request_body, request_headers = _capture_httpx_request(request)
        span = _start_httpx_span(request)
        context = resolve_context(runtime._store, span)
        patch_token = _httpx_patch_span_var.set(span)
        start = time.perf_counter()
        deferred = False
        try:
            await runtime.apreflight(
                span,
                hook_type=HookType.HTTP_REQUEST,
                identifier=url,
                fields={
                    "http_method": _decode_method(getattr(request, "method", None)),
                    "http_url": url,
                    "request_headers": request_headers,
                    "request_body": request_body,
                },
            )
            response = await original_async_send(self, request, *args, **kwargs)
            if not response.is_stream_consumed:
                if context is None or not context.activity_id or not context.activity_type:
                    return response

                async def complete(body: str | None, error: str | None, partial: bool) -> None:
                    try:
                        fields = _stream_completed_fields(
                            request, response, span, start, request_body, request_headers,
                            body, error, partial,
                        )
                        with activity_scope(context, store=runtime._store):
                            await runtime.acompleted(
                                span, hook_type=HookType.HTTP_REQUEST, fields=fields
                            )
                        schedule_openrouter_provenance(
                            runtime, context, span, request, request_body, body
                        )
                    finally:
                        span.end()

                capture_async_response_stream(
                    response,
                    max_chars=runtime._runtime.config.privacy.max_body_size,
                    capture_text=_is_text_content_type(response.headers.get("content-type")),
                    complete=complete,
                )
                deferred = True
                return response
            duration_ns = int((time.perf_counter() - start) * 1e9)
            response_body, response_headers = _capture_httpx_response(response)
            await runtime.acompleted(
                span,
                hook_type=HookType.HTTP_REQUEST,
                fields=_httpx_completed_fields(
                    request,
                    response,
                    duration_ns,
                    request_body,
                    request_headers,
                    response_body,
                    response_headers,
                ),
            )
            schedule_openrouter_provenance(
                runtime, context, span, request, request_body, response_body
            )
            return response
        finally:
            _httpx_patch_span_var.reset(patch_token)
            if not deferred:
                with contextlib.suppress(Exception):
                    span.end()

    library.Client.send = _patched_send
    library.AsyncClient.send = _patched_async_send
    _original_httpx_sends[module_name] = (library, original_send, original_async_send)
    return True


def uninstall_httpx_body_capture() -> None:
    """Restore httpx send (idempotent). Call BEFORE ``uninstall_httpx`` so the
    original chain is unwound in reverse install order."""
    _uninstall_httpx_body_capture("httpx")


def uninstall_httpx2_body_capture() -> None:
    """Restore httpx2 send independently of httpx (idempotent)."""
    _uninstall_httpx_body_capture("httpx2")


def _uninstall_httpx_body_capture(module_name: str) -> None:
    originals = _original_httpx_sends.pop(module_name, None)
    if originals is None:
        return
    library, original_send, original_async_send = originals
    try:
        library.Client.send = original_send
        library.AsyncClient.send = original_async_send
    except Exception:
        logger.debug("%s body-capture restore skipped", module_name, exc_info=True)


# ── urllib3 (OTel URLLib3Instrumentor hooks) ─────────────────────────────────
#
# ``requests`` runs on urllib3; OTel suppresses the nested urllib3 span while a
# RequestsInstrumentor call is active, so a requests call does NOT double-fire
# these hooks. urllib3-native calls reach here.

_urllib3_installed = False


def _urllib3_url(pool: Any, request_info: Any) -> str:
    # The OTel urllib3 instrumentor resolves the FULL url on RequestInfo.url;
    # use it as-is. Only reconstruct if we somehow got a bare path.
    url = getattr(request_info, "url", None) or getattr(request_info, "request_url", None)
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    scheme = getattr(pool, "scheme", "http")
    host = getattr(pool, "host", "unknown")
    port = getattr(pool, "port", None)
    path = url or "/"
    if port and port not in (80, 443):
        return f"{scheme}://{host}:{port}{path}"
    return f"{scheme}://{host}{path}"


def _urllib3_body(request_info: Any) -> str | None:
    try:
        raw = getattr(request_info, "body", None)
        if not raw:
            return None
        return (
            raw.decode("utf-8", errors="ignore")
            if isinstance(raw, (bytes, bytearray))
            else str(raw)
        )
    except Exception:
        return None


def _urllib3_request_hook(span: Any, pool: Any, request_info: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    url = _urllib3_url(pool, request_info)
    if should_ignore_url(url):
        return
    _record_timing(span)
    runtime.preflight(
        span,
        hook_type=HookType.HTTP_REQUEST,
        identifier=url,
        fields={
            "http_method": getattr(request_info, "method", None) or "UNKNOWN",
            "http_url": url,
            "request_body": _urllib3_body(request_info),
            "request_headers": sanitize_headers(getattr(request_info, "headers", None)),
        },
    )


def _urllib3_response_hook(span: Any, pool: Any, response: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    # The response hook has no request_info; the URL degrades to the host root.
    # Read only ALREADY-BUFFERED content (``_body``, set when preload_content=True)
    # so a live streaming response is never consumed.
    scheme = getattr(pool, "scheme", "http")
    host = getattr(pool, "host", "unknown")
    port = getattr(pool, "port", None)
    url = f"{scheme}://{host}:{port}/" if port and port not in (80, 443) else f"{scheme}://{host}/"
    if should_ignore_url(url):
        return
    status_code = getattr(response, "status", None)
    body = None
    try:
        content_type = (
            response.headers.get("content-type", "") if getattr(response, "headers", None) else ""
        )
        if _is_text_content_type(content_type):
            raw = getattr(response, "_body", None)
            if isinstance(raw, (bytes, bytearray)):
                body = bytes(raw).decode("utf-8", errors="ignore")
            elif isinstance(raw, str):
                body = raw
    except Exception:
        pass
    runtime.completed(
        span,
        hook_type=HookType.HTTP_REQUEST,
        fields={
            "http_method": "UNKNOWN",
            "http_url": url,
            "http_status_code": status_code,
            "response_body": body,
            "response_headers": sanitize_headers(getattr(response, "headers", None)),
            "duration_ns": _pop_duration_ns(span),
            "error": f"HTTP {status_code}"
            if isinstance(status_code, int) and status_code >= 400
            else None,
        },
    )


def install_urllib3() -> bool:
    global _urllib3_installed
    if _urllib3_installed:
        return True
    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
    except ImportError:
        logger.info("urllib3 instrumentation not available (install extra [http]) — deferred")
        return False
    URLLib3Instrumentor().instrument(
        request_hook=_urllib3_request_hook, response_hook=_urllib3_response_hook
    )
    _urllib3_installed = True
    return True


def uninstall_urllib3() -> None:
    global _urllib3_installed
    if not _urllib3_installed:
        return
    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        URLLib3Instrumentor().uninstrument()
    except Exception:
        logger.debug("urllib3 uninstrument skipped", exc_info=True)
    _urllib3_installed = False


# ── urllib (stdlib, OTel URLLibInstrumentor hooks) ───────────────────────────
#
# Response body is NOT captured: urllib's HTTPResponse.read() consumes the
# socket stream and would break the caller. Completed carries status only.

_urllib_installed = False


def _urllib_url(request: Any) -> str:
    url = getattr(request, "full_url", None)
    if url:
        return str(url)
    getter = getattr(request, "get_full_url", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return ""
    return ""


def _urllib_method(request: Any) -> str:
    getter = getattr(request, "get_method", None)
    if callable(getter):
        try:
            return str(getter()) or "UNKNOWN"
        except Exception:
            pass
    return str(getattr(request, "method", None) or "UNKNOWN")


def _urllib_request_hook(span: Any, request: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    url = _urllib_url(request)
    if should_ignore_url(url):
        return
    _record_timing(span)
    body = None
    try:
        raw = getattr(request, "data", None)
        if raw:
            body = (
                raw.decode("utf-8", errors="ignore")
                if isinstance(raw, (bytes, bytearray))
                else str(raw)
            )
    except Exception:
        pass
    runtime.preflight(
        span,
        hook_type=HookType.HTTP_REQUEST,
        identifier=url,
        fields={
            "http_method": _urllib_method(request),
            "http_url": url,
            "request_body": body,
            "request_headers": sanitize_headers(getattr(request, "headers", None)),
        },
    )


def _urllib_response_hook(span: Any, request: Any, response: Any) -> None:
    runtime = get_hook_runtime()
    if runtime is None:
        return
    url = _urllib_url(request)
    if should_ignore_url(url):
        return
    status_code = getattr(response, "status", None) or getattr(response, "code", None)
    runtime.completed(
        span,
        hook_type=HookType.HTTP_REQUEST,
        fields={
            "http_method": _urllib_method(request),
            "http_url": url,
            "http_status_code": status_code,
            "duration_ns": _pop_duration_ns(span),
            "error": f"HTTP {status_code}"
            if isinstance(status_code, int) and status_code >= 400
            else None,
        },
    )


def install_urllib() -> bool:
    global _urllib_installed
    if _urllib_installed:
        return True
    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor
    except ImportError:
        logger.info("urllib instrumentation not available (install extra [http]) — deferred")
        return False
    URLLibInstrumentor().instrument(
        request_hook=_urllib_request_hook, response_hook=_urllib_response_hook
    )
    _urllib_installed = True
    return True


def uninstall_urllib() -> None:
    global _urllib_installed
    if not _urllib_installed:
        return
    try:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        URLLibInstrumentor().uninstrument()
    except Exception:
        logger.debug("urllib uninstrument skipped", exc_info=True)
    _urllib_installed = False
