"""Request ids: taken from X-Request-Id (sent by the backend) or generated, echoed back and added to every log line."""

import logging
import re
import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-request-id"
NO_REQUEST_ID = "-"
# Ids are copied into log lines and response headers, so only short, plain ids are trusted
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

# A ContextVar follows the request through awaits and into asyncio.to_thread workers
request_id_var: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)


def current_request_id() -> str:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record so the formatters can print it ("-" outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _incoming_request_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == REQUEST_ID_HEADER:
            candidate = value.decode("latin-1").strip()
            return candidate if VALID_REQUEST_ID.fullmatch(candidate) else None
    return None


class RequestIdMiddleware:
    """Plain ASGI middleware (not BaseHTTPMiddleware) so streaming responses pass through untouched."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        token = request_id_var.set(request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (REQUEST_ID_HEADER, request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)
