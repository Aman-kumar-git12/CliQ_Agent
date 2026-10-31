"""Health check and the HTML status page at /."""

from functools import lru_cache
from pathlib import Path
from string import Template

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .keepalive import KeepAlive

router = APIRouter(tags=["system"])

TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
NO_HEARTBEATS = '<div style="opacity:0.3">Waiting for first heartbeat...</div>'


@lru_cache
def _dashboard() -> Template:
    return Template(TEMPLATE_PATH.read_text(encoding="utf-8"))


@router.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
def status_page(request: Request) -> str:
    keepalive: KeepAlive = request.app.state.keepalive
    hours, rest = divmod(keepalive.uptime_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    heartbeats = "".join(
        f'<div class="heartbeat-item">SUCCESS <span>{beat}</span></div>' for beat in keepalive.heartbeats
    )
    return _dashboard().substitute(uptime=f"{hours}h {minutes}m {seconds}s", heartbeats=heartbeats or NO_HEARTBEATS)
