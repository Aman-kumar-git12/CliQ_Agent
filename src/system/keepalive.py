"""Pings the service itself so free hosting tiers (Render) don't put it to sleep."""

import asyncio
import logging
import time
from collections import deque

import httpx

logger = logging.getLogger(__name__)

PING_INTERVAL_SECONDS = 12 * 60
STARTUP_DELAY_SECONDS = 10
MAX_HEARTBEATS = 10
PING_TIMEOUT_SECONDS = 10
# The lightest endpoint: no template, no database
PING_PATH = "/api/health"


class KeepAlive:
    def __init__(self, port: int, public_url: str = "") -> None:
        # Render only counts traffic that comes in through its proxy, so the public URL must be pinged;
        # localhost (used when no public URL is known, e.g. in development) keeps nothing awake there
        base = public_url.strip().rstrip("/") or f"http://localhost:{port}"
        self.url = f"{base}{PING_PATH}"
        self.started_at = time.time()
        # Times of the last successful pings, newest first, shown on the status page
        self.heartbeats: deque[str] = deque(maxlen=MAX_HEARTBEATS)

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.started_at)

    async def run(self) -> None:
        await asyncio.sleep(STARTUP_DELAY_SECONDS)  # let the server finish starting
        while True:
            await self.ping()
            await asyncio.sleep(PING_INTERVAL_SECONDS)

    async def ping(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=PING_TIMEOUT_SECONDS) as client:
                response = await client.get(self.url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Self-ping failed: %s", exc)
            return
        self.heartbeats.appendleft(time.strftime("%H:%M:%S"))
        logger.info("Self-ping OK at %s", self.heartbeats[0])
