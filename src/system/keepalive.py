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


class KeepAlive:
    def __init__(self, port: int) -> None:
        self.url = f"http://localhost:{port}/"
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
            async with httpx.AsyncClient() as client:
                await client.get(self.url)
        except httpx.HTTPError as exc:
            logger.warning("Self-ping failed: %s", exc)
            return
        self.heartbeats.appendleft(time.strftime("%H:%M:%S"))
        logger.info("Self-ping OK at %s", self.heartbeats[0])
