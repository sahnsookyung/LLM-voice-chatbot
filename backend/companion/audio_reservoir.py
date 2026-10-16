"""Bounded prebuffering prevents a slower-than-real-time voice chopping words.

The 8-bit voice may produce audio slower than the device consumes it. Begin a
clause once complete, or once the six-second reservoir has a safe lead. Delivery
still observes the separate one-second native acknowledgement window.
"""

import asyncio
from collections import deque


class AudioReservoir:
    # Leave room for four codec/IPC packets inside the six-second budget.
    limit = 6 * 48000 - 4 * 3840

    def __init__(self):
        self.items = deque()
        self.bytes = 0
        self.condition = asyncio.Condition()

    async def put(self, item):
        length = len(item[0]) if item else 0
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.bytes + length <= self.limit and len(self.items) < 256
            )
            self.items.append(item)
            self.bytes += length
            self.condition.notify_all()

    async def begin_clause(self):
        async with self.condition:
            await self.condition.wait_for(
                lambda: (
                    self.bytes >= self.limit - 3840
                    or any(item is None or item[1] is not None for item in self.items)
                )
            )

    async def get(self):
        async with self.condition:
            await self.condition.wait_for(lambda: bool(self.items))
            item = self.items.popleft()
            self.bytes -= len(item[0]) if item else 0
            self.condition.notify_all()
            return item
