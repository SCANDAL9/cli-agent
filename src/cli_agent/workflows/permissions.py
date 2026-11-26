"""Async human approval broker for workflows running alongside the REPL."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..security import RiskLevel


@dataclass(frozen=True)
class WorkflowPermissionRequest:
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk: RiskLevel


class WorkflowPermissionBroker:
    """Lets the main REPL collect a decision without competing for terminal input."""

    def __init__(
        self,
        on_request: Callable[[WorkflowPermissionRequest], None] | None = None,
    ) -> None:
        self.on_request = on_request
        self.pending: WorkflowPermissionRequest | None = None
        self._decision: asyncio.Future[bool] | None = None

    async def request(self, tool_name: str, arguments: dict[str, Any], risk: RiskLevel) -> bool:
        if self.pending is not None:
            raise RuntimeError("A workflow permission request is already pending.")
        loop = asyncio.get_running_loop()
        self._decision = loop.create_future()
        request = WorkflowPermissionRequest(
            request_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            arguments=dict(arguments),
            risk=risk,
        )
        self.pending = request
        try:
            if self.on_request is not None:
                self.on_request(request)
            return await self._decision
        finally:
            self.pending = None
            self._decision = None

    def resolve(self, allowed: bool) -> bool:
        decision = self._decision
        if decision is None or decision.done():
            return False
        decision.set_result(allowed)
        return True
