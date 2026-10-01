"""Abstract base class for all agent and LLM benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from tokpulse.core.models import BenchmarkResult


class BaseAgentRunner(ABC):
    """Interface for agent and provider runners."""

    @abstractmethod
    async def run_prompt(
        self,
        prompt: str,
        model: str | None = None,
        ttft_timeout_s: float = 15.0,
        stall_timeout_s: float = 10.0,
        deadline_timeout_s: float = 60.0,
        on_chunk: Callable[[str], None] | None = None,
    ) -> BenchmarkResult:
        """Executes a benchmark prompt and collects high-precision timing & token metrics."""
        pass

    async def cleanup_session(self, session_id: str | None = None) -> None:  # noqa: B027
        """Privacy hook: deletes temporary chat sessions or scratch logs from the agent host."""
        return None
