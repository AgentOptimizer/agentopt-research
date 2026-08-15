"""Minimal proxy stubs so model_selection imports resolve without the full MITM stack.

This research checkout keeps selector algorithms + benchmark data only.
Live LLM tracking / caching lives in the main ``agentopt`` package.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class CallRecord:
    """Placeholder call record (unused in offline research)."""

    model: str = ""
    data_id: str = ""
    combo_id: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SessionInfo:
    """Placeholder session returned by :meth:`LLMTracker.track`."""

    data_id: str = ""
    combo_id: str = ""
    records: List[CallRecord] = field(default_factory=list)


class LLMTracker:
    """No-op tracker stub for offline / algorithm-only research.

    Provides the methods :class:`~agentopt.model_selection.base.BaseModelSelector`
    expects. Does not intercept LLM calls.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._records: List[CallRecord] = []
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    @contextmanager
    def track(
        self,
        data_id: str = "",
        combo_id: str = "",
        **kwargs: Any,
    ) -> Iterator[SessionInfo]:
        yield SessionInfo(data_id=data_id, combo_id=combo_id)

    def get_records(self, **kwargs: Any) -> List[CallRecord]:
        return list(self._records)

    def get_usage(
        self,
        data_id: Optional[str] = None,
        combo_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"input_tokens": {}, "output_tokens": {}}

    def get_cached_latency(self, data_id: str = "", **kwargs: Any) -> float:
        return 0.0


__all__ = ["LLMTracker", "CallRecord", "SessionInfo"]
