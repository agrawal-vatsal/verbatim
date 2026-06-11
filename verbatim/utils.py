import time
import functools
from typing import Any, Callable, Dict, List, TypeVar, cast
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda

from verbatim.reranker import RerankerResult

F = TypeVar('F', bound=Callable[..., Any])


class PipelineState(TypedDict, total=False):
    user_input: str                        # set at pipeline entry, always present
    extracted: Dict[str, Any]             # added by _extract
    candidates: List[Dict[str, Any]]      # added by _retrieve
    rerank_result: RerankerResult         # added by _rerank
    answer: str                           # added by _synthesize


class PipelineAbort(Exception):
    """Raised to short-circuit a pipeline with a user-facing message."""


class runnable_step:
    """Descriptor that makes a method directly usable in a LangChain | chain.

    Accessing self._extract returns a RunnableLambda wrapping the bound method,
    so the pipeline can be written as:
        self._extract | self._retrieve | self._rerank | ...
    """

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        return RunnableLambda(functools.partial(self.func, obj))


def track_latency(func: F) -> F:
    """Decorator to record method execution time in milliseconds."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)

        # Ensure latencies dict exists on the instance
        if not hasattr(self, 'latencies'):
            self.latencies = {}

        self.latencies[func.__name__] = int((time.time() - start_time) * 1000)
        return result

    return cast(F, wrapper)
