"""
RIS Latency Tracker.
Provides a context manager and decorator to measure and log function runtimes in milliseconds.
"""

import time
import logging
import functools
from typing import Callable, Any, Optional

logger = logging.getLogger("latency_tracker")

class LatencyTracker:
    """
    Measures execution time of a code block or function in milliseconds.
    Can be used as a context manager or decorator.
    """
    def __init__(self, label: str):
        self.label = label
        self.start_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            logger.info(f"[{self.label}] execution time: {self.elapsed_ms:.2f} ms")

def profile_latency(label: Optional[str] = None) -> Callable:
    """
    Decorator to measure execution time of functions.
    
    Args:
        label: Custom label for the output. If None, uses the function name.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_label = label or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with LatencyTracker(func_label) as tracker:
                return func(*args, **kwargs)
        return wrapper
    return decorator
