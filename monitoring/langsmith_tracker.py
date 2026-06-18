"""
RIS LangSmith Tracing Client Wrapper.
Configures environment bindings and hooks for LangSmith observability tracing.
Fails-safe: if the langsmith SDK is not installed or API keys are missing, degrades to local debug logs.
"""

import os
import logging
from typing import Callable, Any, Optional
from configs.config_manager import get_config

logger = logging.getLogger("langsmith_tracker")

# Try to import the LangSmith SDK
try:
    from langsmith import Client
    from langsmith.run_helpers import traceable as ls_traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    # No-op decorator if SDK is not present
    def ls_traceable(*args, **kwargs):
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

class LangSmithTracker:
    """
    Manages connections and metadata settings for sending trace runs to LangSmith.
    """
    def __init__(self):
        config = get_config()
        # Enable tracing only if config parameter is true and SDK is available
        self.enabled = config.langchain_tracing and LANGSMITH_AVAILABLE
        self.client = None

        if self.enabled:
            try:
                # Retrieve configuration keys
                self.api_key = config.langchain_api_key or os.getenv("LANGCHAIN_API_KEY")
                self.project = config.langchain_project or os.getenv("LANGCHAIN_PROJECT", "research-intelligence-system")
                self.endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
                
                if not self.api_key:
                    logger.warning("LANGCHAIN_API_KEY is not set. LangSmith tracing is disabled.")
                    self.enabled = False
                    return
                
                # Expose environments to LangChain/LangSmith automatically
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
                os.environ["LANGCHAIN_API_KEY"] = self.api_key
                os.environ["LANGCHAIN_PROJECT"] = self.project
                
                self.client = Client()
                logger.info(f"LangSmith Tracing configured successfully for project '{self.project}'.")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith trace engine: {e}")
                self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

# Decorator proxy to route tracing through LangSmith if active
def traceable(name: Optional[str] = None, run_type: str = "chain") -> Callable:
    """
    A unified decorator for tracing pipeline steps in LangSmith.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if LANGSMITH_AVAILABLE:
            # Wrap the function with LangSmith's native traceable decorator
            return ls_traceable(name=name, run_type=run_type)(func)
        else:
            return func
    return decorator
