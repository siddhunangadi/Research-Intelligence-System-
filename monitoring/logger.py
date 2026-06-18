"""
RIS Central Logger.
Configures baseline logging formatting standards, streaming to stdout, and file sinks.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Configures the root logger with stream and rotating file handlers.
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # Define common format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if logs directory is not writable
        print(f"Warning: Could not configure file logging: {e}", file=sys.stderr)

    logging.info(f"Logging initialized with level {log_level}. Logs saved to {log_file}")
    return root_logger

# Automatically run setup on import to configure the process logs
setup_logging()
