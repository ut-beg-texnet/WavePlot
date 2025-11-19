"""Logging utility for WavePlot debug mode.

This module provides a lightweight logging setup that activates only when
debug mode is enabled. Logs are written to rotating files to prevent
unbounded growth.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(debug: bool, log_dir: Optional[str] = None) -> Optional[logging.Logger]:
    """Set up logging for WavePlot application.
    
    When debug is True, creates a rotating file handler that writes to
    a log file in the specified directory. The log file rotates when it
    reaches 10MB, keeping up to 3 backup files.
    
    Args:
        debug: If True, enable logging. If False, returns None.
        log_dir: Directory for log files. Defaults to 'logs' relative to this module.
        
    Returns:
        Logger instance if debug is True, None otherwise.
        
    Note:
        Logging failures are handled gracefully and won't crash the application.
    """
    if not debug:
        return None
    
    try:
        # Determine log directory
        if log_dir is None:
            # Use logs/ directory relative to this module
            module_dir = Path(__file__).parent
            log_dir = module_dir / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create logs directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        logger = logging.getLogger("waveplot")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create rotating file handler
        log_file = log_dir / "waveplot_debug.log"
        max_bytes = 10 * 1024 * 1024  # 10MB
        backup_count = 3
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        handler.setLevel(logging.DEBUG)
        
        # Set up formatter: timestamp, level, module, message
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        # Log initial message
        logger.info("=" * 60)
        logger.info("WavePlot debug logging started")
        logger.info(f"Log file: {log_file}")
        logger.info("=" * 60)
        
        return logger
        
    except Exception as e:
        # Gracefully handle logging setup failures
        # Don't crash the application if logging can't be set up
        print(f"Warning: Failed to set up logging: {e}")
        return None


def get_logger() -> Optional[logging.Logger]:
    """Get the WavePlot logger instance.
    
    Returns:
        Logger instance if logging is enabled, None otherwise.
    """
    logger = logging.getLogger("waveplot")
    if logger.handlers:
        return logger
    return None

