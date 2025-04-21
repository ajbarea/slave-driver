import logging
import os
from datetime import datetime


def setup_logger(name, level=logging.INFO, log_to_file=False, log_dir="logs"):
    """
    Configure and return a logger with the specified name and level.

    Args:
        name (str): Logger name, typically __name__ of the calling module
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file (bool): Whether to also log to a file
        log_dir (str): Directory for log files if log_to_file is True

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates when reconfiguring
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler with a formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally add file handler
    if log_to_file:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name.split('.')[-1]}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name, level=logging.INFO, log_to_file=False):
    """
    Get a logger with the given name. Convenience function.

    Args:
        name (str): Logger name
        level (int): Logging level
        log_to_file (bool): Whether to log to a file

    Returns:
        logging.Logger: Configured logger
    """
    return setup_logger(name, level, log_to_file)
