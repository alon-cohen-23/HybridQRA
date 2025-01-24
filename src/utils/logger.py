import logging
from logging.handlers import RotatingFileHandler

# Centralized logger setup
def get_logger():
    logger = logging.getLogger("shared_logger")
    logger.setLevel(logging.INFO)  # Adjust logging level as needed

    # File handler with rotation
    file_handler = RotatingFileHandler(
        "pipeline.log",  # Log file name
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3  # Keep 3 backup files
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
