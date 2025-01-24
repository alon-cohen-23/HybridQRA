import logging
from logging.handlers import RotatingFileHandler

def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                "pipeline.log", 
                maxBytes=5 * 1024 * 1024,  # 5 MB
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("shared_logger")