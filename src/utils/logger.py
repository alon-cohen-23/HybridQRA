import logging

# Centralized logger setup
def get_logger():
    logging.basicConfig(
        level=logging.INFO,  # Adjust logging level as needed
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("pipeline.log"),  # Shared log file
            logging.StreamHandler()  # Log to console as well
        ]
    )
    return logging.getLogger("shared_logger")

