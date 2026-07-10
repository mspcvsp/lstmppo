import os

from loguru import logger


def init_reproducibility_logger():
    """
    Initializes a deterministic logger for reproducibility tests.

    This logger is used to capture all relevant information during training and evaluation, ensuring that the
    results can be reproduced exactly in future runs. It writes logs to a file in a deterministic manner, avoiding
    any non-deterministic behavior that could arise from concurrent logging or other sources.
    """
    # Create logs directory at repo root
    LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Remove default stderr handler
    logger.remove()

    # Add deterministic file sink
    logger.add(
        os.path.join(LOG_DIR, "repro.log"),
        format="{message}",  # pure text, diff-friendly
        level="DEBUG",  # use DEBUG to avoid cluttering stdout
        mode="w",  # overwrite each run
        enqueue=False,  # synchronous writes for determinism
    )
