import logging
import sys


def create_repro_logger(train_seed, env_seed, log_dir):
    logger = logging.getLogger(f"repro_{train_seed}_{env_seed}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # prevents global leak

    # Remove any existing handlers (important for repeated runs)
    logger.handlers.clear()

    # File handler (DEBUG only)
    fh = logging.FileHandler(f"{log_dir}/repro_seed_{train_seed}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(f"TRAIN={train_seed} ENV={env_seed} | %(message)s"))
    logger.addHandler(fh)

    # Stdout handler (INFO only)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(f"TRAIN={train_seed} ENV={env_seed} | %(message)s"))
    logger.addHandler(sh)

    return logger
