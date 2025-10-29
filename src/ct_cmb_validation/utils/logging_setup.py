import logging
from typing import Optional


def setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("ct_cmb_validation")
    if logger.handlers:
        return logger

    level_num = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_num,
        format=fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Quiet noisy third-party loggers if present
    try:
        logging.getLogger("healpy").setLevel(logging.WARNING)
    except Exception:
        pass
    return logger
