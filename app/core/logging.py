from loguru import logger
from pathlib import Path
from .config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# FIXED: retention=10  (keep last 10 rotated log files)
logger.add(
    LOGS_DIR / "app.log",
    rotation="10 MB",
    retention=10,           
    enqueue=True,
    backtrace=False,
    diagnose=False,
    compression="zip"       
)
