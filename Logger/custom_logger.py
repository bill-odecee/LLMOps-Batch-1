import logging
import os
import sys
from datetime import datetime
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs", level=logging.INFO):
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%Y_%m_%d')}.log"
        self.log_file_path = os.path.join(self.log_dir, log_file)

        self._configure_structlog(level)

    def _configure_structlog(self, level):
        # 1. Standard Python logging setup (The Backend)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.FileHandler(self.log_file_path),  # Writes to your file
                logging.StreamHandler(sys.stdout),  # Also prints to console
            ],
        )

        # 2. Structlog configuration (The Frontend)
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # Use JSON for files, but we'll use ConsoleRenderer for now for readability
                structlog.dev.ConsoleRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def get_logger(self, name=None):
        # Use the provided name or fall back to the calling module
        return structlog.get_logger(name or "app_logger")


# Usage
if __name__ == "__main__":
    # Initialize ONCE at the start of your app
    log_manager = CustomLogger()
    logger = log_manager.get_logger("document_portal")

    logger.info("Service started", version="1.0.0")
    logger.error("Upload failed", user_id=123, error="Timeout")
