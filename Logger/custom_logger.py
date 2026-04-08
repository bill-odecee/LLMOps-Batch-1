import logging       # Python's built-in logging module (used internally by structlog as backend)
import os            # OS module for file path operations and directory creation
from datetime import datetime  # datetime class to generate timestamps for log filenames
import structlog     # Structured logging library — supports key-value pairs in log messages (e.g., user_id=123)

class CustomLogger:
    def __init__(self, log_dir="logs"):  # Constructor with a default log directory name "logs"
        #Ensure logs directory exists
        self.log_dir = os.path.join(os.getcwd(), log_dir)  # Build absolute path: <current_working_directory>/logs
        os.makedirs(self.log_dir, exist_ok=True)  # Create the logs directory if it doesn't already exist (no error if it does)
        # Timestamped log file (for persistence)
        log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"  # Generate a unique log filename using current timestamp, e.g., "2026_04_08_10_30_00.log"
        self.log_file_path = os.path.join(self.log_dir, log_file)  # Build full path to the log file, e.g., "/project/logs/2026_04_08_10_30_00.log"

    def get_logger(self, name=__file__):  # Returns a structlog logger; defaults to this file's name if no name is provided
        return structlog.get_logger(os.path.basename(name))  # Create and return a structlog logger named after the basename of the file (e.g., "custom_logger.py")
    
if __name__ == "__main__":  # Only runs when this file is executed directly (not when imported as a module)
    logger = CustomLogger().get_logger(__file__)  # Create a CustomLogger instance, set up the log directory/file, then get a structlog logger named after this file
    logger.info("This is an info message", user_id=12345, filename="report.pdf")  # Log an INFO-level message with structured key-value context (user_id and filename)
    logger.error("This is an error message", error="File not found", user_id=12345)  # Log an ERROR-level message with structured context (error description and user_id)
