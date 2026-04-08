"""Logger module for Document Portal
Provides a global logger instance for application-wide logging.
"""

from logger.custom_logger import CustomLogger

# Create a global logger instance for use throughout the application
GLOBAL_LOGGER = CustomLogger().get_logger("document_portal")