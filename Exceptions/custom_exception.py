import sys
import traceback
from typing import Optional, Any


class DocumentPortalException(Exception):
    """
    A custom exception that captures specific context (file, line, traceback)
    automatically from the current execution context.
    """

    def __init__(
        self, message: str, original_exception: Optional[BaseException] = None
    ):
        self.error_message = message
        self.original_exception = original_exception

        # Capture the current exception info if not provided
        exc_type, exc_value, exc_tb = sys.exc_info()

        # If an explicit exception was passed, use its traceback instead
        if isinstance(original_exception, BaseException):
            exc_type = type(original_exception)
            exc_value = original_exception
            exc_tb = original_exception.__traceback__

        # Extract location info using the traceback module safely
        if exc_tb:
            # get_lines() or walk_tb can find the tip of the error
            summary = traceback.extract_tb(exc_tb)[-1]
            self.file_name = summary.filename
            self.lineno = summary.lineno
        else:
            self.file_name = "<unknown>"
            self.lineno = -1

        # Generate formatted traceback once
        if exc_type and exc_value:
            self.traceback_str = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.traceback_str = ""

        # Pass only the core message to the base Exception class
        super().__init__(self.error_message)

    def __str__(self):
        # Professional, structured format
        header = (
            f"[{self.__class__.__name__}] in {self.file_name} at line {self.lineno}"
        )
        body = f"Message: {self.error_message}"

        if self.traceback_str:
            return f"{header}\n{body}\n{'-'*20}\n{self.traceback_str}{'-'*20}"
        return f"{header} | {body}"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(file={self.file_name!r}, "
            f"line={self.lineno}, message={self.error_message!r})"
        )
