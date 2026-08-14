"""DXMT Python SDK Exceptions.

This module defines all custom exceptions used by the DXMT Python SDK.
"""


class DXMTError(Exception):
    """Base exception for all DXMT errors."""

    pass


class DXMTConnectionError(DXMTError):
    """Raised when connection to DXMT server fails."""

    pass


class DXMTTableError(DXMTError):
    """Raised when a table operation fails."""

    pass


class DXMTRecordError(DXMTError):
    """Raised when a record operation fails."""

    pass


class DXMTSearchError(DXMTError):
    """Raised when a search operation fails."""

    pass


class DXMTValidationError(DXMTError):
    """Raised when input validation fails."""

    pass


class DXMTTimeoutError(DXMTError):
    """Raised when a request times out."""

    pass
