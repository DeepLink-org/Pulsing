"""Pulsing exception hierarchy.

This module provides Python exceptions that correspond to Rust error types.
The exceptions are defined in Python but correspond to Rust error types defined
in crates/pulsing-actor/src/error.rs using thiserror.

Errors are divided into two categories (matching Rust error structure):

1. PulsingRuntimeError: Framework/system-level errors
   Corresponds to: pulsing_actor::error::RuntimeError

   These are framework-level errors, not caused by user code:
   - Actor system errors (NotFound, Stopped, etc.)
   - Transport errors (ConnectionFailed, etc.)
   - Cluster errors (NodeNotFound, etc.)
   - Config errors (InvalidValue, etc.)
   - I/O errors, Serialization errors

2. PulsingActorError: User Actor execution errors
   Corresponds to: pulsing_actor::error::ActorError

   These are errors raised by user code during Actor execution:
   - Business errors (user input errors) → PulsingBusinessError
   - System errors (internal errors from user code) → PulsingSystemError
   - Timeout errors (operation timeouts) → PulsingTimeoutError
   - Unsupported errors (unsupported operations) → PulsingUnsupportedError

Note: Due to PyO3 abi3 limitations, we define exceptions in Python and
Rust code raises them using PyRuntimeError with message prefixes.
The Python layer can catch and re-raise as appropriate types.

For Actor execution errors, use the specific exception types below which
will be automatically converted to Rust ActorError variants.
"""


class PulsingError(Exception):
    """Base exception for all Pulsing errors.

    This corresponds to pulsing_actor::error::PulsingError in Rust.
    """

    pass


class PulsingRuntimeError(PulsingError):
    """Framework/system-level errors.

    This corresponds to pulsing_actor::error::RuntimeError in Rust.

    These are framework-level errors, not caused by user code:
    - Actor system errors (NotFound, Stopped, etc.)
    - Transport errors (ConnectionFailed, etc.)
    - Cluster errors (NodeNotFound, etc.)
    - Config errors (InvalidValue, etc.)
    - I/O errors
    - Serialization errors
    """

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class PulsingActorError(PulsingError):
    """User Actor execution errors.

    This corresponds to pulsing_actor::error::ActorError in Rust.

    These are errors raised by user code during Actor execution:
    - Business errors (user input errors)
    - System errors (internal errors from user code)
    - Timeout errors (operation timeouts)
    - Unsupported errors (unsupported operations)

    Note: Framework-level errors like "Actor not found" are RuntimeError,
    not ActorError.
    """

    def __init__(
        self,
        message: str,
        actor_name: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.actor_name = actor_name
        self.cause = cause


# ============================================================================
# Business-level error types (automatically converted to ActorError)
# ============================================================================


class PulsingBusinessError(PulsingActorError):
    """Business error: User input error, business logic error.

    These errors are recoverable and should be returned to the caller.
    Automatically converted to ActorError::Business in Rust.

    Example:
        @remote
        class UserActor:
            async def validate_age(self, age: int) -> bool:
                if age < 18:
                    raise PulsingBusinessError(400, "Age must be >= 18",
                                             details="User validation failed")
                return True
    """

    def __init__(self, code: int, message: str, details: str | None = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code}] {message}", cause=None)


class PulsingSystemError(PulsingActorError):
    """System error: Internal error, resource error.

    May trigger Actor restart depending on recoverable flag.
    Automatically converted to ActorError::System in Rust.

    Example:
        @remote
        class DataProcessor:
            async def process(self, data: str) -> str:
                try:
                    return process_data(data)
                except Exception as e:
                    raise PulsingSystemError(f"Processing failed: {e}", recoverable=True)
    """

    def __init__(self, error: str, recoverable: bool = True):
        self.error = error
        self.recoverable = recoverable
        super().__init__(error, cause=None)


class PulsingTimeoutError(PulsingActorError):
    """Timeout error: Operation timed out.

    Usually recoverable, can be retried.
    Automatically converted to ActorError::Timeout in Rust.

    Example:
        @remote
        class NetworkActor:
            async def fetch(self, url: str) -> str:
                try:
                    return await asyncio.wait_for(httpx.get(url), timeout=5.0)
                except asyncio.TimeoutError:
                    raise PulsingTimeoutError("fetch", duration_ms=5000)
    """

    def __init__(self, operation: str, duration_ms: int = 0):
        self.operation = operation
        self.duration_ms = duration_ms
        super().__init__(
            f"Operation '{operation}' timed out after {duration_ms}ms", cause=None
        )


class PulsingUnsupportedError(PulsingActorError):
    """Unsupported operation error.

    Not recoverable. Indicates that the requested operation is not supported.
    Automatically converted to ActorError::Unsupported in Rust.

    Example:
        @remote
        class LegacyActor:
            async def process(self, data: str) -> str:
                if data.startswith("legacy:"):
                    raise PulsingUnsupportedError("process")
                return process_data(data)
    """

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(f"Unsupported operation: {operation}", cause=None)
