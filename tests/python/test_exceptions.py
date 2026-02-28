"""Tests for pulsing.exceptions — cover all exception classes, constructors, and attributes."""

import pytest

from pulsing.exceptions import (
    PulsingActorError,
    PulsingBusinessError,
    PulsingError,
    PulsingRuntimeError,
    PulsingSystemError,
    PulsingTimeoutError,
    PulsingUnsupportedError,
)


class TestPulsingError:
    def test_base_exception(self):
        err = PulsingError("base error")
        assert str(err) == "base error"
        assert isinstance(err, Exception)

    def test_is_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise PulsingError("catch me")


class TestPulsingRuntimeError:
    def test_basic(self):
        err = PulsingRuntimeError("system down")
        assert str(err) == "system down"
        assert err.cause is None
        assert isinstance(err, PulsingError)

    def test_with_cause(self):
        cause = ConnectionError("refused")
        err = PulsingRuntimeError("transport failed", cause=cause)
        assert str(err) == "transport failed"
        assert err.cause is cause

    def test_cause_none_explicit(self):
        err = PulsingRuntimeError("msg", cause=None)
        assert err.cause is None


class TestPulsingActorError:
    def test_basic(self):
        err = PulsingActorError("actor failed")
        assert str(err) == "actor failed"
        assert err.actor_name is None
        assert err.cause is None
        assert isinstance(err, PulsingError)

    def test_with_actor_name(self):
        err = PulsingActorError("fail", actor_name="my_actor")
        assert err.actor_name == "my_actor"

    def test_with_cause(self):
        cause = ValueError("bad value")
        err = PulsingActorError("fail", cause=cause)
        assert err.cause is cause

    def test_with_all_params(self):
        cause = RuntimeError("inner")
        err = PulsingActorError("fail", actor_name="worker/1", cause=cause)
        assert str(err) == "fail"
        assert err.actor_name == "worker/1"
        assert err.cause is cause


class TestPulsingBusinessError:
    def test_basic(self):
        err = PulsingBusinessError(400, "Bad Request")
        assert str(err) == "[400] Bad Request"
        assert err.code == 400
        assert err.message == "Bad Request"
        assert err.details is None
        assert isinstance(err, PulsingActorError)

    def test_with_details(self):
        err = PulsingBusinessError(
            422, "Validation failed", details="age must be >= 18"
        )
        assert err.code == 422
        assert err.message == "Validation failed"
        assert err.details == "age must be >= 18"
        assert str(err) == "[422] Validation failed"

    def test_inherits_actor_error(self):
        err = PulsingBusinessError(500, "Server error")
        assert isinstance(err, PulsingActorError)
        assert isinstance(err, PulsingError)
        assert err.cause is None

    def test_zero_code(self):
        err = PulsingBusinessError(0, "Unknown")
        assert err.code == 0
        assert str(err) == "[0] Unknown"


class TestPulsingSystemError:
    def test_basic(self):
        err = PulsingSystemError("OOM")
        assert str(err) == "OOM"
        assert err.error == "OOM"
        assert err.recoverable is True
        assert isinstance(err, PulsingActorError)

    def test_not_recoverable(self):
        err = PulsingSystemError("fatal crash", recoverable=False)
        assert err.recoverable is False
        assert err.error == "fatal crash"

    def test_explicitly_recoverable(self):
        err = PulsingSystemError("transient", recoverable=True)
        assert err.recoverable is True


class TestPulsingTimeoutError:
    def test_basic(self):
        err = PulsingTimeoutError("fetch")
        assert err.operation == "fetch"
        assert err.duration_ms == 0
        assert "fetch" in str(err)
        assert "0ms" in str(err)
        assert isinstance(err, PulsingActorError)

    def test_with_duration(self):
        err = PulsingTimeoutError("db_query", duration_ms=5000)
        assert err.operation == "db_query"
        assert err.duration_ms == 5000
        assert str(err) == "Operation 'db_query' timed out after 5000ms"

    def test_zero_duration(self):
        err = PulsingTimeoutError("op", duration_ms=0)
        assert err.duration_ms == 0


class TestPulsingUnsupportedError:
    def test_basic(self):
        err = PulsingUnsupportedError("legacy_rpc")
        assert err.operation == "legacy_rpc"
        assert str(err) == "Unsupported operation: legacy_rpc"
        assert isinstance(err, PulsingActorError)

    def test_inherits_full_chain(self):
        err = PulsingUnsupportedError("op")
        assert isinstance(err, PulsingActorError)
        assert isinstance(err, PulsingError)
        assert isinstance(err, Exception)
        assert err.cause is None


class TestExceptionHierarchy:
    """Verify the full inheritance chain for catch-all patterns."""

    def test_catch_all_pulsing_error(self):
        exceptions = [
            PulsingRuntimeError("rt"),
            PulsingActorError("actor"),
            PulsingBusinessError(400, "biz"),
            PulsingSystemError("sys"),
            PulsingTimeoutError("op"),
            PulsingUnsupportedError("op"),
        ]
        for exc in exceptions:
            assert isinstance(
                exc, PulsingError
            ), f"{type(exc).__name__} not PulsingError"

    def test_catch_actor_errors_only(self):
        actor_errors = [
            PulsingActorError("a"),
            PulsingBusinessError(400, "b"),
            PulsingSystemError("s"),
            PulsingTimeoutError("t"),
            PulsingUnsupportedError("u"),
        ]
        for exc in actor_errors:
            assert isinstance(exc, PulsingActorError)

        runtime_err = PulsingRuntimeError("rt")
        assert not isinstance(runtime_err, PulsingActorError)

    def test_raise_and_catch_business(self):
        with pytest.raises(PulsingBusinessError) as exc_info:
            raise PulsingBusinessError(403, "Forbidden", details="not allowed")
        assert exc_info.value.code == 403
        assert exc_info.value.details == "not allowed"

    def test_raise_and_catch_as_parent(self):
        with pytest.raises(PulsingError):
            raise PulsingTimeoutError("slow_op", duration_ms=10000)
