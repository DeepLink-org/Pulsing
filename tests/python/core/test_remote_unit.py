"""Pure unit tests for core/remote.py — no ActorSystem required.

Covers wire format helpers, SingleValueIterator, _extract_methods,
_register_actor_metadata, ActorProxy attribute validation, and
_DelayedCallProxy edge cases.
"""

import asyncio

import pytest

from pulsing.core.remote import (
    _extract_methods,
    _register_actor_metadata,
    _unwrap_call,
    _unwrap_response,
    _wrap_call,
    _wrap_response,
    get_actor_metadata,
    ActorClass,
    ActorProxy,
    _actor_metadata_registry,
)


class _SingleValueIterator:
    """Local test helper: minimal single-value async iterator."""

    def __init__(self, value):
        self._value = value
        self._consumed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._consumed:
            raise StopAsyncIteration
        self._consumed = True
        return self._value


# ============================================================================
# _wrap_call / _unwrap_call
# ============================================================================


class TestWrapCall:
    def test_basic(self):
        msg = _wrap_call("greet", ("hello",), {"lang": "en"}, False)
        assert msg["__call__"] == "greet"
        assert msg["__async__"] is False
        assert msg["args"] == ("hello",)
        assert msg["kwargs"] == {"lang": "en"}

    def test_async_flag(self):
        msg = _wrap_call("stream", (), {}, True)
        assert msg["__async__"] is True

    def test_empty_args(self):
        msg = _wrap_call("no_args", (), {}, False)
        assert msg["args"] == ()
        assert msg["kwargs"] == {}


class TestUnwrapCall:
    def test_roundtrip(self):
        msg = _wrap_call("method", (1, 2, 3), {"key": "val"}, True)
        method, args, kwargs, is_async = _unwrap_call(msg)
        assert method == "method"
        assert args == (1, 2, 3)
        assert kwargs == {"key": "val"}
        assert is_async is True

    def test_missing_fields(self):
        method, args, kwargs, is_async = _unwrap_call({})
        assert method == ""
        assert args == ()
        assert kwargs == {}
        assert is_async is False

    def test_partial_message(self):
        msg = {"__call__": "foo"}
        method, args, kwargs, is_async = _unwrap_call(msg)
        assert method == "foo"
        assert args == ()
        assert kwargs == {}
        assert is_async is False


# ============================================================================
# _wrap_response / _unwrap_response
# ============================================================================


class TestWrapResponse:
    def test_success(self):
        resp = _wrap_response(result=42)
        assert resp["__result__"] == 42

    def test_error(self):
        resp = _wrap_response(error="something broke")
        assert resp["__error__"] == "something broke"

    def test_none_result(self):
        resp = _wrap_response(result=None)
        assert resp["__result__"] is None


class TestUnwrapResponse:
    def test_flat_result(self):
        resp = _wrap_response(result="ok")
        result, error = _unwrap_response(resp)
        assert result == "ok"
        assert error is None

    def test_flat_error(self):
        resp = _wrap_response(error="fail")
        result, error = _unwrap_response(resp)
        assert result is None
        assert error == "fail"

    def test_rust_json_result(self):
        result, error = _unwrap_response({"result": "top"})
        assert result == "top"
        assert error is None

    def test_rust_json_error(self):
        result, error = _unwrap_response({"error": "top_err"})
        assert result is None
        assert error == "top_err"

    def test_empty_dict(self):
        result, error = _unwrap_response({})
        assert result is None
        assert error is None

    def test_flat_takes_precedence_over_rust_json(self):
        resp = {"__error__": "flat", "result": "rust_json"}
        result, error = _unwrap_response(resp)
        assert error == "flat"
        assert result is None

    def test_stream_frame_final(self):
        frame = {"__final__": True, "__result__": 42}
        result, error = _unwrap_response(frame)
        assert result == 42
        assert error is None

    def test_stream_frame_error(self):
        frame = {"__error__": "stream failed"}
        result, error = _unwrap_response(frame)
        assert result is None
        assert error == "stream failed"


# ============================================================================
# _SingleValueIterator
# ============================================================================


class TestSingleValueIterator:
    @pytest.mark.asyncio
    async def test_yields_one_value(self):
        it = _SingleValueIterator(42)
        assert await it.__anext__() == 42
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    async def test_aiter_protocol(self):
        it = _SingleValueIterator("hello")
        assert it.__aiter__() is it
        items = []
        async for item in it:
            items.append(item)
        assert items == ["hello"]

    @pytest.mark.asyncio
    async def test_none_value(self):
        it = _SingleValueIterator(None)
        assert await it.__anext__() is None
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    @pytest.mark.asyncio
    async def test_dict_value(self):
        val = {"key": "val"}
        it = _SingleValueIterator(val)
        result = await it.__anext__()
        assert result == val


# ============================================================================
# _extract_methods
# ============================================================================


class TestExtractMethods:
    def test_basic_class(self):
        class MyClass:
            def public_a(self):
                pass

            async def public_b(self):
                pass

            def _private(self):
                pass

        methods, async_methods = _extract_methods(MyClass)
        assert "public_a" in methods
        assert "public_b" in methods
        assert "_private" not in methods
        assert "public_b" in async_methods
        assert "public_a" not in async_methods

    def test_async_generator(self):
        class GenClass:
            async def stream(self):
                yield 1

        methods, async_methods = _extract_methods(GenClass)
        assert "stream" in methods
        assert "stream" in async_methods

    def test_actor_class_unwrap(self):
        class Inner:
            def method_x(self):
                pass

        ac = ActorClass(Inner)
        methods, async_methods = _extract_methods(ac)
        assert "method_x" in methods

    def test_empty_class(self):
        class Empty:
            pass

        methods, async_methods = _extract_methods(Empty)
        assert methods == []
        assert async_methods == set()


# ============================================================================
# _register_actor_metadata / get_actor_metadata
# ============================================================================


class TestActorMetadataRegistry:
    def test_register_and_get(self):
        class FakeActor:
            pass

        _register_actor_metadata("test/fake", FakeActor)
        meta = get_actor_metadata("test/fake")
        assert meta is not None
        assert "python_class" in meta
        assert "FakeActor" in meta["python_class"]

    def test_get_nonexistent(self):
        assert get_actor_metadata("nonexistent/actor") is None


# ============================================================================
# ActorProxy (without ActorSystem — attribute validation only)
# ============================================================================


class TestActorProxyAttributes:
    def _make_proxy(self, methods=None, async_methods=None):
        class FakeRef:
            class actor_id:
                id = 12345

        return ActorProxy(FakeRef(), methods, async_methods)

    def test_private_attr_raises(self):
        proxy = self._make_proxy(["foo"])
        with pytest.raises(AttributeError, match="private"):
            _ = proxy._internal

    def test_unknown_method_raises(self):
        proxy = self._make_proxy(["foo", "bar"])
        with pytest.raises(AttributeError, match="No method"):
            _ = proxy.nonexistent

    def test_valid_method_returns_caller(self):
        proxy = self._make_proxy(["greet"], {"greet"})
        caller = proxy.greet
        assert caller is not None

    def test_any_proxy_allows_all(self):
        proxy = self._make_proxy(None, None)
        caller = proxy.any_method_name
        assert caller is not None

    def test_as_any(self):
        proxy = self._make_proxy(["foo"], {"foo"})
        any_proxy = proxy.as_any()
        assert any_proxy._method_names is None
        assert any_proxy._async_methods is None

    def test_ref_property(self):
        class FakeRef:
            class actor_id:
                id = 1

        ref = FakeRef()
        proxy = ActorProxy(ref)
        assert proxy.ref is ref

    def test_constructor_with_methods(self):
        class FakeRef:
            class actor_id:
                id = 1

        ref = FakeRef()
        proxy = ActorProxy(ref, method_names=["a", "b"], async_methods={"b"})
        assert "a" in proxy._method_names
        assert "b" in proxy._async_methods


# ============================================================================
# ActorClass (class-level, no system)
# ============================================================================


class TestActorClassUnit:
    def test_direct_call_returns_instance(self):
        class Counter:
            def __init__(self, init=0):
                self.value = init

        ac = ActorClass(Counter)
        instance = ac(init=5)
        assert instance.value == 5
        assert isinstance(instance, Counter)

    def test_methods_collected(self):
        class Svc:
            def method_a(self):
                pass

            async def method_b(self):
                pass

            def _private(self):
                pass

        ac = ActorClass(Svc)
        assert "method_a" in ac._methods
        assert "method_b" in ac._methods
        assert "_private" not in ac._methods
        assert "method_b" in ac._async_methods

    def test_registered_in_registry(self):
        from pulsing.core.remote import _actor_class_registry

        class UniqueTestCls:
            def ping(self):
                return "pong"

        ac = ActorClass(UniqueTestCls)
        key = f"{UniqueTestCls.__module__}.{UniqueTestCls.__name__}"
        assert key in _actor_class_registry
        assert _actor_class_registry[key] is UniqueTestCls

    def test_supervision_params(self):
        class Supervised:
            pass

        ac = ActorClass(
            Supervised,
            restart_policy="on-failure",
            max_restarts=10,
            min_backoff=0.5,
            max_backoff=60.0,
        )
        assert ac._restart_policy == "on-failure"
        assert ac._max_restarts == 10
        assert ac._min_backoff == 0.5
        assert ac._max_backoff == 60.0


# ============================================================================
# remote() decorator
# ============================================================================


class TestRemoteDecorator:
    def test_plain_decorator(self):
        from pulsing.core.remote import remote

        @remote
        class MyActor:
            def hello(self):
                return "hi"

        assert isinstance(MyActor, ActorClass)
        assert "hello" in MyActor._methods

    def test_decorator_with_params(self):
        from pulsing.core.remote import remote

        @remote(restart_policy="always", max_restarts=5)
        class MyActor2:
            def hello(self):
                return "hi"

        assert isinstance(MyActor2, ActorClass)
        assert MyActor2._restart_policy == "always"
        assert MyActor2._max_restarts == 5

    def test_decorator_preserves_class_name(self):
        from pulsing.core.remote import remote

        @remote
        class SpecificName:
            pass

        assert "SpecificName" in SpecificName._class_name


# ============================================================================
# _consume_task_exception
# ============================================================================


class TestConsumeTaskException:
    @pytest.mark.asyncio
    async def test_cancelled_task(self):
        from pulsing.core.remote import _consume_task_exception

        async def cancel_me():
            await asyncio.sleep(100)

        task = asyncio.create_task(cancel_me())
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        _consume_task_exception(task)

    @pytest.mark.asyncio
    async def test_stream_closed_error(self):
        from pulsing.core.remote import _consume_task_exception

        async def stream_closed():
            raise RuntimeError("stream closed by peer")

        task = asyncio.create_task(stream_closed())
        try:
            await task
        except RuntimeError:
            pass
        _consume_task_exception(task)

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        from pulsing.core.remote import _consume_task_exception

        async def fail():
            raise ValueError("unexpected")

        task = asyncio.create_task(fail())
        try:
            await task
        except ValueError:
            pass
        _consume_task_exception(task)

    @pytest.mark.asyncio
    async def test_connection_error(self):
        from pulsing.core.remote import _consume_task_exception

        async def conn_err():
            raise ConnectionError("connection closed unexpectedly")

        task = asyncio.create_task(conn_err())
        try:
            await task
        except ConnectionError:
            pass
        _consume_task_exception(task)

    @pytest.mark.asyncio
    async def test_os_error_non_stream(self):
        from pulsing.core.remote import _consume_task_exception

        async def os_err():
            raise OSError("disk full")

        task = asyncio.create_task(os_err())
        try:
            await task
        except OSError:
            pass
        _consume_task_exception(task)
