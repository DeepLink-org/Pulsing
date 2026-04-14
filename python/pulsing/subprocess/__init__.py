"""
pulsing.subprocess — subprocess-compatible synchronous API.

Calls without ``resources`` delegate to Python's native ``subprocess`` module.
Passing a non-empty ``resources=...`` runs the subprocess through Pulsing
actors only when ``USE_POLSING_SUBPROCESS`` is enabled. In that resource-backed
mode this module lazily initializes Pulsing internally, so callers do not need
to call ``await pul.init()`` before using it. These synchronous APIs must not
be called from the active Pulsing event loop thread; in async code, prefer
native async APIs or move subprocess calls into ``asyncio.to_thread(...)``.

Usage::

    import pulsing.subprocess as subprocess

    result = subprocess.run(["echo", "hello"], capture_output=True)
    proc = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(input=b"hi")
    remote = subprocess.run(
        ["echo", "hello from pulsing"],
        capture_output=True,
        resources={"num_cpus": 1},
    )

Shell usage::

    python examples/python/subprocess_example.py --resources
    USE_POLSING_SUBPROCESS=1 python examples/python/subprocess_example.py --resources
"""

import subprocess

PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT
DEVNULL = subprocess.DEVNULL
TimeoutExpired = subprocess.TimeoutExpired

from .process import ProcessActor
from .popen import Popen, CompletedProcess, run, call, check_output, check_call

__all__ = [
    "PIPE",
    "STDOUT",
    "DEVNULL",
    "TimeoutExpired",
    "Popen",
    "CompletedProcess",
    "run",
    "call",
    "check_call",
    "check_output",
    "ProcessActor",
]
