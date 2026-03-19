"""
pulsing.subprocess — subprocess-compatible sync API running processes as remote actors.

The public interface is **synchronous**, identical to the stdlib subprocess module.
Internally, a dedicated background event loop thread handles all async actor calls,
so there is no risk of deadlock regardless of the caller's context.

Usage (drop-in replacement — no await needed)::

    # stdlib
    import subprocess

    # pulsing version — processes run as remote actors
    import pulsing.subprocess as subprocess

    result = subprocess.run(["echo", "hello"], capture_output=True)
    proc   = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(input=b"hi")
"""

import subprocess

PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT
DEVNULL = subprocess.DEVNULL

from .process import ProcessActor
from .popen import Popen, CompletedProcess, run, call, check_output, check_call

__all__ = [
    "PIPE",
    "STDOUT",
    "DEVNULL",
    "Popen",
    "CompletedProcess",
    "run",
    "call",
    "check_call",
    "check_output",
    "ProcessActor",
]
