"""Process-wide watchdog for CPU and accelerator stream timeouts.

This module exposes timeout guards backed by a libuv timer loop that runs on a
background thread. The module-level helpers operate on a single process-wide
:class:`Watchdog` instance; construct :class:`Watchdog` directly for an isolated
instance (primarily useful in tests).

All callbacks fire on the watchdog's background thread and therefore must not
block, otherwise other timeouts may not be serviced.
"""

from collections.abc import Callable
from datetime import timedelta

from torch._C._distributed_c10d import (
    _Watchdog as Watchdog,
    _WatchdogStreamTimeoutGuard,
    _WatchdogTimeoutGuard,
)


__all__ = [
    "Watchdog",
    "is_available",
    "context_timeout",
    "stream_timeout",
]


def is_available() -> bool:
    """Whether the watchdog timer backend (libuv) is compiled in."""
    return Watchdog.available()


def context_timeout(
    callback: Callable[[], None], timeout: timedelta
) -> "_WatchdogTimeoutGuard":
    """Return a context manager that runs ``callback`` if the guarded block takes
    longer than ``timeout``.

    The timer is cancelled on a clean exit, so ``callback`` only fires when the
    guarded (typically blocking) section overruns. ``callback`` runs on the
    watchdog background thread and must not block.
    """
    return Watchdog._singleton().context_timeout(callback, timeout)


def stream_timeout(
    timeout: timedelta,
    started_callback: Callable[[], None] | None = None,
    timedout_callback: Callable[[], None] | None = None,
) -> "_WatchdogStreamTimeoutGuard":
    """Return a context manager that monitors accelerator work enqueued within it.

    A start event is recorded on the current stream on entry and an end event on
    exit. Once the start event completes (the work has begun executing on the
    device), ``started_callback`` fires and a ``timeout`` clock begins; if the
    end event has not completed within ``timeout`` of that point,
    ``timedout_callback`` fires. Both callbacks run on the watchdog background
    thread and must not block.
    """
    return Watchdog._singleton().stream_timeout(
        timeout, started_callback, timedout_callback
    )
