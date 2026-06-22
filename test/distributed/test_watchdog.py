# Owner(s): ["oncall: distributed"]

import threading
import time
from datetime import timedelta

import torch
import torch.distributed.watchdog as watchdog
from torch.distributed.watchdog import Watchdog
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase


class WatchdogTest(TestCase):
    def test_is_available(self):
        # The libuv timer backend should be compiled into a standard build.
        self.assertTrue(watchdog.is_available())

    def test_singleton_is_stable(self):
        self.assertIs(Watchdog._singleton(), Watchdog._singleton())

    def test_context_timeout_fires_on_overrun(self):
        wd = Watchdog()
        fired = threading.Event()
        with wd.context_timeout(fired.set, timedelta(milliseconds=50)):
            # time.sleep releases the GIL so the callback thread can run.
            time.sleep(0.5)
        self.assertTrue(fired.is_set())

    def test_context_timeout_cancelled_on_clean_exit(self):
        wd = Watchdog()
        fired = threading.Event()
        with wd.context_timeout(fired.set, timedelta(seconds=10)):
            time.sleep(0.05)
        # The timer was cancelled on exit, so the callback must not fire.
        self.assertFalse(fired.wait(timeout=0.3))

    def test_context_timeout_via_module_singleton(self):
        fired = threading.Event()
        with watchdog.context_timeout(fired.set, timedelta(milliseconds=50)):
            time.sleep(0.5)
        self.assertTrue(fired.is_set())

    def test_context_timeout_callback_exception_is_swallowed(self):
        wd = Watchdog()

        def boom():
            raise RuntimeError("callback error")

        # A raising callback must not crash the watchdog loop thread, and the
        # watchdog must keep working afterwards.
        with wd.context_timeout(boom, timedelta(milliseconds=50)):
            time.sleep(0.3)

        fired = threading.Event()
        with wd.context_timeout(fired.set, timedelta(milliseconds=50)):
            time.sleep(0.5)
        self.assertTrue(fired.is_set())

    def test_no_active_stream_timeouts(self):
        wd = Watchdog()
        self.assertEqual(wd.num_active_stream_timeouts(), 0)

    @skipIfRocm
    def test_stream_timeout_fires_when_busy(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = Watchdog()
        started = threading.Event()
        timed_out = threading.Event()
        with wd.stream_timeout(
            timedelta(milliseconds=1),
            started_callback=started.set,
            timedout_callback=timed_out.set,
        ):
            # Keep the stream busy well past the (tiny) timeout.
            torch.cuda._sleep(500_000_000)
        self.assertTrue(started.wait(timeout=10.0))
        self.assertTrue(timed_out.wait(timeout=10.0))
        torch.cuda.synchronize()

    @skipIfRocm
    def test_stream_timeout_completes_without_firing(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wd = Watchdog()
        started = threading.Event()
        timed_out = threading.Event()
        with wd.stream_timeout(
            timedelta(seconds=30),
            started_callback=started.set,
            timedout_callback=timed_out.set,
        ):
            torch.cuda._sleep(1_000_000)
        torch.cuda.synchronize()
        self.assertTrue(started.wait(timeout=10.0))
        self.assertFalse(timed_out.wait(timeout=0.5))
        # The completed monitor is eventually reaped by the poll loop.
        deadline = time.time() + 5.0
        while wd.num_active_stream_timeouts() > 0 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(wd.num_active_stream_timeouts(), 0)


if __name__ == "__main__":
    run_tests()
