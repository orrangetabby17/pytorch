#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <c10/core/Event.h>
#include <c10/macros/Export.h>

namespace c10d::watchdog {

// Callback invoked from the watchdog's timer loop thread. Callbacks must not
// block, since all timeouts are serviced on that single thread.
using Callback = std::function<void()>;

// Whether the watchdog is backed by a running timer loop. This requires the
// libuv timer backend (TORCH_USE_LIBUV) to be compiled in.
TORCH_API bool isAvailable();

// A process-wide timer/timeout service backed by a libuv event loop running on
// a dedicated background thread.
//
// A single global instance is available via singleton(); additional instances
// can be constructed directly, which is primarily useful for tests that want an
// isolated loop.
//
// Two kinds of timeouts are supported:
//   - registerTimer: a CPU timeout that fires unless cancelled in time. This
//     backs a guard around a blocking CPU section.
//   - registerStreamTimeout: a device-stream timeout bracketed by two events.
class TORCH_API Watchdog {
 public:
  Watchdog();
  ~Watchdog();
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  Watchdog(Watchdog&&) = delete;
  Watchdog& operator=(Watchdog&&) = delete;

  // Process-wide instance. Intentionally leaked so the background thread is
  // never joined during interpreter shutdown.
  static const std::shared_ptr<Watchdog>& singleton();

  // Schedules onTimeout to fire after timeout has elapsed. Returns an id that
  // can be passed to cancelTimer to cancel it before it fires (e.g. when a
  // guarded CPU section completes in time).
  uint64_t registerTimer(std::chrono::milliseconds timeout, Callback onTimeout);

  // Cancels a timer previously returned by registerTimer. No-op if the timer
  // has already fired or was already cancelled.
  void cancelTimer(uint64_t id);

  // Monitors a device operation bracketed by two events recorded on the same
  // stream. startEvent marks the beginning of the operation and endEvent its
  // completion; both must already be recorded by the caller.
  //
  // Once startEvent completes (the operation has started executing on device),
  // onStarted fires and a timeout clock begins. If endEvent has not completed
  // within timeout of that point, onTimedout fires. Either callback may be
  // null.
  void registerStreamTimeout(
      c10::Event startEvent,
      c10::Event endEvent,
      std::chrono::milliseconds timeout,
      Callback onStarted,
      Callback onTimedout);

  // Number of stream timeouts currently being monitored. Primarily for tests.
  size_t numActiveStreamTimeouts() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace c10d::watchdog
