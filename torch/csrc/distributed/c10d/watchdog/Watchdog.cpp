#include <torch/csrc/distributed/c10d/watchdog/Watchdog.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <c10/util/thread_name.h>

#ifdef TORCH_USE_LIBUV
#include <uv.h>
#endif

namespace c10d::watchdog {

bool isAvailable() {
#ifdef TORCH_USE_LIBUV
  return true;
#else
  return false;
#endif
}

#ifdef TORCH_USE_LIBUV

namespace {
// How often the loop polls device events while stream timeouts are active.
constexpr uint64_t kPollIntervalMs = 10;
} // namespace

struct Watchdog::Impl {
  // A scheduled one-shot CPU timer.
  struct CpuTimer {
    uv_timer_t handle{};
    Impl* impl{nullptr};
    uint64_t id{0};
    Callback onTimeout;
  };

  // A device operation being monitored via two events recorded on a stream.
  struct StreamTimeout {
    c10::Event startEvent;
    c10::Event endEvent;
    std::chrono::milliseconds timeout;
    Callback onStarted;
    Callback onTimedout;
    bool started{false};
    std::chrono::steady_clock::time_point startedAt;
  };

  // libuv loop and the handles that drive it. All uv_* state below is only
  // touched from the loop thread, except requestAsync_/stopAsync_ which are
  // signalled cross-thread via the thread-safe uv_async_send.
  uv_loop_t loop_{};
  uv_async_t requestAsync_{};
  uv_async_t stopAsync_{};
  uv_timer_t pollTimer_{};
  bool pollActive_{false};
  std::thread thread_;
  std::atomic<bool> running_{false};

  // Work to run on the loop thread, posted from arbitrary threads.
  std::mutex requestMutex_;
  std::vector<std::function<void()>> requests_;

  // Loop-thread-only state.
  std::unordered_map<uint64_t, std::unique_ptr<CpuTimer>> cpuTimers_;
  std::vector<std::shared_ptr<StreamTimeout>> streamTimeouts_;

  std::atomic<uint64_t> nextId_{1};
  std::atomic<size_t> activeStreamTimeouts_{0};
  std::atomic<bool> stopping_{false};

  // During shutdown, user callbacks are moved here so they are destroyed by the
  // thread that joins the loop (which, for Python callbacks, holds the GIL)
  // rather than on the loop thread. Only touched on the loop thread before the
  // join and on the joining thread after; the join is the synchronization
  // point.
  std::vector<Callback> pendingCallbackDeletion_;

  // Events pending destruction. c10::Event destruction (e.g. cudaEventDestroy)
  // can block, so it is deferred off the loop thread and drained by callers.
  std::mutex delMutex_;
  std::vector<c10::Event> delQueue_;

  Impl();
  ~Impl();
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  void start();
  void enqueue(std::function<void()> request);
  void drainDelQueue();

  // Loop thread.
  void run();
  void onRequests();
  void onStop();
  void pollStreamTimeouts();
  void armCpuTimer(
      uint64_t id,
      std::chrono::milliseconds timeout,
      Callback onTimeout);
  void closeCpuTimer(uint64_t id);
  void maybeStartPollTimer();

  static Impl& fromHandle(uv_handle_t* handle) {
    return *static_cast<Impl*>(uv_handle_get_data(handle));
  }
};

Watchdog::Impl::Impl() {
  TORCH_CHECK(uv_loop_init(&loop_) == 0, "Failed to init watchdog uv loop");

  TORCH_CHECK(
      uv_async_init(
          &loop_,
          &requestAsync_,
          [](uv_async_t* h) {
            fromHandle(reinterpret_cast<uv_handle_t*>(h)).onRequests();
          }) == 0,
      "Failed to init watchdog request handle");
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&requestAsync_), this);

  TORCH_CHECK(
      uv_async_init(
          &loop_,
          &stopAsync_,
          [](uv_async_t* h) {
            fromHandle(reinterpret_cast<uv_handle_t*>(h)).onStop();
          }) == 0,
      "Failed to init watchdog stop handle");
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&stopAsync_), this);

  TORCH_CHECK(
      uv_timer_init(&loop_, &pollTimer_) == 0,
      "Failed to init watchdog poll timer");
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&pollTimer_), this);
}

Watchdog::Impl::~Impl() {
  if (running_.load()) {
    uv_async_send(&stopAsync_);
    thread_.join();
    running_.store(false);
  }
  drainDelQueue();
}

void Watchdog::Impl::start() {
  thread_ = std::thread([this] {
    c10::setThreadName("pt_watchdog");
    run();
  });
  running_.store(true);
}

void Watchdog::Impl::run() {
  uv_run(&loop_, UV_RUN_DEFAULT);

  // onStop closed all handles; pump the loop until their close callbacks have
  // fired and the loop can be closed cleanly.
  while (uv_loop_close(&loop_) != 0) {
    if (uv_run(&loop_, UV_RUN_NOWAIT) != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void Watchdog::Impl::enqueue(std::function<void()> request) {
  {
    std::lock_guard<std::mutex> lock(requestMutex_);
    requests_.push_back(std::move(request));
  }
  uv_async_send(&requestAsync_);
}

void Watchdog::Impl::onRequests() {
  // Once shutdown has started, don't arm new work; pending requests are
  // discarded (and destroyed) by the joining thread.
  if (stopping_.load()) {
    return;
  }
  std::vector<std::function<void()>> requests;
  {
    std::lock_guard<std::mutex> lock(requestMutex_);
    requests.swap(requests_);
  }
  for (auto& request : requests) {
    request();
  }
}

void Watchdog::Impl::armCpuTimer(
    uint64_t id,
    std::chrono::milliseconds timeout,
    Callback onTimeout) {
  auto timer = std::make_unique<CpuTimer>();
  timer->impl = this;
  timer->id = id;
  timer->onTimeout = std::move(onTimeout);
  uv_timer_init(&loop_, &timer->handle);
  uv_handle_set_data(
      reinterpret_cast<uv_handle_t*>(&timer->handle), timer.get());

  CpuTimer* raw = timer.get();
  cpuTimers_[id] = std::move(timer);
  uv_timer_start(
      &raw->handle,
      [](uv_timer_t* h) {
        auto* t = static_cast<CpuTimer*>(
            uv_handle_get_data(reinterpret_cast<uv_handle_t*>(h)));
        if (t->onTimeout) {
          t->onTimeout();
        }
        t->impl->closeCpuTimer(t->id);
      },
      static_cast<uint64_t>(timeout.count()),
      /*repeat=*/0);
}

void Watchdog::Impl::closeCpuTimer(uint64_t id) {
  auto it = cpuTimers_.find(id);
  if (it == cpuTimers_.end()) {
    return;
  }
  // Transfer ownership to the close callback, which deletes the timer once
  // libuv has finished closing the handle.
  CpuTimer* timer = it->second.release();
  cpuTimers_.erase(it);
  uv_timer_stop(&timer->handle);
  uv_close(reinterpret_cast<uv_handle_t*>(&timer->handle), [](uv_handle_t* h) {
    delete static_cast<CpuTimer*>(uv_handle_get_data(h));
  });
}

void Watchdog::Impl::maybeStartPollTimer() {
  if (pollActive_) {
    return;
  }
  uv_timer_start(
      &pollTimer_,
      [](uv_timer_t* h) {
        fromHandle(reinterpret_cast<uv_handle_t*>(h)).pollStreamTimeouts();
      },
      kPollIntervalMs,
      kPollIntervalMs);
  pollActive_ = true;
}

void Watchdog::Impl::pollStreamTimeouts() {
  auto now = std::chrono::steady_clock::now();
  std::vector<c10::Event> toDelete;

  auto it = streamTimeouts_.begin();
  while (it != streamTimeouts_.end()) {
    StreamTimeout& st = **it;
    bool done = false;

    if (!st.started && st.startEvent.query()) {
      st.started = true;
      st.startedAt = now;
      if (st.onStarted) {
        st.onStarted();
      }
    }

    if (st.started) {
      if (st.endEvent.query()) {
        done = true;
      } else if (now - st.startedAt > st.timeout) {
        if (st.onTimedout) {
          st.onTimedout();
        }
        done = true;
      }
    }

    if (done) {
      toDelete.push_back(std::move(st.startEvent));
      toDelete.push_back(std::move(st.endEvent));
      it = streamTimeouts_.erase(it);
    } else {
      ++it;
    }
  }

  activeStreamTimeouts_.store(streamTimeouts_.size());
  if (streamTimeouts_.empty() && pollActive_) {
    uv_timer_stop(&pollTimer_);
    pollActive_ = false;
  }

  if (!toDelete.empty()) {
    std::lock_guard<std::mutex> lock(delMutex_);
    for (auto& event : toDelete) {
      delQueue_.push_back(std::move(event));
    }
  }
}

void Watchdog::Impl::onStop() {
  stopping_.store(true);

  for (auto& [id, timer] : cpuTimers_) {
    if (timer->onTimeout) {
      pendingCallbackDeletion_.push_back(std::move(timer->onTimeout));
    }
    CpuTimer* raw = timer.release();
    uv_timer_stop(&raw->handle);
    uv_close(reinterpret_cast<uv_handle_t*>(&raw->handle), [](uv_handle_t* h) {
      delete static_cast<CpuTimer*>(uv_handle_get_data(h));
    });
  }
  cpuTimers_.clear();

  {
    std::lock_guard<std::mutex> lock(delMutex_);
    for (auto& st : streamTimeouts_) {
      if (st->onStarted) {
        pendingCallbackDeletion_.push_back(std::move(st->onStarted));
      }
      if (st->onTimedout) {
        pendingCallbackDeletion_.push_back(std::move(st->onTimedout));
      }
      delQueue_.push_back(std::move(st->startEvent));
      delQueue_.push_back(std::move(st->endEvent));
    }
  }
  streamTimeouts_.clear();
  activeStreamTimeouts_.store(0);

  if (pollActive_) {
    uv_timer_stop(&pollTimer_);
    pollActive_ = false;
  }
  uv_close(reinterpret_cast<uv_handle_t*>(&pollTimer_), nullptr);
  uv_close(reinterpret_cast<uv_handle_t*>(&requestAsync_), nullptr);
  uv_close(reinterpret_cast<uv_handle_t*>(&stopAsync_), nullptr);
  uv_stop(&loop_);
}

void Watchdog::Impl::drainDelQueue() {
  std::vector<c10::Event> toDelete;
  {
    std::lock_guard<std::mutex> lock(delMutex_);
    toDelete.swap(delQueue_);
  }
  // Events are destroyed here, off the loop thread.
}

Watchdog::Watchdog() {
  impl_ = std::make_unique<Impl>();
  impl_->start();
}

Watchdog::~Watchdog() = default;

uint64_t Watchdog::registerTimer(
    std::chrono::milliseconds timeout,
    Callback onTimeout) {
  impl_->drainDelQueue();
  uint64_t id = impl_->nextId_.fetch_add(1);
  impl_->enqueue(
      [impl = impl_.get(), id, timeout, cb = std::move(onTimeout)]() mutable {
        impl->armCpuTimer(id, timeout, std::move(cb));
      });
  return id;
}

void Watchdog::cancelTimer(uint64_t id) {
  impl_->enqueue([impl = impl_.get(), id]() { impl->closeCpuTimer(id); });
}

void Watchdog::registerStreamTimeout(
    c10::Event startEvent,
    c10::Event endEvent,
    std::chrono::milliseconds timeout,
    Callback onStarted,
    Callback onTimedout) {
  impl_->drainDelQueue();
  auto st = std::make_shared<Impl::StreamTimeout>(Impl::StreamTimeout{
      std::move(startEvent),
      std::move(endEvent),
      timeout,
      std::move(onStarted),
      std::move(onTimedout)});
  impl_->enqueue([impl = impl_.get(), st]() {
    impl->streamTimeouts_.push_back(st);
    impl->activeStreamTimeouts_.store(impl->streamTimeouts_.size());
    impl->maybeStartPollTimer();
  });
}

size_t Watchdog::numActiveStreamTimeouts() const {
  return impl_->activeStreamTimeouts_.load();
}

#else // TORCH_USE_LIBUV

struct Watchdog::Impl {};

namespace {
[[noreturn]] void notAvailable() {
  TORCH_CHECK(
      false,
      "c10d::watchdog::Watchdog requires the libuv timer backend, which is "
      "only available when PyTorch is built with USE_DISTRIBUTED and "
      "USE_TENSORPIPE.");
}
} // namespace

Watchdog::Watchdog() = default;
Watchdog::~Watchdog() = default;

uint64_t Watchdog::registerTimer(std::chrono::milliseconds, Callback) {
  notAvailable();
}

void Watchdog::cancelTimer(uint64_t) {
  notAvailable();
}

void Watchdog::registerStreamTimeout(
    c10::Event,
    c10::Event,
    std::chrono::milliseconds,
    Callback,
    Callback) {
  notAvailable();
}

size_t Watchdog::numActiveStreamTimeouts() const {
  return 0;
}

#endif // TORCH_USE_LIBUV

const std::shared_ptr<Watchdog>& Watchdog::singleton() {
  // Intentionally leaked: the global watchdog owns a background thread that we
  // never want to join during interpreter/static destruction.
  static auto* instance =
      new std::shared_ptr<Watchdog>(std::make_shared<Watchdog>());
  return *instance;
}

} // namespace c10d::watchdog
