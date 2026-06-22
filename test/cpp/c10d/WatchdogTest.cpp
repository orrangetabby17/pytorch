#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/watchdog/Watchdog.hpp>

using namespace std::chrono_literals;

namespace {

// Counts callback invocations and lets a test block until a target count is
// reached (or a timeout elapses).
class Latch {
 public:
  void notify() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++count_;
    }
    cv_.notify_all();
  }

  bool waitFor(int target, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return count_ >= target; });
  }

  int count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_{0};
};

} // namespace

TEST(WatchdogTest, available) {
  EXPECT_TRUE(c10d::watchdog::isAvailable());
}

TEST(WatchdogTest, singletonNotNull) {
  const auto& watchdog = c10d::watchdog::Watchdog::singleton();
  EXPECT_NE(watchdog, nullptr);
  // The singleton is stable across calls.
  EXPECT_EQ(watchdog, c10d::watchdog::Watchdog::singleton());
}

TEST(WatchdogTest, timerFires) {
  auto watchdog = std::make_shared<c10d::watchdog::Watchdog>();
  auto latch = std::make_shared<Latch>();

  watchdog->registerTimer(20ms, [latch] { latch->notify(); });

  EXPECT_TRUE(latch->waitFor(1, 2000ms));
  EXPECT_EQ(latch->count(), 1);
}

TEST(WatchdogTest, timerCancelPreventsFire) {
  auto watchdog = std::make_shared<c10d::watchdog::Watchdog>();
  auto latch = std::make_shared<Latch>();

  uint64_t id = watchdog->registerTimer(200ms, [latch] { latch->notify(); });
  watchdog->cancelTimer(id);

  // Wait well past the timeout; the callback must not have fired.
  EXPECT_FALSE(latch->waitFor(1, 500ms));
  EXPECT_EQ(latch->count(), 0);
}

TEST(WatchdogTest, cancelUnknownTimerIsNoop) {
  auto watchdog = std::make_shared<c10d::watchdog::Watchdog>();
  // Should not crash or throw.
  watchdog->cancelTimer(123456);
}

TEST(WatchdogTest, multipleTimersFire) {
  auto watchdog = std::make_shared<c10d::watchdog::Watchdog>();
  auto latch = std::make_shared<Latch>();

  constexpr int kNumTimers = 8;
  for (int i = 0; i < kNumTimers; ++i) {
    watchdog->registerTimer(20ms, [latch] { latch->notify(); });
  }

  EXPECT_TRUE(latch->waitFor(kNumTimers, 2000ms));
  EXPECT_EQ(latch->count(), kNumTimers);
}

TEST(WatchdogTest, destroyWithPendingTimerDoesNotFire) {
  // A timer that outlives its watchdog must be cancelled cleanly on shutdown
  // and must not invoke its callback.
  auto latch = std::make_shared<Latch>();
  {
    auto watchdog = std::make_shared<c10d::watchdog::Watchdog>();
    watchdog->registerTimer(60s, [latch] { latch->notify(); });
    // Give the loop a moment to actually arm the timer before teardown.
    std::this_thread::sleep_for(50ms);
  }
  EXPECT_EQ(latch->count(), 0);
}

TEST(WatchdogTest, noActiveStreamTimeouts) {
  auto watchdog = std::make_shared<c10d::watchdog::Watchdog>();
  EXPECT_EQ(watchdog->numActiveStreamTimeouts(), 0u);
}
