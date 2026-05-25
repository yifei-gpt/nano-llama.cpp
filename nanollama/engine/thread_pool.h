// thread_pool.h — fixed worker pool for a blocking parallel-for (avoids per-call thread spawn)
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace nano {

// parallel_for(n, fn) runs fn(0..n-1) strided across the pool and blocks until done.
// The calling thread participates as worker 0, so a pool of size N spawns N-1 threads.
struct ThreadPool {
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    void parallel_for(int n, const std::function<void(int)> & fn);

private:
    void worker(int id);

    int n_threads_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    const std::function<void(int)> * fn_ = nullptr;
    int  n_items_  = 0;
    unsigned epoch_ = 0;  // bumped per dispatch; workers wake when it changes (unsigned: defined wraparound)
    int  finished_ = 0;
    bool stop_     = false;
};

} // namespace nano
