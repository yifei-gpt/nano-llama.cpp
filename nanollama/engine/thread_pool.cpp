// thread_pool.cpp — see thread_pool.h
#include "nanollama/engine/thread_pool.h"

namespace nano {

ThreadPool::ThreadPool(int n_threads) : n_threads_(n_threads < 1 ? 1 : n_threads) {
    for (int id = 1; id < n_threads_; id++) workers_.emplace_back([this, id] { worker(id); });
}

ThreadPool::~ThreadPool() {
    { std::lock_guard<std::mutex> lk(m_); stop_ = true; epoch_++; }
    cv_start_.notify_all();
    for (auto & w : workers_) w.join();
}

void ThreadPool::worker(int id) {
    int seen = 0;
    while (true) {
        std::unique_lock<std::mutex> lk(m_);
        cv_start_.wait(lk, [&] { return stop_ || epoch_ != seen; });
        if (stop_) return;
        seen = epoch_;
        lk.unlock();
        for (int i = id; i < n_items_; i += n_threads_) (*fn_)(i);   // fn_/n_items_ fixed for this epoch
        lk.lock();
        if (++finished_ == n_threads_ - 1) cv_done_.notify_one();
    }
}

void ThreadPool::parallel_for(int n, const std::function<void(int)> & fn) {
    if (n_threads_ <= 1 || n <= 1) { for (int i = 0; i < n; i++) fn(i); return; }
    {
        std::lock_guard<std::mutex> lk(m_);
        fn_ = &fn; n_items_ = n; finished_ = 0; epoch_++;
    }
    cv_start_.notify_all();
    for (int i = 0; i < n; i += n_threads_) fn(i);   // calling thread = worker 0
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return finished_ == n_threads_ - 1; });
}

} // namespace nano
