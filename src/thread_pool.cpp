/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cstdlib>
#include <utility>
#include "thread_pool.h"

#include "device_guard.h"
#include "log.h"

namespace nvimgcdcs {

ThreadPool::ThreadPool(int num_thread, int device_id, bool set_affinity, const char* name)
    : threads_(num_thread), running_(true), work_complete_(true), started_(false)
    , active_threads_(0) {
  if (num_thread) {
    NVIMGCDCS_LOG_FATAL("Thread pool must have non-zero size");
  }
#if NVML_ENABLED
  // only for the CPU pipeline
  if (device_id != CPU_ONLY_DEVICE_ID) {
    nvml::Init();
  }
#endif
  // Start the threads in the main loop
  for (int i = 0; i < num_thread; ++i) {
    std::stringstream ss;
    ss << "[NVIMGCODECS][TP"<< i<< "]"<< name;
    threads_[i] =
        std::thread(std::bind(&ThreadPool::threadMain, this, i, device_id, set_affinity, ss.str()));
  }
  tl_errors_.resize(num_thread);
}

ThreadPool::~ThreadPool() {
 waitForWork(false);

  std::unique_lock<std::mutex> lock(mutex_);
  running_ = false;
  condition_.notify_all();
  lock.unlock();

  for (auto &thread : threads_) {
    thread.join();
  }
#if NVML_ENABLED
  nvml::Shutdown();
#endif
}

void ThreadPool::addWork(Work work, int64_t priority, bool start_immediately) {
  bool started_before = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    work_queue_.push({priority, std::move(work)});
    work_complete_ = false;
    started_before = started_;
    started_ |= start_immediately;
  }
  if (started_) {
    if (!started_before)
      condition_.notify_all();
    else
      condition_.notify_one();
  }
}

// Blocks until all work issued to the thread pool is complete
void ThreadPool::waitForWork(bool checkForErrors) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_.wait(lock, [this] { return this->work_complete_; });
  started_ = false;
  if (checkForErrors) {
    // Check for errors
    for (size_t i = 0; i < threads_.size(); ++i) {
      if (!tl_errors_[i].empty()) {
        // Throw the first error that occurred
        std::stringstream ss;
        ss << "Error in thread " << i << ": " << tl_errors_[i].front();
        std::string error = ss.str();
        tl_errors_[i].pop();
        throw std::runtime_error(error);
      }
    }
  }
}

void ThreadPool::runAll(bool wait) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = true;
  }
  condition_.notify_all();  // other threads will be waken up if needed
  if (wait) {
    waitForWork();
  }
}

int ThreadPool::getThreadsNum() const
{
  return threads_.size();
}

std::vector<std::thread::id> ThreadPool::getThreadIds() const {
  std::vector<std::thread::id> tids;
  tids.reserve(threads_.size());
  for (const auto &thread : threads_)
    tids.emplace_back(thread.get_id());
  return tids;
}


void ThreadPool::threadMain(int thread_id, int device_id, bool set_affinity,
                            const std::string &name) {
  //setThreadName(name.c_str());
  DeviceGuard g(device_id);
  try {
#if NVML_ENABLED
    if (set_affinity) {
      const char *env_affinity = std::getenv("NVIMGCODECS_AFFINITY_MASK");
      int core = -1;
      if (env_affinity) {
        const auto &vec = string_split(env_affinity, ',');
        if ((size_t)thread_id < vec.size()) {
          core = std::stoi(vec[thread_id]);
        } else {
          NVIMGCDCS_LOG_WARNING(
              "NVIMGCODECS environment variable is set, "
              "but does not have enough entries: thread_id (",
              thread_id, ") vs #entries (", vec.size(), "). Ignoring...");
        }
      }
      nvml::SetCPUAffinity(core);
    }
#endif
  } catch (std::exception &e) {
    tl_errors_[thread_id].push(e.what());
  } catch (...) {
    tl_errors_[thread_id].push("Caught unknown exception");
  }

  while (running_) {
    // Block on the condition to wait for work
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !running_ || (!work_queue_.empty() && started_); });
    // If we're no longer running, exit the run loop
    if (!running_) break;

    // Get work from the queue & mark
    // this thread as active
    Work work = std::move(work_queue_.top().second);
    work_queue_.pop();
    ++active_threads_;

    // Unlock the lock
    lock.unlock();

    // If an error occurs, we save it in tl_errors_. When
    // WaitForWork is called, we will check for any errors
    // in the threads and return an error if one occured.
    try {
      work(thread_id);
    } catch (std::exception &e) {
      lock.lock();
      tl_errors_[thread_id].push(e.what());
      lock.unlock();
    } catch (...) {
      lock.lock();
      tl_errors_[thread_id].push("Caught unknown exception");
      lock.unlock();
    }

    // Mark this thread as idle & check for complete work
    lock.lock();
    --active_threads_;
    if (work_queue_.empty() && active_threads_ == 0) {
      work_complete_ = true;
      lock.unlock();
      completed_.notify_one();
    }
  }
}

} // namespace nvimgcdcs