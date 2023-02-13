/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nvimgcdcs {

// arbitrary value for the CPU device ID
constexpr int CPU_ONLY_DEVICE_ID = -99999;

class ThreadPool
{
  public:
    // Basic unit of work that our threads do
    typedef std::function<void(int)> Work;

    ThreadPool(int num_thread, int device_id, bool set_affinity, const char* name);

    ThreadPool(int num_thread, int device_id, bool set_affinity, const std::string& name)
        : ThreadPool(num_thread, device_id, set_affinity, name.c_str())
    {
    }

    ~ThreadPool();

    /**
   * @brief Adds work to the queue with optional priority, and optionally starts processing
   *
   * The jobs are queued but the workers don't pick up the work unless they have
   * already been started by a previous call to AddWork with start_immediately = true or RunAll.
   * Once work is started, the threads will continue to pick up whatever work is scheduled
   * until WaitForWork is called.
   */
    void addWork(Work work, int64_t priority = 0, bool start_immediately = false);

    /**
   * @brief Wakes up all the threads to complete all the queued work,
   *        optionally not waiting for the work to be finished before return
   *        (the default wait=true is equivalent to invoking WaitForWork after RunAll).
   */
    void runAll(bool wait = true);

    /**
   * @brief Waits until all work issued to the thread pool is complete
   */
    void waitForWork(bool checkForErrors = true);

    int getThreadsNum() const;

    std::vector<std::thread::id> getThreadIds() const;

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

  private:
    void threadMain(int thread_id, int device_id, bool set_affinity, const std::string& name);

    std::vector<std::thread> threads_;

    using PrioritizedWork = std::pair<int64_t, Work>;
    struct SortByPriority
    {
        bool operator()(const PrioritizedWork& a, const PrioritizedWork& b)
        {
            return a.first < b.first;
        }
    };
    std::priority_queue<PrioritizedWork, std::vector<PrioritizedWork>, SortByPriority> work_queue_;

    bool running_;
    bool work_complete_;
    bool started_;
    int active_threads_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completed_;

    //  Stored error strings for each thread
    std::vector<std::queue<std::string>> tl_errors_;
};

} // namespace nvimgcdcs
