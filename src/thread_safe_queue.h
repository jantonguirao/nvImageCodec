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

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace nvimgcdcs {

template <typename T>
class ThreadSafeQueue
{
  public:
    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(lock_);
            queue_.push(std::move(item));
        }
        cond_.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock{lock_};
        cond_.wait(lock, [&]() { return !queue_.empty() || interrupt_; });
        if (interrupt_) {
            return {};
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    const T& peek()
    {
        static const auto int_return = T{};
        std::unique_lock<std::mutex> lock{lock_};
        cond_.wait(lock, [&]() { return !queue_.empty() || interrupt_; });
        if (interrupt_) {
            return int_return;
        }
        return queue_.front();
    }

    bool empty() const { return queue_.empty(); }

    typename std::queue<T>::size_type size() const { return queue_.size(); }

    void shutdown()
    {
        interrupt_ = true;
        cond_.notify_all();
    }

  private:
    std::queue<T> queue_;
    std::mutex lock_;
    std::condition_variable cond_;
    bool interrupt_ = false;
};

} // namespace nvimgcdcs
