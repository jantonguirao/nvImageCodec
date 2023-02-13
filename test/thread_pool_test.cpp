/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "../src/thread_pool.h"
#include <gtest/gtest.h>
#include <atomic>

namespace nvimgcdcs { namespace test {

TEST(ThreadPool, AddWork) {
  ThreadPool tp(16, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.addWork(increase);
  }
  ASSERT_EQ(count, 0);
  tp.runAll();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, AddWorkImmediateStart) {
  ThreadPool tp(16, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto increase = [&count](int thread_id) { count++; };
  for (int i = 0; i < 64; i++) {
    tp.addWork(increase, 0, true);
  }
  tp.waitForWork();
  ASSERT_EQ(count, 64);
}

TEST(ThreadPool, AddWorkWithPriority) {
  // only one thread to ensure deterministic behavior
  ThreadPool tp(1, 0, false, "ThreadPool test");
  std::atomic<int> count{0};
  auto set_to_1 = [&count](int thread_id) {
    count = 1;
  };
  auto increase_by_1 = [&count](int thread_id) {
    count++;
  };
  auto mult_by_2 = [&count](int thread_id) {
    int val = count.load();
    while (!count.compare_exchange_weak(val, val * 2)) {}
  };
  tp.addWork(increase_by_1, 2);
  tp.addWork(mult_by_2, 7);
  tp.addWork(mult_by_2, 9);
  tp.addWork(mult_by_2, 8);
  tp.addWork(increase_by_1, 100);
  tp.addWork(set_to_1, 1000);

  tp.runAll();
  ASSERT_EQ(((1+1) << 3) + 1, count);
}


}  // namespace test

} // namespace nvimgcdcs
