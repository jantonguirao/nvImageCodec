/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../src/device_guard.h"
#include "../src/exception.h"

namespace nvimgcdcs {
namespace test {

TEST(DeviceGuard, ConstructorWithDevice) {
  int test_device = 0;
  int guard_device = 0;
  int current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CUDA(cudaSetDevice(test_device));
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g(guard_device);
    CHECK_CUDA(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
  }
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
}

TEST(DeviceGuard, ConstructorNoArgs) {
  int test_device = 0;
  int guard_device = 0;
  int current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CUDA(cudaSetDevice(test_device));
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
  {
    DeviceGuard g;
    CHECK_CUDA(cudaSetDevice(guard_device));
    CHECK_CUDA(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
  }
  CHECK_CUDA(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, test_device);
}

namespace {
struct CUDAContext  {
  CUDAContext(CUcontext handle):handle_(handle){}
  inline ~CUDAContext() { DestroyHandle(handle_); }

  static CUDAContext Create(int flags, CUdevice dev) {
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    return CUDAContext(ctx);
  }

  static void DestroyHandle(CUcontext ctx) {
    CHECK_CU(cuCtxDestroy(ctx));
  }
  CUcontext handle_;
};

}  // namespace

TEST(DeviceGuard, Checkcontext) {
  int test_device = 0;
  CUdevice cu_test_device = 0;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device = 0;
  CUcontext cu_current_ctx = nullptr;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CU(cuDeviceGet(&cu_test_device, test_device));
  auto cu_test_ctx = CUDAContext::Create(0, cu_test_device);
  CHECK_CU(cuCtxSetCurrent(cu_test_ctx.handle_));
  CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx.handle_);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g(guard_device);
    CHECK_CUDA(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, guard_device);
    CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
    EXPECT_NE(cu_current_ctx, cu_test_ctx.handle_);
  }
  CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx.handle_);
  EXPECT_EQ(cu_current_device, cu_test_device);
}

TEST(DeviceGuard, CheckcontextNoArgs) {
  int test_device = 0;
  CUdevice cu_test_device = 0;
  CUcontext cu_current_ctx = nullptr;
  int guard_device = 0;
  int current_device;
  CUdevice cu_current_device = 0;
  int count = 1;

  ASSERT_TRUE(cuInitChecked());
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count > 1) {
    guard_device = 1;
  }

  CHECK_CU(cuDeviceGet(&cu_test_device, test_device));
  auto cu_test_ctx = CUDAContext::Create(0, cu_test_device);

  CHECK_CU(cuCtxSetCurrent(cu_test_ctx.handle_));
  CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx.handle_);
  EXPECT_EQ(cu_current_device, cu_test_device);
  {
    DeviceGuard g;
    CHECK_CUDA(cudaSetDevice(guard_device));
    CHECK_CUDA(cudaGetDevice(&current_device));
    CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
    EXPECT_NE(cu_current_ctx, cu_test_ctx.handle_);
  }
  CHECK_CU(cuCtxGetCurrent(&cu_current_ctx));
  CHECK_CU(cuCtxGetDevice(&cu_current_device));
  EXPECT_EQ(cu_current_ctx, cu_test_ctx.handle_);
  EXPECT_EQ(cu_current_device, cu_test_device);
}

} //namespace test
} // namespace nvimgcdcs