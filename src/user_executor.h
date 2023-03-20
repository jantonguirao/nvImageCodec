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

#include <nvimgcodecs.h>
#include <map>
#include "iexecutor.h"
#include "thread_pool.h"

namespace nvimgcdcs {

class UserExecutor : public IExecutor
{
  public:
    explicit UserExecutor(nvimgcdcsExecutorDesc_t executor_desc) : desc_(executor_desc) {}
    ~UserExecutor() override = default;
    nvimgcdcsExecutorDesc_t getExecutorDesc() override { return desc_; }

  private:
    nvimgcdcsExecutorDesc_t desc_;
};

} // namespace nvimgcdcs