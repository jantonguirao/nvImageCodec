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

#include <nvimgcodec.h>
#include <map>
#include "iexecutor.h"
#include "thread_pool.h"

namespace nvimgcodec {

 class ILogger; 

class DefaultExecutor : public IExecutor
{
  public:
    explicit DefaultExecutor(ILogger* logger, int num_threads);
    ~DefaultExecutor() override;
    nvimgcodecExecutorDesc_t* getExecutorDesc() override;

  private:
    nvimgcodecStatus_t launch(int device_id, int sample_idx, void* task_context,
        void (*task)(int thread_id, int sample_idx, void* task_context));
    int get_num_threads() const;

    static nvimgcodecStatus_t static_launch(
        void* instance, int device_id, int sample_idx, void* task_context,
        void (*task)(int thread_id, int sample_idx, void* task_context));
    static int static_get_num_threads(void* instance);

    ILogger* logger_;
    nvimgcodecExecutorDesc_t desc_;
    int num_threads_;
    std::map<int, ThreadPool> device_id2thread_pool_;
};

} // namespace nvimgcodec