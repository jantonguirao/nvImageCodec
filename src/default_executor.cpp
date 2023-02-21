/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "default_executor.h"
#include <cassert>
#include "exception.h"
#include "log.h"

namespace nvimgcdcs {

DefaultExecutor::DefaultExecutor(int num_threads)
    : desc_{NVIMGCDCS_STRUCTURE_TYPE_EXECUTOR_DESC, nullptr, this, &static_launch}
    , num_threads_(num_threads)
{
}

DefaultExecutor::~DefaultExecutor()
{
}

nvimgcdcsExecutorDesc_t DefaultExecutor::getExecutorDesc()
{
    return &desc_;
}

nvimgcdcsStatus_t DefaultExecutor::launch(
    int device_id, void* task_context, void (*task)(void* task_context))
{
    try {
        std::stringstream ss;
        ss << "Executor-" << device_id;
        auto it =
            device_id2thread_pool_.try_emplace(device_id, num_threads_, device_id, false, ss.str());

        auto& thread_pool = it.first->second;
        auto task_wrapper = [task_context, task](int thread_id) { task(task_context); };
        thread_pool.addWork(task_wrapper, 0, true);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DefaultExecutor::static_launch(
    void* instance, int device_id, void* task_context, void (*task)(void* task_context))
{
    DefaultExecutor* handle = reinterpret_cast<DefaultExecutor*>(instance);
    return handle->launch(device_id, task_context, task);
}
} // namespace nvimgcdcs