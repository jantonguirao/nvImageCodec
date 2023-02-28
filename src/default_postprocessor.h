/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "ipostprocessor.h"
#include "thread_pool.h"

namespace nvimgcdcs {

class DefaultPostprocessor : public IPostprocessor
{
  public:
    DefaultPostprocessor();
    ~DefaultPostprocessor() override;
    nvimgcdcsPostprocessorDesc_t getPostprocessorDesc() override;

  private:
    // TODO(janton): define API
    nvimgcdcsStatus_t convert_cpu(int sample_idx);
    nvimgcdcsStatus_t convert_gpu(int device_id, int sample_idx);

    static nvimgcdcsStatus_t static_convert_cpu(
        void* instance, int device_id, int sample_idx);
    static nvimgcdcsStatus_t static_convert_gpu(
        void* instance, int device_id, int sample_idx);

    nvimgcdcsPostprocessorDesc desc_;
    std::map<int, ThreadPool> device_id2thread_pool_;
};

} // namespace nvimgcdcs