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
#include "iimage_processor.h"
#include "thread_pool.h"

namespace nvimgcdcs {

class DefaultImageProcessor : public IImageProcessor
{
  public:
    DefaultImageProcessor();
    ~DefaultImageProcessor() override;
    nvimgcdcsImageProcessorDesc_t getImageProcessorDesc() override;

  private:
    // TODO(janton): define API
    nvimgcdcsStatus_t convert_cpu(const nvimgcdcsImageProcessorConvertParams_t* params,
        nvimgcdcsPinnedAllocator_t* pinned_allocator);
    nvimgcdcsStatus_t convert_gpu(const nvimgcdcsImageProcessorConvertParams_t* params,
        nvimgcdcsDeviceAllocator_t dev_allocator, cudaStream_t cuda_stream);

    static nvimgcdcsStatus_t static_convert_cpu(void* instance,
        const nvimgcdcsImageProcessorConvertParams_t* params,
        nvimgcdcsPinnedAllocator_t* pinned_allocator);
    static nvimgcdcsStatus_t static_convert_gpu(void* instance,
        const nvimgcdcsImageProcessorConvertParams_t* params,
        nvimgcdcsDeviceAllocator_t dev_allocator, cudaStream_t cuda_stream);

    nvimgcdcsImageProcessorDesc desc_;
    std::map<int, ThreadPool> device_id2thread_pool_;
};

} // namespace nvimgcdcs
