/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "default_image_processor.h"
#include <cassert>
#include "exception.h"
#include "log.h"

namespace nvimgcdcs {

DefaultImageProcessor::DefaultImageProcessor()
    : desc_{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_PROCESSOR_DESC, nullptr, this, &static_convert_cpu, &static_convert_gpu}
{
}

DefaultImageProcessor::~DefaultImageProcessor()
{
}

nvimgcdcsImageProcessorDesc_t DefaultImageProcessor::getImageProcessorDesc()
{
    return &desc_;
}

nvimgcdcsStatus_t DefaultImageProcessor::convert_cpu(
    const nvimgcdcsImageProcessorConvertParams_t* params,
    nvimgcdcsPinnedAllocator_t* pinned_allocator)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvimgcdcsStatus_t DefaultImageProcessor::static_convert_cpu(void* instance,
    const nvimgcdcsImageProcessorConvertParams_t* params,
    nvimgcdcsPinnedAllocator_t* pinned_allocator)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

} // namespace nvimgcdcs
