/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "default_postprocessor.h"
#include <cassert>
#include "exception.h"
#include "log.h"

namespace nvimgcdcs {

DefaultPostprocessor::DefaultPostprocessor()
    : desc_{NVIMGCDCS_STRUCTURE_TYPE_POSTPROCESSOR_DESC, nullptr, this, &static_convert_cpu, &static_convert_gpu}
{
}

DefaultPostprocessor::~DefaultPostprocessor()
{
}

nvimgcdcsPostprocessorDesc_t DefaultPostprocessor::getPostprocessorDesc()
{
    return &desc_;
}

nvimgcdcsStatus_t DefaultPostprocessor::convert_cpu(int sample_idx)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvimgcdcsStatus_t DefaultPostprocessor::convert_gpu(int device_id, int sample_idx)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvimgcdcsStatus_t DefaultPostprocessor::static_convert_cpu(
    void* instance, int device_id, int sample_idx)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvimgcdcsStatus_t DefaultPostprocessor::static_convert_gpu(
    void* instance, int device_id, int sample_idx)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

} // namespace nvimgcdcs