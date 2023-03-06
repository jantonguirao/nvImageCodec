/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <cuda_runtime.h>
#include "default_image_processor.h"
#include <cassert>
#include "exception.h"
#include "log.h"

namespace nvimgcdcs {

template <typename Out, typename In>
__global__ void nvimgcdcsImageProcessorConvert(Out* out_ptr, int64_t out_stride_y,
    int64_t out_stride_x, int64_t out_stride_c, int out_nchannels,
    nvimgcdcsColorSpec_t out_colorspace, const In* in_roi_start_ptr, int64_t roi_size_y,
    int64_t roi_size_x, int64_t in_stride_y, int64_t in_stride_x, int64_t in_stride_c,
    nvimgcdcsColorSpec_t in_colorspace, bool flip_y, bool flip_x)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (out_x >= roi_size_x || out_y >= roi_size_y)
        return;
    int in_x = flip_x ? roi_size_x - 1 - out_x : out_x;
    int in_y = flip_y ? roi_size_y - 1 - out_y : out_y;

    // if (out_colorspace == NVIMGCDCS_COLORSPEC_UNKNOWN || in_colorspace == NVIMGCDCS_COLORSPEC_UNKNOWN) {
    //     const In *in_pixel_ptr = in_roi_start_ptr + new_y * in_stride_y + new_x * in_stride_x;
    //     Out *out_pixel_ptr = out_ptr + cY * out_stride_y + cX * out_stride_x;
    //     for (int c = 0; c < out_nchannels; c++) {
    //         In in_value = *(in_pixel_ptr + c * in_stride_c);
    //         Out out_value = (Out) in_value;
    //         *(out_pixel_ptr + c * out_stride_c) = out_value;
    //     }
    // } else {
    //     Out out_pixel[3];
    //     In in_pixel[3];
    //     TODO(janton) : run color space conversion here and store in the output buffer
    // }
}

nvimgcdcsStatus_t DefaultImageProcessor::convert_gpu(
    const nvimgcdcsImageProcessorConvertParams_t* params, nvimgcdcsDeviceAllocator_t dev_allocator,
    cudaStream_t cuda_stream)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

nvimgcdcsStatus_t DefaultImageProcessor::static_convert_gpu(void* instance,
    const nvimgcdcsImageProcessorConvertParams_t* params, nvimgcdcsDeviceAllocator_t dev_allocator,
    cudaStream_t cuda_stream)
{
    return NVIMGCDCS_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

} // namespace nvimgcdcs
