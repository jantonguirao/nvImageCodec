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
#include "convert.h"
#include <cassert>
#include "exception.h"
#include "log.h"

namespace nvimgcdcs {

template <typename Out, typename In>
__device__ void identity(
    Out* out_pixel_start, int64_t out_stride_c, const In* in_pixel_start, int64_t in_stride_c, int nchannels)
{
    for (int c = 0; c < nchannels; c++)
        *(out_pixel_start + c * out_stride_c) =
            ConvertSatNorm<Out>(*(in_pixel_start + c * in_stride_c));
}

template <typename Out, typename In>
__device__ void rgb2bgr(
    Out* out_pixel_start, int64_t out_stride_c, const In* in_pixel_start, int64_t in_stride_c)
{
    *(out_pixel_start + 0 * out_stride_c) =
        ConvertSatNorm<Out>(*(in_pixel_start + 2 * in_stride_c));
    *(out_pixel_start + 1 * out_stride_c) =
        ConvertSatNorm<Out>(*(in_pixel_start + 1 * in_stride_c));
    *(out_pixel_start + 2 * out_stride_c) =
        ConvertSatNorm<Out>(*(in_pixel_start + 0 * in_stride_c));
}

__host__ __device__ bool is_planar(nvimgcdcsSampleFormat_t sample_format)
{
    return sample_format == NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED ||
           sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB ||
           sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR ||
           sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y ||
           sample_format == NVIMGCDCS_SAMPLEFORMAT_P_YUV;
}

__host__ __device__ bool is_same_colorspace(
    nvimgcdcsSampleFormat_t out_format, nvimgcdcsSampleFormat_t in_format)
{
    return out_format == in_format || out_format == NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED ||
           (out_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB &&
               in_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) ||
           (out_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR &&
               in_format == NVIMGCDCS_SAMPLEFORMAT_I_BGR) ||
           (out_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB &&
               in_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) ||
           (out_format == NVIMGCDCS_SAMPLEFORMAT_I_BGR &&
               in_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR);
}

template <typename Out, typename In>
__global__ void nvimgcdcsImageProcessorConvert(Out* out_ptr, int64_t out_stride_y,
    int64_t out_stride_x, int64_t out_stride_c, int out_nchannels,
    nvimgcdcsSampleFormat_t out_sample_format, const In* in_roi_start_ptr, int64_t roi_size_y,
    int64_t roi_size_x, int64_t in_stride_y, int64_t in_stride_x, int64_t in_stride_c,
    nvimgcdcsSampleFormat_t in_sample_format, bool flip_y, bool flip_x)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (out_x >= roi_size_x || out_y >= roi_size_y)
        return;
    int in_x = flip_x ? roi_size_x - 1 - out_x : out_x;
    int in_y = flip_y ? roi_size_y - 1 - out_y : out_y;

    const In *in_pixel_ptr = in_roi_start_ptr + in_y * in_stride_y + in_x * in_stride_x;
    Out *out_pixel_ptr = out_ptr + out_x * out_stride_y + out_x * out_stride_x;

    if (is_same_colorspace(out_sample_format, in_sample_format)) {
        identity(out_pixel_ptr, out_stride_c, in_pixel_ptr, in_stride_c, out_nchannels);
    } else {
        assert(false);  // TODO(janton): implement
    }
}

template <typename Out, typename In>
nvimgcdcsStatus_t ConvertGPUImpl(const nvimgcdcsImageProcessorConvertParams_t* params,
    nvimgcdcsDeviceAllocator_t dev_allocator, cudaStream_t cuda_stream)
{
    int64_t region_size_y = params->image_info.height;
    int64_t region_size_x = params->image_info.width;
    if (params->image_info.region.ndim > 0) {
        int64_t roi_start_y = params->image_info.region.start[0];
        int64_t roi_start_x = params->image_info.region.start[1];
        int64_t roi_end_y = params->image_info.region.end[0];
        int64_t roi_end_x = params->image_info.region.end[1];
        region_size_y = roi_end_y - roi_start_y;
        region_size_x = roi_end_x - roi_start_x;
    }

    int64_t out_stride_y, out_stride_x, out_stride_c;
    int num_channels = is_planar(params->out_sample_format)?params->image_info.num_planes
                                                           :params->image_info.plane_info[0].num_channels;

    if (is_planar(params->out_sample_format)) {
        out_stride_c = region_size_y * region_size_x;
        out_stride_y = region_size_x;
        out_stride_x = 1;
    } else {  // interleaved
        out_stride_y = region_size_x * num_channels;
        out_stride_x = num_channels;
        out_stride_c = 1;
    }

    int64_t in_stride_y, in_stride_x, in_stride_c;
    if (is_planar(params->image_info.sample_format)) {
        in_stride_c = params->image_info.height * params->image_info.width;
        in_stride_y = params->image_info.width;
        in_stride_x = 1;
    } else {  // interleaved
        in_stride_y = params->image_info.width * num_channels;
        in_stride_x = num_channels;
        in_stride_c = 1;
    }

    Out* out_ptr = static_cast<Out*>(params->out_buffer);
    const In* in_ptr = static_cast<In*>(params->in_buffer);
    if (params->image_info.region.ndim > 0) {
        int64_t roi_start_y = params->image_info.region.start[0];
        int64_t roi_start_x = params->image_info.region.start[1];
        in_ptr += roi_start_y * in_stride_y;
        in_ptr += roi_start_x * in_stride_x;
    }

    dim3 block(32, 32);
    dim3 grid((region_size_y + 31) / 32, (region_size_x + 31) / 32);

    // TODO(janton): support colorspace conversion
    if (!is_same_colorspace(params->out_sample_format, params->image_info.sample_format))
        return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;

    nvimgcdcsImageProcessorConvert<Out, In><<<grid, block, 0, cuda_stream>>>(out_ptr, out_stride_y,
        out_stride_x, out_stride_c, num_channels, params->out_sample_format, in_ptr, in_stride_y,
        in_stride_x, in_stride_c, region_size_y, region_size_x, params->image_info.sample_format,
        params->flip_y, params->flip_x);
    CHECK_CUDA(cudaGetLastError());
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DefaultImageProcessor::convert_gpu(
    const nvimgcdcsImageProcessorConvertParams_t* params, nvimgcdcsDeviceAllocator_t dev_allocator,
    cudaStream_t cuda_stream)
{
    if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 &&
        params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8)
        return ConvertGPUImpl<uint8_t, uint8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)
        return ConvertGPUImpl<uint8_t, uint16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8)
        return ConvertGPUImpl<uint8_t, int8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16)
        return ConvertGPUImpl<uint8_t, int16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32)
        return ConvertGPUImpl<uint8_t, float>(params, dev_allocator, cuda_stream);

    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8)
        return ConvertGPUImpl<uint16_t, uint8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)
        return ConvertGPUImpl<uint16_t, uint16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8)
        return ConvertGPUImpl<uint16_t, int8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16)
        return ConvertGPUImpl<uint16_t, int16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32)
        return ConvertGPUImpl<uint16_t, float>(params, dev_allocator, cuda_stream);

    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8)
        return ConvertGPUImpl<int8_t, uint8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)
        return ConvertGPUImpl<int8_t, uint16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8)
        return ConvertGPUImpl<int8_t, int8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16)
        return ConvertGPUImpl<int8_t, int16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32)
        return ConvertGPUImpl<int8_t, float>(params, dev_allocator, cuda_stream);

    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8)
        return ConvertGPUImpl<int16_t, uint8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)
        return ConvertGPUImpl<int16_t, uint16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8)
        return ConvertGPUImpl<int16_t, int8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16)
        return ConvertGPUImpl<int16_t, int16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32)
        return ConvertGPUImpl<int16_t, float>(params, dev_allocator, cuda_stream);

    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8)
        return ConvertGPUImpl<float, uint8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)
        return ConvertGPUImpl<float, uint16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8)
        return ConvertGPUImpl<float, int8_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16)
        return ConvertGPUImpl<float, int16_t>(params, dev_allocator, cuda_stream);
    else if (params->out_sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32 &&
             params->image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32)
        return ConvertGPUImpl<float, float>(params, dev_allocator, cuda_stream);
    else
        return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcdcsStatus_t DefaultImageProcessor::static_convert_gpu(void* instance,
    const nvimgcdcsImageProcessorConvertParams_t* params, nvimgcdcsDeviceAllocator_t dev_allocator,
    cudaStream_t cuda_stream)
{
    assert(instance);
    DefaultImageProcessor* processor = reinterpret_cast<DefaultImageProcessor*>(instance);
    return processor->convert_gpu(params, dev_allocator, cuda_stream);
}

} // namespace nvimgcdcs
