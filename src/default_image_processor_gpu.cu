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

// template <typename Out, typename In, class LaunchParams>
// __global__ __launch_bounds__(LaunchParams::BLOCKTHREADS)
// void nvimgcdcsImageProcessorConvert(
//                 Out *out_ptr, int64_t out_stride_y, int64_t out_stride_x,
//                 const In* in_roi_ptr, int64_t roi_size_y, int64_t roi_size_x,
//                 int64_t in_stride_y, int64_t in_stride_x,
//                 bool flip_y, bool flip_x)
// {
//     uchar3 src_pix;
//     uchar3 dst_pix;
//     int cX = threadIdx.x + blockIdx.x * LaunchParams::BLOCKDIMX;
//     int cY = threadIdx.y + blockIdx.y * LaunchParams::BLOCKDIMY;

//     int new_x = cX;
//     int new_y = cY;

//     if (rotate_params.horizontal_flip)
//     {
//         new_x = dims.width - new_x - 1;
//     }

//     if (rotate_params.vertical_flip)
//     {
//         new_y = dims.height - new_y - 1;
//     }

//     // Rotate
//     int temp_x = new_x;
//     int temp_y = new_y;
//     new_x = (temp_x * rotate_params.cos_term) - (temp_y * rotate_params.sin_term);
//     new_y = (temp_x * rotate_params.sin_term) + (temp_y * rotate_params.cos_term);

//     // Shifting
//     new_x += rotate_params.x_shift;
//     new_y += rotate_params.y_shift;

//     if (cX < dims.width)
//     {
//         if (cY < dims.height)
//         {
//             src_pix = read_pixel_yuv<ss, fancy_upsampling>(cX + roi_offset.width, cY + roi_offset.height, src, dims.width + roi_offset.width, dims.height + roi_offset.height);
//             dst_pix = convert_pixel< COLORSPACE_YCBCR, OutputFormatTraits<output_format>::colorspace >(src_pix);
//             if (output_format == NVJPEG_OUTPUT_BGR || output_format == NVJPEG_OUTPUT_BGRI)
//             {
//                 unsigned char t = dst_pix.x;
//                 dst_pix.x = dst_pix.z;
//                 dst_pix.z = t;
//             }

//             write_pixel_format<OutputFormatTraits<output_format>::is_interleaved, 
//                         OutputFormatTraits<output_format>::num_components>(
//                             dst_pix, new_x , new_y, dst);
//         }
//     }
// }

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
