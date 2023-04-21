
/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <fstream>

#include <gtest/gtest.h>

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <nvjpeg.h>

namespace nvimgcdcs { namespace test {

class NvJpegExtTestBase
{
  public:
    virtual ~NvJpegExtTestBase() = default;

    virtual void SetUp()
    {
        nvimgcdcsInstanceCreateInfo_t create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
        create_info.num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        nvjpeg_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvjpeg_extension_desc(&nvjpeg_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvjpeg_extension_, &nvjpeg_extension_desc_));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        jpeg_info_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
        image_info_.next = &jpeg_info_;

        images_.clear();
        streams_.clear();
    }

    virtual void TearDown()
    {
        if (future_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureDestroy(future_));
        if (in_image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(in_image_));
        if (out_image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(out_image_));
        if (in_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(in_code_stream_));
        if (out_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(out_code_stream_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvjpeg_extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_));
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t nvjpeg_extension_desc_{};
    nvimgcdcsExtension_t nvjpeg_extension_;

    nvimgcdcsCodeStream_t in_code_stream_ = nullptr;
    nvimgcdcsCodeStream_t out_code_stream_ = nullptr;
    std::vector<unsigned char> in_buffer_;
    std::vector<unsigned char> out_buffer_;
    nvimgcdcsImageInfo_t image_info_;
    nvimgcdcsJpegImageInfo_t jpeg_info_;
    nvimgcdcsImage_t in_image_ = nullptr;
    nvimgcdcsImage_t out_image_ = nullptr;
    std::vector<nvimgcdcsImage_t> images_;
    std::vector<nvimgcdcsCodeStream_t> streams_;
    nvimgcdcsFuture_t future_ = nullptr;
};

constexpr bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

constexpr int format_to_num_components(nvjpegOutputFormat_t format, int num_planes)
{
    switch (format) {
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI:
    case NVJPEG_OUTPUT_BGR:
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_YUV:
        return 3;
    case NVJPEG_OUTPUT_UNCHANGED:
        return num_planes;
    case NVJPEG_OUTPUT_Y:
        return 1;
    default:
        return 3;
    }
}

class NvJpegTestBase
{
  public:
    virtual ~NvJpegTestBase() = default;

    virtual void SetUp()
    {
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegCreateSimple(&nvjpeg_handle_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecoderCreate(nvjpeg_handle_, backend_, &nvjpeg_decoder_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decode_state_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &nvjpeg_pinned_buffer_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &nvjpeg_device_buffer_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamCreate(nvjpeg_handle_, &nvjpeg_jpeg_stream_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegStateAttachPinnedBuffer(nvjpeg_decode_state_, nvjpeg_pinned_buffer_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegStateAttachDeviceBuffer(nvjpeg_decode_state_, nvjpeg_device_buffer_));
    }

    virtual void TearDown()
    {
        if (nvjpeg_decode_params_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsDestroy(nvjpeg_decode_params_));
            nvjpeg_decode_params_ = nullptr;
        }
        if (nvjpeg_jpeg_stream_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamDestroy(nvjpeg_jpeg_stream_));
            nvjpeg_jpeg_stream_ = nullptr;
        }
        if (nvjpeg_pinned_buffer_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferPinnedDestroy(nvjpeg_pinned_buffer_));
            nvjpeg_pinned_buffer_ = nullptr;
        }
        if (nvjpeg_device_buffer_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferDeviceDestroy(nvjpeg_device_buffer_));
            nvjpeg_device_buffer_ = nullptr;
        }
        if (nvjpeg_decode_state_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStateDestroy(nvjpeg_decode_state_));
            nvjpeg_decode_state_ = nullptr;
        }
        if (nvjpeg_decoder_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecoderDestroy(nvjpeg_decoder_));
            nvjpeg_decoder_ = nullptr;
        }
        if (nvjpeg_handle_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDestroy(nvjpeg_handle_));
            nvjpeg_handle_ = nullptr;
        }
    };

    virtual void decodeReference(const std::string& file_name, nvjpegOutputFormat_t output_format)
    {
        std::ifstream input_stream(file_name.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        ASSERT_EQ(true, input_stream.is_open());
        std::streamsize file_size = input_stream.tellg();
        input_stream.seekg(0, std::ios::beg);
        std::vector<unsigned char> compressed_buffer(file_size);
        input_stream.read(reinterpret_cast<char*>(compressed_buffer.data()), file_size);
        ASSERT_EQ(true, input_stream.good());
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegJpegStreamParse(nvjpeg_handle_, compressed_buffer.data(), static_cast<size_t>(file_size), 0, 0, nvjpeg_jpeg_stream_));

        unsigned int nComponent = 0;
        nvjpegChromaSubsampling_t subsampling;
        unsigned int frame_width, frame_height;
        unsigned int widths[NVJPEG_MAX_COMPONENT];
        unsigned int heights[NVJPEG_MAX_COMPONENT];

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetFrameDimensions(nvjpeg_jpeg_stream_, &frame_width, &frame_height));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetComponentsNum(nvjpeg_jpeg_stream_, &nComponent));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetChromaSubsampling(nvjpeg_jpeg_stream_, &subsampling));
        for (unsigned int i = 0; i < nComponent; i++) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetComponentDimensions(nvjpeg_jpeg_stream_, i, &widths[i], &heights[i]));
        }
        nvjpegExifOrientation_t orientation_flag = NVJPEG_ORIENTATION_UNKNOWN;
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetExifOrientation(nvjpeg_jpeg_stream_, &orientation_flag));

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsSetExifOrientation(nvjpeg_decode_params_, orientation_flag));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params_, output_format));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsSetAllowCMYK(nvjpeg_decode_params_, 1));

        if (orientation_flag >= NVJPEG_ORIENTATION_TRANSPOSE) {
            std::swap(frame_width, frame_height);
        }

        unsigned int output_format_num_components = format_to_num_components(output_format, nComponent);

        unsigned char* pBuffer = NULL;
        size_t buffer_size = frame_width * frame_height * output_format_num_components;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&pBuffer), buffer_size));
        nvjpegImage_t imgdesc = {{pBuffer, pBuffer + frame_width * frame_height, pBuffer + frame_width * frame_height * 2,
                                     pBuffer + frame_width * frame_height * 3},
            {(unsigned int)(is_interleaved(output_format) ? frame_width * 3 : frame_width), (unsigned int)frame_width,
                (unsigned int)frame_width, (unsigned int)frame_width}};

        cudaDeviceSynchronize();
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegDecodeJpegHost(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decode_state_, nvjpeg_decode_params_, nvjpeg_jpeg_stream_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decode_state_, nvjpeg_jpeg_stream_, NULL));
        nvjpegDecodeJpegDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decode_state_, &imgdesc, NULL);
        cudaDeviceSynchronize();

        ref_buffer_.resize(buffer_size);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(pBuffer), buffer_size,
                                   ::cudaMemcpyDeviceToHost));

        cudaFree(pBuffer);
    }

    nvjpegBackend_t backend_ = NVJPEG_BACKEND_DEFAULT;
    nvjpegHandle_t nvjpeg_handle_;
    nvjpegJpegDecoder_t nvjpeg_decoder_;
    nvjpegBufferPinned_t nvjpeg_pinned_buffer_;
    nvjpegBufferDevice_t nvjpeg_device_buffer_;
    nvjpegJpegState_t nvjpeg_decode_state_;
    nvjpegJpegStream_t nvjpeg_jpeg_stream_;
    nvjpegDecodeParams_t nvjpeg_decode_params_;
    nvjpegImage_t decoded_image_;
    std::vector<unsigned char> ref_buffer_;
};
}} // namespace nvimgcdcs::test
