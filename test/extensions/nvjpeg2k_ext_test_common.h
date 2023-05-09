
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

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <nvjpeg2k.h>

#include "common.h"

namespace nvimgcdcs { namespace test {

class NvJpeg2kExtTestBase :  public ExtensionTestBase
{
  public:
    virtual ~NvJpeg2kExtTestBase() = default;

    virtual void SetUp()
    {
        ExtensionTestBase::SetUp();
        nvjpeg2k_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg2k_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvjpeg2k_extension_desc(&nvjpeg2k_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvjpeg2k_extension_, &nvjpeg2k_extension_desc_));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    }

    virtual void TearDown()
    {
        TearDownCodecResources();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvjpeg2k_extension_));
        ExtensionTestBase::TearDown();
    }

    nvimgcdcsExtensionDesc_t nvjpeg2k_extension_desc_{};
    nvimgcdcsExtension_t nvjpeg2k_extension_;
};

class NvJpeg2kTestBase
{
  public:
    virtual ~NvJpeg2kTestBase() = default;

    virtual void SetUp()
    {
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kCreateSimple(&nvjpeg2k_handle_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeStateCreate(nvjpeg2k_handle_, &nvjpeg2k_decode_state_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamCreate(&nvjpeg2k_stream_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeParamsCreate(&nvjpeg2k_decode_params_));
    }

    virtual void TearDown()
    {
        if (nvjpeg2k_decode_params_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeParamsDestroy(nvjpeg2k_decode_params_));
            nvjpeg2k_decode_params_ = nullptr;
        }
        if (nvjpeg2k_stream_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamDestroy(nvjpeg2k_stream_));
            nvjpeg2k_stream_ = nullptr;
        }
        if (nvjpeg2k_decode_state_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeStateDestroy(nvjpeg2k_decode_state_));
            nvjpeg2k_decode_state_ = nullptr;
        }
        if (nvjpeg2k_handle_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDestroy(nvjpeg2k_handle_));
            nvjpeg2k_handle_ = nullptr;
        }
    };

    virtual void DecodeReference(const std::string& resources_dir, const std::string& file_name, nvimgcdcsSampleFormat_t output_format,
        bool enable_color_convert, nvimgcdcsImageInfo_t* cs_image_info = nullptr)
    {
        std::string file_path(resources_dir + '/' + file_name);
        std::ifstream input_stream(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        ASSERT_EQ(true, input_stream.is_open());
        std::streamsize file_size = input_stream.tellg();
        input_stream.seekg(0, std::ios::beg);
        std::vector<unsigned char> compressed_buffer(file_size);
        input_stream.read(reinterpret_cast<char*>(compressed_buffer.data()), file_size);
        ASSERT_EQ(true, input_stream.good());

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kStreamParse(nvjpeg2k_handle_, compressed_buffer.data(), static_cast<size_t>(file_size), 0, 0, nvjpeg2k_stream_));

        nvjpeg2kImageInfo_t image_info;
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamGetImageInfo(nvjpeg2k_stream_, &image_info));

        std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info(image_info.num_components);
        for (uint32_t c = 0; c < image_info.num_components; c++) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamGetImageComponentInfo(nvjpeg2k_stream_, &image_comp_info[c], c));
        }
        nvjpeg2kImage_t decoded_image;
        int bytes_per_element = 1;
        if (image_comp_info[0].precision > 8 && image_comp_info[0].precision <= 16) {
            bytes_per_element = 2;
            decoded_image.pixel_type = image_comp_info[0].sgn ? NVJPEG2K_UINT16 : NVJPEG2K_INT16;
        } else if (image_comp_info[0].precision == 8) {
            bytes_per_element = 1;
            decoded_image.pixel_type = NVJPEG2K_UINT8;
        } else {
            ASSERT_EQ(false, true); //Unsupported precision
        }
        if (cs_image_info) {
            cs_image_info->plane_info[0].width = image_info.image_width;
            cs_image_info->plane_info[0].height = image_info.image_height;
        }
        unsigned char* pBuffer = NULL;
        size_t buffer_size = image_info.image_width * image_info.image_height * bytes_per_element * image_info.num_components;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&pBuffer), buffer_size));

        std::vector<unsigned char*> decode_output(image_info.num_components);
        std::vector<size_t> decode_output_pitch(image_info.num_components);
        for (uint32_t c = 0; c < image_info.num_components; c++) {
            decode_output[c] = pBuffer + c * image_info.image_width * image_info.image_height * bytes_per_element;
            decode_output_pitch[c] = image_info.image_width * bytes_per_element;
        }

        decoded_image.pixel_data = (void**)decode_output.data();
        decoded_image.pitch_in_bytes = decode_output_pitch.data();
        decoded_image.num_components = image_info.num_components;

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeParamsSetRGBOutput(nvjpeg2k_decode_params_, enable_color_convert));

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kDecodeImage(nvjpeg2k_handle_, nvjpeg2k_decode_state_, nvjpeg2k_stream_, nvjpeg2k_decode_params_, &decoded_image, 0));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ref_buffer_.resize(buffer_size);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(pBuffer), buffer_size,
                                   ::cudaMemcpyDeviceToHost));

        cudaFree(pBuffer);
    }

    nvjpeg2kBackend_t backend_ = NVJPEG2K_BACKEND_DEFAULT;
    nvjpeg2kHandle_t nvjpeg2k_handle_;
    nvjpeg2kDecodeState_t nvjpeg2k_decode_state_;
    nvjpeg2kStream_t nvjpeg2k_stream_;
    nvjpeg2kDecodeParams_t nvjpeg2k_decode_params_;
    std::vector<unsigned char> ref_buffer_;
};

}} // namespace nvimgcdcs::test
