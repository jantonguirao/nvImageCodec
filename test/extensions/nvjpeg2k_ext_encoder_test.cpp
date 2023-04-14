/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class NvJpeg2kExtEncoderTest : public ::testing::Test
{
  public:
    NvJpeg2kExtEncoderTest() {}

    void SetUp() override
    {
        nvimgcdcsInstanceCreateInfo_t create_info;
        create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.next = nullptr;
        create_info.device_allocator = nullptr;
        create_info.pinned_allocator = nullptr;
        create_info.load_extension_modules = false;
        create_info.executor = nullptr;
        create_info.num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        nvjpeg2k_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg2k_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvjpeg2k_extension_desc(&nvjpeg2k_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvjpeg2k_extension_, &nvjpeg2k_extension_desc_));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        jpeg_info_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
        image_info_.next = &jpeg_info_;

        image_info_.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        image_info_.num_planes = 3;
        image_info_.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        image_info_.chroma_subsampling = NVIMGCDCS_SAMPLING_444;
        for (int p = 0; p < image_info_.num_planes; p++) {
            image_info_.plane_info[p].height = 320;
            image_info_.plane_info[p].width = 320;
            image_info_.plane_info[p].num_channels = 1;
            image_info_.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        }
        image_info_.buffer_size = image_info_.num_planes * image_info_.plane_info[0].height * image_info_.plane_info[0].width;

        in_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = in_buffer_.data();
        image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;

        out_buffer_.resize(image_info_.buffer_size);
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &in_image_, &image_info_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            nvimgcdcsCodeStreamCreateToHostMem(instance_, &out_code_stream_, out_buffer_.data(), out_buffer_.size(), "jpeg2k", &image_info_));

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCreate(instance_, &encoder_));
        images_.clear();
        images_.push_back(in_image_);
        streams_.clear();
        streams_.push_back(out_code_stream_);
        jpeg2k_enc_params_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, 0};
        jpeg2k_enc_params_.stream_type = NVIMGCDCS_JPEG2K_STREAM_J2K;
        jpeg2k_enc_params_.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_LRCP;
        jpeg2k_enc_params_.num_resolutions = 2;
        jpeg2k_enc_params_.code_block_w = 32;
        jpeg2k_enc_params_.code_block_h = 32;
        bool irreversible = false;
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, &jpeg2k_enc_params_, 0};
        params_.quality = 0;
        params_.target_psnr = 30;
        params_.num_backends = 0; //Zero means that all backends are allowed.
        params_.mct_mode = NVIMGCDCS_MCT_MODE_YCC;
    }

    void TearDown() override
    {
        if (future_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureDestroy(future_));
        if (encoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderDestroy(encoder_));
        if (out_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(out_code_stream_));
        if (in_image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(in_image_));
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(stream_handle_));
        if (out_stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(out_stream_handle_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvjpeg2k_extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_));
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t nvjpeg2k_extension_desc_{};
    nvimgcdcsExtension_t nvjpeg2k_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
    nvimgcdcsCodeStream_t out_stream_handle_ = nullptr;
    std::vector<unsigned char> in_buffer_;
    std::vector<unsigned char> out_buffer_;
    nvimgcdcsImageInfo_t image_info_;
    nvimgcdcsJpegImageInfo_t jpeg_info_;
    nvimgcdcsImage_t in_image_;
    nvimgcdcsCodeStream_t out_code_stream_;
    std::vector<nvimgcdcsImage_t> images_;
    std::vector<nvimgcdcsCodeStream_t> streams_;
    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsJpeg2kEncodeParams_t jpeg2k_enc_params_;
    nvimgcdcsEncodeParams_t params_;
    nvimgcdcsFuture_t future_;
};

TEST_F(NvJpeg2kExtEncoderTest, Encode_LRCP)
{
    jpeg2k_enc_params_.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_LRCP;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

TEST_F(NvJpeg2kExtEncoderTest, Encode_RLCP)
{
    jpeg2k_enc_params_.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_RLCP;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

TEST_F(NvJpeg2kExtEncoderTest, Encode_RPCL)
{
    jpeg2k_enc_params_.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_RPCL;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

TEST_F(NvJpeg2kExtEncoderTest, Encode_PCRL)
{
    jpeg2k_enc_params_.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_PCRL;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

TEST_F(NvJpeg2kExtEncoderTest, Encode_CPRL)
{
    jpeg2k_enc_params_.prog_order = NVIMGCDCS_JPEG2K_PROG_ORDER_CPRL;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

}} // namespace nvimgcdcs::test