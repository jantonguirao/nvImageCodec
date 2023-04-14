/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class NvJpegExtEncoderTest : public ::testing::Test
{
  public:
    NvJpegExtEncoderTest() {}

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

        nvjpeg_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvjpeg_extension_desc(&nvjpeg_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvjpeg_extension_, &nvjpeg_extension_desc_));

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
            nvimgcdcsCodeStreamCreateToHostMem(instance_, &out_code_stream_, out_buffer_.data(), out_buffer_.size(), "jpeg", &image_info_));

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCreate(instance_, &encoder_));
        images_.clear();
        images_.push_back(in_image_);
        streams_.clear();
        streams_.push_back(out_code_stream_);
        jpeg_enc_params_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, 0};
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, &jpeg_enc_params_, 0};
        params_.quality = 95;
        params_.target_psnr = 0;
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
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvjpeg_extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_));
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t nvjpeg_extension_desc_{};
    nvimgcdcsExtension_t nvjpeg_extension_;
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
    nvimgcdcsJpegEncodeParams_t jpeg_enc_params_;
    nvimgcdcsEncodeParams_t params_;
    nvimgcdcsFuture_t future_;
};

TEST_F(NvJpegExtEncoderTest, Encode_CSS_444_Extended_Jpeg_info_Baseline_DCT)
{
    jpeg_enc_params_.encoding = NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));

    LoadImageFromHostMemory(instance_, stream_handle_, out_buffer_.data(), out_buffer_.size());
    nvimgcdcsImageInfo_t load_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsJpegImageInfo_t load_jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    load_info.next = &load_jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &load_info));
    EXPECT_EQ(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, load_jpeg_info.encoding);
}

TEST_F(NvJpegExtEncoderTest, Encode_CSS_444_Extended_Jpeg_info_Progressive_DCT)
{
    jpeg_enc_params_.encoding = NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);

    LoadImageFromHostMemory(instance_, stream_handle_, out_buffer_.data(), out_buffer_.size());
    nvimgcdcsImageInfo_t load_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsJpegImageInfo_t load_jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    load_info.next = &load_jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &load_info));
    EXPECT_EQ(NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, load_jpeg_info.encoding);
}

}} // namespace nvimgcdcs::test