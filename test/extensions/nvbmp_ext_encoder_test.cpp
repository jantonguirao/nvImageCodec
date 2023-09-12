/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <extensions/nvbmp/nvbmp_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "nvimgcodecs_tests.h"
#include "common.h"
#include "parsers/bmp.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

namespace nvimgcdcs { namespace test {

class NvbmpExtTestBase : public ExtensionTestBase
{
  public:
    virtual ~NvbmpExtTestBase() = default;

    virtual void SetUp()
    {
        ExtensionTestBase::SetUp();

        nvbmp_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvbmp_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvbmp_extension_desc(&nvbmp_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvbmp_extension_, &nvbmp_extension_desc_));

        nvbmp_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvbmp_parser_extension_desc.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_bmp_parser_extension_desc(&nvbmp_parser_extension_desc));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvbmp_parser_extension_, &nvbmp_parser_extension_desc));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    }

    virtual void TearDown()
    {
        ExtensionTestBase::TearDownCodecResources();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvbmp_parser_extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvbmp_extension_));
        ExtensionTestBase::TearDown();
    }

    nvimgcdcsExtensionDesc_t nvbmp_extension_desc_{};
    nvimgcdcsExtension_t nvbmp_extension_;
    nvimgcdcsExtensionDesc_t nvbmp_parser_extension_desc{};
    nvimgcdcsExtension_t nvbmp_parser_extension_;
};

class NvbmpExtEncoderTest :  public ::testing::Test, public NvbmpExtTestBase
{
  public:
    NvbmpExtEncoderTest() {}

    void SetUp() override
    {
        NvbmpExtTestBase::SetUp();

        const char* options = nullptr;
        nvimgcdcsExecutionParams_t exec_params{NVIMGCDCS_STRUCTURE_TYPE_EXECUTION_PARAMS, 0};
        exec_params.device_id = NVIMGCDCS_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCreate(instance_, &encoder_, &exec_params, options));

        params_ = {NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, 0};
        params_.quality = 0;
        params_.target_psnr = 0;

        color_spec_ = NVIMGCDCS_COLORSPEC_SRGB;
        sample_format_ = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        chroma_subsampling_ = NVIMGCDCS_SAMPLING_NONE;

        image_width_ = 256;
        image_height_ = 256;
        num_components_ = 3; 
	random_image_ = nullptr;
        image_size_ = image_width_ * image_height_ * num_components_;
    }

    void TearDown() override
    {
        if (encoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderDestroy(encoder_));

        NvbmpExtTestBase::TearDown();
    }

    void genRandomImage()
    {
        random_image_ = (unsigned char *)malloc(image_size_);

        srand(4771);
        for(unsigned int i = 0; i < image_size_; ++i) {
            random_image_[i] = rand()%255;
        } 
    }

    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsEncodeParams_t params_;

    int image_width_;
    int image_height_;
    int num_components_; 
    int image_size_;
    unsigned char *random_image_;
};

TEST_F(NvbmpExtEncoderTest, NVBMP_RandomImage_RGB_P)
{
    genRandomImage();

    image_info_.plane_info[0].width = image_width_;
    image_info_.plane_info[0].height = image_height_;
    PrepareImageForFormat();

    memcpy(image_buffer_.data(), reinterpret_cast<void*>(random_image_), image_size_);

    nvimgcdcsImageInfo_t cs_image_info(image_info_);
    strcpy(cs_image_info.codec_name, "bmp");

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
                                            &NvbmpExtEncoderTest::ResizeBufferStatic < NvbmpExtEncoderTest>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));

    size_t status_size;
    nvimgcdcsProcessingStatus_t encode_status;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);

    ASSERT_GT(code_stream_buffer_.size(), 0);
}

}} // namespace nvimgcdcs::test
