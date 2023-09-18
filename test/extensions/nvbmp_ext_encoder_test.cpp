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

class NvbmpExtEncoderTest : public NvbmpExtTestBase, public TestWithParam<nvimgcdcsSampleFormat_t>
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

        sample_format_ = GetParam();
        color_spec_ = NVIMGCDCS_COLORSPEC_SRGB;
        chroma_subsampling_ = NVIMGCDCS_SAMPLING_NONE;

        image_width_ = 256;
        image_height_ = 256;
        num_components_ = 3; 
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
        ref_buffer_.resize(image_size_);

        srand(4771);
        for(unsigned int i = 0; i < image_size_; ++i) {
            ref_buffer_[i] = rand()%255;
        } 
    }

    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsEncodeParams_t params_;

    int image_width_;
    int image_height_;
    int num_components_; 
    int image_size_;
    std::vector<unsigned char> ref_buffer_;
};

TEST_P(NvbmpExtEncoderTest, ValidFormatAndParameters)
{
    // generate random image
    genRandomImage();

    image_info_.plane_info[0].width = image_width_;
    image_info_.plane_info[0].height = image_height_;
    PrepareImageForFormat();

    auto image_info_ref = image_info_;
    if (sample_format_ == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
        Convert_P_RGB_to_I_RGB(image_buffer_, ref_buffer_, image_info_);
        image_info_ref.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        image_info_ref.num_planes = image_info_.plane_info[0].num_channels;
        for (int p = 0; p < image_info_ref.num_planes; p++) {
            image_info_ref.plane_info[p].height = image_info_.plane_info[0].height;
            image_info_ref.plane_info[p].width = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].row_stride = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].num_channels = 1;
            image_info_ref.plane_info[p].sample_type = image_info_.plane_info[0].sample_type;
            image_info_ref.plane_info[p].precision = 0;
        }
        image_info_ref.buffer_size = ref_buffer_.size();
        image_info_ref.buffer = ref_buffer_.data();
        image_info_ref.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else {
        memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());
    }

    // encode the image
    nvimgcdcsImageInfo_t cs_image_info(image_info_);
    strcpy(cs_image_info.codec_name, "bmp");
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this, &NvbmpExtEncoderTest::ResizeBufferStatic<NvbmpExtEncoderTest>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));

    size_t status_size;
    nvimgcdcsProcessingStatus_t encode_status;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);

    // read the compressed image info
    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcdcsImageInfo_t load_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &load_info));

    // compare the image info with the original image info
    ASSERT_EQ(load_info.color_spec, image_info_ref.color_spec);
    ASSERT_EQ(load_info.sample_format, image_info_ref.sample_format);
    ASSERT_EQ(load_info.num_planes, image_info_ref.num_planes);
    for (int p = 0; p < load_info.num_planes; p++) {
        ASSERT_EQ(load_info.plane_info[p].width, image_info_ref.plane_info[p].width);
        ASSERT_EQ(load_info.plane_info[p].height, image_info_ref.plane_info[p].height);
        ASSERT_EQ(load_info.plane_info[p].num_channels, image_info_ref.plane_info[p].num_channels);
        ASSERT_EQ(load_info.plane_info[p].sample_type, image_info_ref.plane_info[p].sample_type);
        ASSERT_EQ(load_info.plane_info[p].precision, image_info_ref.plane_info[p].precision);
    }

    std::vector<uint8_t> decode_buffer;
    load_info.buffer_size = image_info_ref.buffer_size;
    decode_buffer.resize(load_info.buffer_size);
    load_info.buffer = decode_buffer.data();
    load_info.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;

    // decode the compressed image
    nvimgcdcsDecoder_t decoder;
    nvimgcdcsExecutionParams_t exec_params{NVIMGCDCS_STRUCTURE_TYPE_EXECUTION_PARAMS, 0};
    exec_params.device_id = NVIMGCDCS_DEVICE_CURRENT;
    exec_params.max_num_cpu_threads = 1;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder, &exec_params, nullptr));

    nvimgcdcsDecodeParams_t decode_params;
    decode_params = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &out_image_, &load_info));

    nvimgcdcsFuture_t decoder_future = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDecode(decoder, &in_code_stream_, &out_image_, 1, &decode_params, &decoder_future));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(decoder_future));
    nvimgcdcsProcessingStatus_t decode_status;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(decoder_future, &decode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, decode_status);

    // compare the decoded image with the original random image
    ASSERT_EQ(0,
        memcmp(reinterpret_cast<void*>(decode_buffer.data()), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size()));

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureDestroy(decoder_future));
}

INSTANTIATE_TEST_SUITE_P(NVBMP_ENCODE_VALID_SRGB_INPUT_FORMATS,
    NvbmpExtEncoderTest,
    Values(NVIMGCDCS_SAMPLEFORMAT_I_RGB, NVIMGCDCS_SAMPLEFORMAT_P_RGB)
);

}} // namespace nvimgcdcs::test
