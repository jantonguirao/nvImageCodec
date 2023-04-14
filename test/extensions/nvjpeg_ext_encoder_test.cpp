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
#include <tuple>
#include "nvimgcodecs_tests.h"
#include "nvjpeg_ext_test_common.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace nvimgcdcs { namespace test {

class NvJpegExtEncoderTestBase: public NvJpegExtTestBase
{
  public:
    NvJpegExtEncoderTestBase() {}

    virtual void SetUp() 
    {
        NvJpegExtTestBase::SetUp();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCreate(instance_, &encoder_));

        jpeg_enc_params_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, 0};
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, 0};
        params_.next = &jpeg_enc_params_;
        params_.quality = 95;
        params_.target_psnr = 0;
        params_.num_backends = 0; //Zero means that all backends are allowed.
        params_.mct_mode = NVIMGCDCS_MCT_MODE_YCC;
    }

    virtual void TearDown()
    {
        if (encoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderDestroy(encoder_));
        NvJpegExtTestBase::TearDown();
    }

    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsJpegEncodeParams_t jpeg_enc_params_;
    nvimgcdcsEncodeParams_t params_;
};

class NvJpegExtEncoderTestSingleImage : public NvJpegExtEncoderTestBase,
                                        public TestWithParam<std::tuple<nvimgcdcsChromaSubsampling_t, nvimgcdcsJpegEncoding_t>>
{
  public:
    virtual ~NvJpegExtEncoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        NvJpegExtEncoderTestBase::SetUp();

        image_info_.chroma_subsampling = std::get<0>(GetParam());
        jpeg_enc_params_.encoding = std::get<1>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpegExtEncoderTestBase::TearDown();
    }
};

TEST_P(NvJpegExtEncoderTestSingleImage, Encode_Single_Image_Extended_Jpeg_info)
{
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
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);

    LoadImageFromHostMemory(instance_, in_code_stream_, out_buffer_.data(), out_buffer_.size());
    nvimgcdcsImageInfo_t load_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsJpegImageInfo_t load_jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    load_info.next = &load_jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &load_info));
    EXPECT_EQ(jpeg_enc_params_.encoding, load_jpeg_info.encoding);
    EXPECT_EQ(image_info_.chroma_subsampling, load_info.chroma_subsampling);
}

static nvimgcdcsJpegEncoding_t encodings[] = {
    NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN};

static nvimgcdcsChromaSubsampling_t css[] = {NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_422 ,
    NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410,
    NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V};

    INSTANTIATE_TEST_SUITE_P(
        ENCODE_CHROMA_SUBSAMPLING_AND_ENCODING, NvJpegExtEncoderTestSingleImage, Combine(::testing::ValuesIn(css), ::testing::ValuesIn(encodings)));

}} // namespace nvimgcdcs::test