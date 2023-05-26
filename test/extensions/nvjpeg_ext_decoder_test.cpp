/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <nvimgcodecs.h>

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <parsers/parser_test_utils.h>

#include "nvjpeg_ext_test_common.h"

#include <test_utils.h>
#include "nvimgcodecs_tests.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

#define NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP 0

namespace nvimgcdcs { namespace test {

class NvJpegExtDecoderTestBase : public NvJpegExtTestBase
{
  public:
    virtual ~NvJpegExtDecoderTestBase() = default;

    void SetUp()
    {
        NvJpegExtTestBase::SetUp();
        std::string dec_options{":fancy_upsampling=0"};
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder_, NVIMGCDCS_DEVICE_CURRENT, dec_options.c_str()));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
        params_.enable_orientation = true;
        params_.enable_color_conversion = true;
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDestroy(decoder_));
        NvJpegExtTestBase::TearDown();
    }


    nvimgcdcsDecoder_t decoder_;
    nvimgcdcsDecodeParams_t params_;
};

class NvJpegExtDecoderTestSingleImage : public NvJpegExtDecoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<const char*, nvimgcdcsColorSpec_t, nvimgcdcsSampleFormat_t,
                                            nvimgcdcsChromaSubsampling_t, nvimgcdcsSampleFormat_t, bool>>
{
  public:
    virtual ~NvJpegExtDecoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        reference_output_format_ = std::get<4>(GetParam());
        NvJpegExtDecoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
        params_.enable_color_conversion = std::get<5>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpegTestBase::TearDown();
        NvJpegExtDecoderTestBase::TearDown();
    }
};

TEST_P(NvJpegExtDecoderTestSingleImage, ValidFormatAndParameters)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + image_file_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    PrepareImageForFormat();

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &out_image_, &image_info_));
    streams_.push_back(in_code_stream_);
    images_.push_back(out_image_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDecode(decoder_, streams_.data(), images_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    cudaDeviceSynchronize();
    nvimgcdcsProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
    DecodeReference(resources_dir, image_file_, reference_output_format_, params_.enable_color_conversion);
    ConvertToPlanar();
    if (NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP) {
        write_bmp("./out.bmp", image_buffer_.data(), image_info_.plane_info[0].width,
            image_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width), image_info_.plane_info[0].width,
            image_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width) * 2, image_info_.plane_info[0].width,
            image_info_.plane_info[0].width, image_info_.plane_info[0].height);

        write_bmp("./ref.bmp", ref_buffer_.data(), image_info_.plane_info[0].width,
            ref_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width), image_info_.plane_info[0].width,
            ref_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width) * 2, image_info_.plane_info[0].width,
            image_info_.plane_info[0].width, image_info_.plane_info[0].height);
    }
    ASSERT_EQ(
        0, memcmp(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(planar_out_buffer_.data()), ref_buffer_.size()));
}

static const char* css_filenames[] = {"/jpeg/padlock-406986_640_410.jpg", "/jpeg/padlock-406986_640_411.jpg",
    "/jpeg/padlock-406986_640_420.jpg", "/jpeg/padlock-406986_640_422.jpg", "/jpeg/padlock-406986_640_440.jpg",
    "/jpeg/padlock-406986_640_444.jpg", "/jpeg/padlock-406986_640_gray.jpg"};

// clang-format off
INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_VARIOUS_CHROMA_WITH_VALID_SRGB_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
    Combine(::testing::ValuesIn(css_filenames),
        Values(NVIMGCDCS_COLORSPEC_SRGB),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB, NVIMGCDCS_SAMPLEFORMAT_I_RGB, NVIMGCDCS_SAMPLEFORMAT_P_BGR, NVIMGCDCS_SAMPLEFORMAT_I_BGR),
        //Various output chroma subsampling are ignored for SRGB 
         Values(NVIMGCDCS_SAMPLING_NONE, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V), 
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB),
        Values(true)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_VARIOUS_CHROMA_WITH_VALID_SYCC_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_SYCC),
         Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
         //Various output chroma subsampling should be ignored - there is not resampling in nvjpeg 
         Values(NVIMGCDCS_SAMPLING_NONE, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V), 
         Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
         Values(true)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_VARIOUS_CHROMA_WITH_VALID_GRAY_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_GRAY),
         Values(NVIMGCDCS_SAMPLEFORMAT_P_Y),
         //Various output chroma subsampling should be ignored - there is only luma
         Values(NVIMGCDCS_SAMPLING_NONE, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V), 
         Values(NVIMGCDCS_SAMPLEFORMAT_P_Y),
         Values(true)));

static const char* cmyk_and_ycck_filenames[] = {"/jpeg/cmyk.jpg", "/jpeg/cmyk-dali.jpg",
    "/jpeg/ycck_colorspace.jpg"};

INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_CYMK_AND_YCCK_WITH_VALID_SRGB_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
    Combine(::testing::ValuesIn(cmyk_and_ycck_filenames),
        Values(NVIMGCDCS_COLORSPEC_SRGB),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB, NVIMGCDCS_SAMPLEFORMAT_I_RGB, NVIMGCDCS_SAMPLEFORMAT_P_BGR, NVIMGCDCS_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCDCS_SAMPLING_NONE), 
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB),
        Values(true)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_CYMK_AND_YCCK_WITH_VALID_CMYK_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_CMYK, NVIMGCDCS_COLORSPEC_YCCK), //for unchanged format it should be ignored
         Values(NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED),
         //Various output chroma subsampling should be ignored - there is not resampling in nvjpeg 
         Values(NVIMGCDCS_SAMPLING_NONE, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V), 
         Values(NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED),
         Values(false)));


// clang-format on
}} // namespace nvimgcdcs::test