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
#include <tuple>

#include <gtest/gtest.h>

#include <nvimgcodecs.h>

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <parsers/parser_test_utils.h>

#include "nvjpeg2k_ext_test_common.h"

#include <test_utils.h>
#include "nvimgcodecs_tests.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

#define NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP 0

namespace nvimgcdcs { namespace test {

class NvJpeg2kExtDecoderTestBase : public NvJpeg2kExtTestBase
{
  public:
    virtual ~NvJpeg2kExtDecoderTestBase() = default;

    void SetUp()
    {
        NvJpeg2kExtTestBase::SetUp();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder_, NVIMGCDCS_DEVICE_CURRENT, 0, nullptr, nullptr));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
        params_.enable_color_conversion = true;
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDestroy(decoder_));
        NvJpeg2kExtTestBase::TearDown();
    }

    nvimgcdcsDecoder_t decoder_;
    nvimgcdcsDecodeParams_t params_;
    std::string image_file_;
};

class NvJpeg2kExtDecoderTestSingleImage : public NvJpeg2kExtDecoderTestBase,
                                          public NvJpeg2kTestBase,
                                          public TestWithParam<std::tuple<const char*, nvimgcdcsColorSpec_t, nvimgcdcsSampleFormat_t,
                                              nvimgcdcsChromaSubsampling_t, nvimgcdcsSampleFormat_t, bool>>
{
  public:
    virtual ~NvJpeg2kExtDecoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        reference_output_format_ = std::get<4>(GetParam());
        NvJpeg2kExtDecoderTestBase::SetUp();
        NvJpeg2kTestBase::SetUp();
        params_.enable_color_conversion = std::get<5>(GetParam());
    }

    void TearDown() override
    {
        NvJpeg2kTestBase::TearDown();
        NvJpeg2kExtDecoderTestBase::TearDown();
    }

};

TEST_P(NvJpeg2kExtDecoderTestSingleImage, ValidFormatAndParameters)
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
    ASSERT_EQ(image_buffer_.size(), ref_buffer_.size());
    ASSERT_EQ(0, memcmp(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(image_buffer_.data()), image_buffer_.size()));
}

static const char* css_filenames[] = {"/jpeg2k/chroma_420/artificial_420_8b3c_dwt97CPRL.jp2",
    "/jpeg2k/chroma_420/cathedral_420_8b3c_dwt53RLCP.jp2", "/jpeg2k/chroma_420/deer_420_8b3c_dwt97RPCL.jp2",
    "/jpeg2k/chroma_420/leavesISO200_420_8b3c_dwt53PCRL.jp2", "/jpeg2k/chroma_420/leavesISO200_420_8b3c_dwt97CPRL.j2k",

    "/jpeg2k/chroma_422/artificial_422_8b3c_dwt53PCRL.jp2", "/jpeg2k/chroma_422/cathedral_422_8b3c_dwt97CPRL.jp2",
    "/jpeg2k/chroma_422/deer_422_8b3c_dwt53RPCL.j2k", "/jpeg2k/chroma_422/deer_422_8b3c_dwt53RPCL.jp2",
    "/jpeg2k/chroma_422/leavesISO200_422_8b3c_dwt97PCRL.jp2"};

// clang-format off
INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_VARIOUS_CHROMA_WITH_VALID_SRGB_OUTPUT_FORMATS, NvJpeg2kExtDecoderTestSingleImage,
    Combine(::testing::ValuesIn(css_filenames),
        Values(NVIMGCDCS_COLORSPEC_SRGB),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB),//NVIMGCDCS_SAMPLEFORMAT_I_RGB, NVIMGCDCS_SAMPLEFORMAT_P_BGR, NVIMGCDCS_SAMPLEFORMAT_I_BGR),
        //Various output chroma subsampling are ignored for SRGB 
         Values(NVIMGCDCS_SAMPLING_NONE, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V), 
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB),
        Values(true)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_VARIOUS_CHROMA_WITH_VALID_SYCC_OUTPUT_FORMATS, NvJpeg2kExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_SYCC),
         Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
         //Chroma subsampling should be the same as file chroma (there is not chroma convert) but nvjpeg2k accepts only 444, 422, 420 
         Values(NVIMGCDCS_SAMPLING_NONE, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420), 
         Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
         Values(true)));



class NvJpeg2kExtDecoderTestSingleImageWithStatus
    : public NvJpeg2kExtDecoderTestBase,
      public TestWithParam<
          std::tuple<const char*, nvimgcdcsColorSpec_t, nvimgcdcsSampleFormat_t, nvimgcdcsChromaSubsampling_t, bool, nvimgcdcsProcessingStatus_t >>
{
  public:
    virtual ~NvJpeg2kExtDecoderTestSingleImageWithStatus() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        NvJpeg2kExtDecoderTestBase::SetUp();
        params_.enable_color_conversion = std::get<4>(GetParam());
        expected_status_ =  std::get<5>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpeg2kExtDecoderTestBase::TearDown();
    }
    nvimgcdcsProcessingStatus_t expected_status_;
};

TEST_P(NvJpeg2kExtDecoderTestSingleImageWithStatus, InvalidFormatsOrParameters)
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
    ASSERT_EQ(expected_status_, status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

 INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_INVALID_OUTPUT_CHROMA, NvJpeg2kExtDecoderTestSingleImageWithStatus,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_SYCC),
         Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
         Values(NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY, NVIMGCDCS_SAMPLING_410V), 
         Values(true),
         Values(NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_INVALID_OUTPUT_FORMAT, NvJpeg2kExtDecoderTestSingleImageWithStatus,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_SRGB, NVIMGCDCS_COLORSPEC_SYCC),
         Values(NVIMGCDCS_SAMPLEFORMAT_P_BGR, NVIMGCDCS_SAMPLEFORMAT_I_BGR),
         Values(NVIMGCDCS_SAMPLING_444), 
         Values(true),
         Values(NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_INVALID_COLORSPEC, NvJpeg2kExtDecoderTestSingleImageWithStatus,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCDCS_COLORSPEC_CMYK, NVIMGCDCS_COLORSPEC_YCCK),
         Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB),
         Values(NVIMGCDCS_SAMPLING_444), 
         Values(true),
         Values(NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED)));


// clang-format on       

}} // namespace nvimgcdcs::test
