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
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "nvimgcodecs_tests.h"
#include "nvjpeg_ext_test_common.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

#define NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM 0
#define NV_DEVELOPER_DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcdcs { namespace test {

class NvJpegExtEncoderTestBase : public NvJpegExtTestBase
{
  public:
    NvJpegExtEncoderTestBase() {}

    virtual void SetUp()
    {
        NvJpegExtTestBase::SetUp();
        const char* options = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderCreate(instance_, &encoder_, NVIMGCDCS_DEVICE_CURRENT, 0, nullptr, options));

        jpeg_enc_params_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, 0};
        jpeg_enc_params_.optimized_huffman = 0;
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_ENCODE_PARAMS, 0};
        params_.next = &jpeg_enc_params_;
        params_.quality = 95;
        params_.target_psnr = 0;
        out_jpeg_image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
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
    nvimgcdcsJpegImageInfo_t out_jpeg_image_info_;
};

class NvJpegExtEncoderTestSingleImage : public NvJpegExtEncoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<const char*, nvimgcdcsColorSpec_t, nvimgcdcsSampleFormat_t,
                                            nvimgcdcsChromaSubsampling_t, nvimgcdcsChromaSubsampling_t, nvimgcdcsJpegEncoding_t>>
{
  public:
    virtual ~NvJpegExtEncoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        NvJpegExtEncoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        encoded_chroma_subsampling_ = std::get<4>(GetParam());
        out_jpeg_image_info_.encoding = std::get<5>(GetParam());
        image_info_.next = &out_jpeg_image_info_;
    }

    void TearDown() override
    {
        NvJpegTestBase::TearDown();
        NvJpegExtEncoderTestBase::TearDown();
    }

    nvimgcdcsChromaSubsampling_t encoded_chroma_subsampling_;
};

TEST_P(NvJpegExtEncoderTestSingleImage, ValidFormatAndParameters)
{
    nvimgcdcsImageInfo_t ref_cs_image_info;
    DecodeReference(resources_dir, image_file_, sample_format_, true, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    PrepareImageForFormat();
    memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());

    nvimgcdcsImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name,"jpeg");
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
                                            &NvJpegExtEncoderTestSingleImage::GetBufferStatic, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
    if (NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM) {
        std::ofstream b_stream("./encoded_out.jpg", std::fstream::out | std::fstream::binary);
        b_stream.write(reinterpret_cast<char*>(code_stream_buffer_.data()), code_stream_buffer_.size());
    }

    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcdcsImageInfo_t load_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsJpegImageInfo_t load_jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    load_info.next = &load_jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &load_info));
    EXPECT_EQ(out_jpeg_image_info_.encoding, load_jpeg_info.encoding);
    EXPECT_EQ(cs_image_info.chroma_subsampling, load_info.chroma_subsampling);

    std::vector<unsigned char> ref_out_buffer;
    EncodeReference(image_info_, params_, jpeg_enc_params_, cs_image_info, out_jpeg_image_info_, &ref_out_buffer);
    ASSERT_EQ(0,
        memcmp(reinterpret_cast<void*>(ref_out_buffer.data()), reinterpret_cast<void*>(code_stream_buffer_.data()), ref_out_buffer.size()));
}

// clang-format off

static const char* css_filenames[] = {"/jpeg/padlock-406986_640_410.jpg", "/jpeg/padlock-406986_640_411.jpg",
    "/jpeg/padlock-406986_640_420.jpg", "/jpeg/padlock-406986_640_422.jpg", "/jpeg/padlock-406986_640_440.jpg",
    "/jpeg/padlock-406986_640_444.jpg", "/jpeg/padlock-406986_640_gray.jpg"};

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SRGB_INPUT_FORMATS_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        ValuesIn(css_filenames),
        Values(NVIMGCDCS_COLORSPEC_SRGB),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_RGB, NVIMGCDCS_SAMPLEFORMAT_I_RGB, NVIMGCDCS_SAMPLEFORMAT_P_BGR, NVIMGCDCS_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCDCS_SAMPLING_444),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS444_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCDCS_SAMPLING_444),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410,  NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS410_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_410.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCDCS_SAMPLING_410),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS411_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_411.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCDCS_SAMPLING_411),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS420_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_420.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCDCS_SAMPLING_420),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS422_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_422.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCDCS_SAMPLING_422),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS440_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_440.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCDCS_SAMPLING_440),
        Values(NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_422 , NVIMGCDCS_SAMPLING_420, 
            NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));


INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_GRAY_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        ValuesIn(css_filenames),
        Values(NVIMGCDCS_COLORSPEC_GRAY, NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_Y),
        Values(NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));


class NvJpegExtEncoderTestSingleImageWithStatus : public NvJpegExtEncoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<const char*, nvimgcdcsColorSpec_t, nvimgcdcsSampleFormat_t,
                                            nvimgcdcsChromaSubsampling_t, nvimgcdcsChromaSubsampling_t, nvimgcdcsJpegEncoding_t, nvimgcdcsProcessingStatus_t>>
{
  public:
    virtual ~NvJpegExtEncoderTestSingleImageWithStatus() = default;

  protected:
    void SetUp() override
    {
        NvJpegExtEncoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        encoded_chroma_subsampling_ = std::get<4>(GetParam());
        out_jpeg_image_info_.encoding = std::get<5>(GetParam());
        expected_status_ =  std::get<6>(GetParam());
        image_info_.next = &out_jpeg_image_info_;

    }

    virtual void TearDown()
    {
        NvJpegTestBase::TearDown();
        NvJpegExtEncoderTestBase::TearDown();
    }

    nvimgcdcsChromaSubsampling_t encoded_chroma_subsampling_;
    nvimgcdcsProcessingStatus_t expected_status_;
};


TEST_P(NvJpegExtEncoderTestSingleImageWithStatus, InvalidFormatsOrParameters)
{
    nvimgcdcsImageInfo_t ref_cs_image_info;
    DecodeReference(resources_dir, image_file_, sample_format_, true, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    PrepareImageForFormat();
    memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());

    nvimgcdcsImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name, "jpeg");
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
     &NvJpegExtEncoderTestSingleImageWithStatus::GetBufferStatic, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));
    nvimgcdcsProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(expected_status_, encode_status);
    ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, 1);
}

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_OUTPUT_CHROMA_FOR_P_Y_FORMAT_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCDCS_COLORSPEC_GRAY, NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_Y),
        Values(NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_410V, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED | NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_INPUT_CHROMA_FOR_P_Y_FORMAT_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCDCS_COLORSPEC_GRAY, NVIMGCDCS_COLORSPEC_SYCC),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_Y),
        Values(NVIMGCDCS_SAMPLING_440, 
                NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410,  NVIMGCDCS_SAMPLING_410V,  NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420),
        Values(NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED | NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_COLOR_SPEC_FOR_P_Y_FORMAT, NvJpegExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCDCS_COLORSPEC_SRGB, NVIMGCDCS_COLORSPEC_YCCK, NVIMGCDCS_COLORSPEC_CMYK),
        Values(NVIMGCDCS_SAMPLEFORMAT_P_Y),
        Values(NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_SAMPLING_GRAY),
        Values(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT),
        Values(NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED| NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

// clang-format on

}} // namespace nvimgcdcs::test