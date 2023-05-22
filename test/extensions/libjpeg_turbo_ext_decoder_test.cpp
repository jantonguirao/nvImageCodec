/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <extensions/libjpeg_turbo/libjpeg_turbo_ext.h>
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/jpeg.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class LibjpegTurboExtDecoderTest : public ::testing::Test, public CommonExtDecoderTest
{
  public:
    LibjpegTurboExtDecoderTest() {}

    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcdcsExtensionDesc_t jpeg_parser_extension_desc;
        jpeg_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &jpeg_parser_extension_desc);

        nvimgcdcsExtensionDesc_t libjpeg_turbo_extension_desc;
        libjpeg_turbo_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        libjpeg_turbo_extension_desc.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_libjpeg_turbo_extension_desc(&libjpeg_turbo_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &libjpeg_turbo_extension_desc));
    }

    void TearDown() override
    {
        CommonExtDecoderTest::TearDown();
    }
};

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_410_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_410.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_411_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_411.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_420_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_420.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_420_BGR_I)
{
    TestSingleImage("jpeg/padlock-406986_640_420.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_420_RGB_P)
{
    TestSingleImage("jpeg/padlock-406986_640_420.jpg", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_420_BGR_P)
{
    TestSingleImage("jpeg/padlock-406986_640_420.jpg", NVIMGCDCS_SAMPLEFORMAT_P_BGR);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_422_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_440_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_440.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_444_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_444.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_RGB_Grayscale_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_gray.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_Grayscale_P_Y)
{
    TestSingleImage("jpeg/padlock-406986_640_gray.jpg", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_CMYK_RGB_I)
{
    TestSingleImage("jpeg/cmyk.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_YCCK_RGB_I)
{
    TestSingleImage("jpeg/ycck_colorspace.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, SingleImage_Progressive_RGB_I)
{
    TestSingleImage("jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibjpegTurboExtDecoderTest, EXIFOrientationUnsupported)
{
    std::vector<std::string> image_names = {
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_rotate_180.jpg",
        "jpeg/exif/padlock-406986_640_rotate_270.jpg"};
    for (auto image_name : image_names) {
        TestNotSupported(image_name, NVIMGCDCS_SAMPLEFORMAT_I_RGB, NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8,
            NVIMGCDCS_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
    }
}

TEST_F(LibjpegTurboExtDecoderTest, ROIDecodingWholeImage)
{
    // Whole image
    nvimgcdcsRegion_t region1{NVIMGCDCS_STRUCTURE_TYPE_REGION, nullptr, 2};
    region1.start[0] = 0;
    region1.start[1] = 0;
    region1.end[0] = 426;
    region1.end[1] = 640;
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB, region1);
}

TEST_F(LibjpegTurboExtDecoderTest, ROIDecodingPortion)
{
    // Actual ROI
    nvimgcdcsRegion_t region2{NVIMGCDCS_STRUCTURE_TYPE_REGION, nullptr, 2};
    region2.start[0] = 10;
    region2.start[1] = 20;
    region2.end[0] = 10 + 100;
    region2.end[1] = 20 + 100;
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB, region2);
}

TEST_F(LibjpegTurboExtDecoderTest, SampleTypeUnsupported)
{
    for (auto sample_type : {NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32, NVIMGCDCS_SAMPLE_DATA_TYPE_INT16, NVIMGCDCS_SAMPLE_DATA_TYPE_INT8,
             NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16}) {
        TestNotSupported(
            "jpeg/padlock-406986_640_444.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB, sample_type, NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED);
    }
}

}} // namespace nvimgcdcs::test
