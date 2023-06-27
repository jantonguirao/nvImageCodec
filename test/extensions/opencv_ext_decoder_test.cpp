/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <extensions/opencv/opencv_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/bmp.h>
#include <parsers/jpeg.h>
#include <parsers/jpeg2k.h>
#include <parsers/parser_test_utils.h>
#include <parsers/png.h>
#include <parsers/pnm.h>
#include <parsers/tiff.h>
#include <parsers/webp.h>
#include <test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "common_ext_decoder_test.h"
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class OpenCVExtDecoderTest : public ::testing::Test, public CommonExtDecoderTest
{
  public:
    OpenCVExtDecoderTest() {}

    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcdcsExtensionDesc_t jpeg_parser_extension_desc;
        jpeg_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &jpeg_parser_extension_desc);

        nvimgcdcsExtensionDesc_t jpeg2k_parser_extension_desc;
        jpeg2k_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &jpeg2k_parser_extension_desc);

        nvimgcdcsExtensionDesc_t png_parser_extension_desc;
        png_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_png_parser_extension_desc(&png_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &png_parser_extension_desc);

        nvimgcdcsExtensionDesc_t bmp_parser_extension_desc;
        bmp_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_bmp_parser_extension_desc(&bmp_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &bmp_parser_extension_desc);

        nvimgcdcsExtensionDesc_t pnm_parser_extension_desc;
        pnm_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_pnm_parser_extension_desc(&pnm_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &pnm_parser_extension_desc);

        nvimgcdcsExtensionDesc_t tiff_parser_extension_desc;
        tiff_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc);

        nvimgcdcsExtensionDesc_t webp_parser_extension_desc;
        webp_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_webp_parser_extension_desc(&webp_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &webp_parser_extension_desc);

        nvimgcdcsExtensionDesc_t opencv_extension_desc;
        opencv_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        opencv_extension_desc.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_opencv_extension_desc(&opencv_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &opencv_extension_desc));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        images_.clear();
        streams_.clear();

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder_, NVIMGCDCS_DEVICE_CURRENT, 0, nullptr, nullptr));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
        params_.enable_orientation = true;
        params_.enable_color_conversion = true;
    }

    void TearDown() override
    {
         CommonExtDecoderTest::TearDown();
    }
};

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_RGB_420_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_420.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_RGB_420_RGB_P)
{
    TestSingleImage("jpeg/padlock-406986_640_420.jpg", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_Grayscale_RGB_I)
{
    TestSingleImage("jpeg/padlock-406986_640_gray.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_Grayscale_P_Y)
{
    TestSingleImage("jpeg/padlock-406986_640_gray.jpg", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_CMYK_RGB_I)
{
    TestSingleImage("jpeg/cmyk.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_YCCK_RGB_I)
{
    TestSingleImage("jpeg/ycck_colorspace.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_SingleImage_Progressive_RGB_I)
{
    TestSingleImage("jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationMirrorHorizontalRotate90)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationMirrorHorizontalRotate270)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationMirrorHorizontal)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_mirror_horizontal.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationMirrorVertical)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_mirror_vertical.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationRotate90)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_rotate_90.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationRotate180)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_rotate_180.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_EXIFOrientationRotate270)
{
    TestSingleImage("jpeg/exif/padlock-406986_640_rotate_270.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG_ROIDecodingWholeImage)
{
    // Whole image
    nvimgcdcsRegion_t region1{NVIMGCDCS_STRUCTURE_TYPE_REGION, nullptr, 2};
    region1.start[0] = 0;
    region1.start[1] = 0;
    region1.end[0] = 426;
    region1.end[1] = 640;
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB, region1);
}

TEST_F(OpenCVExtDecoderTest, JPEG_ROIDecodingPortion)
{
    // Actual ROI
    nvimgcdcsRegion_t region2{NVIMGCDCS_STRUCTURE_TYPE_REGION, nullptr, 2};
    region2.start[0] = 10;
    region2.start[1] = 20;
    region2.end[0] = 10 + 100;
    region2.end[1] = 20 + 100;
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB, region2);
}

TEST_F(OpenCVExtDecoderTest, JPEG2K_SingleImage_RGB_I)
{
    TestSingleImage("jpeg2k/cat-1046544_640.jp2", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG2K_SingleImage_RGB_P)
{
    TestSingleImage("jpeg2k/cat-1046544_640.jp2", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, JPEG2K_SingleImage_Grayscale)
{
    TestSingleImage("jpeg2k/cat-1046544_640.jp2", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(OpenCVExtDecoderTest, PNG_SingleImage_RGB_I)
{
    TestSingleImage("png/cat-1245673_640.png", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, PNG_SingleImage_RGB_P)
{
    TestSingleImage("png/cat-1245673_640.png", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, PNG_SingleImage_Grayscale)
{
    TestSingleImage("png/cat-1245673_640.png", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(OpenCVExtDecoderTest, BMP_SingleImage_RGB_I)
{
    TestSingleImage("bmp/cat-111793_640.bmp", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, BMP_SingleImage_RGB_P)
{
    TestSingleImage("bmp/cat-111793_640.bmp", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, BMP_SingleImage_Grayscale)
{
    TestSingleImage("bmp/cat-111793_640.bmp", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(OpenCVExtDecoderTest, Webp_SingleImage_RGB_I)
{
    TestSingleImage("webp/lossy/cat-3113513_640.webp", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, Webp_SingleImage_RGB_P)
{
    TestSingleImage("webp/lossy/cat-3113513_640.webp", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, Webp_SingleImage_Grayscale)
{
    TestSingleImage("webp/lossy/cat-3113513_640.webp", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(OpenCVExtDecoderTest, PNM_SingleImage_RGB_I)
{
    TestSingleImage("pnm/cat-1245673_640.pgm", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, PNM_SingleImage_RGB_P)
{
    TestSingleImage("pnm/cat-1245673_640.pgm", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, PNM_SingleImage_Grayscale)
{
    TestSingleImage("pnm/cat-1245673_640.pgm", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

TEST_F(OpenCVExtDecoderTest, TIFF_SingleImage_RGB_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(OpenCVExtDecoderTest, TIFF_SingleImage_RGB_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(OpenCVExtDecoderTest, TIFF_SingleImage_Grayscale)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

}} // namespace nvimgcdcs::test
