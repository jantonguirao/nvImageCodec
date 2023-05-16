/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <extensions/libtiff/libtiff_ext.h>
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/tiff.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class LibtiffExtDecoderTest : public ::testing::Test, public CommonExtDecoderTest
{
  public:
    LibtiffExtDecoderTest() {}

    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcdcsExtensionDesc_t tiff_parser_extension_desc;
        tiff_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc);

        nvimgcdcsExtensionDesc_t libtiff_extension_desc;
        libtiff_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        libtiff_extension_desc.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_libtiff_extension_desc(&libtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &libtiff_extension_desc));
    }

    void TearDown() override
    {
        CommonExtDecoderTest::TearDown();
    }
};

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_RGB_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_BGR_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_I_BGR);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_UNCHANGED_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_RGB_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_BGR_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_P_BGR);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_UNCHANGED_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_Grayscale)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCDCS_SAMPLEFORMAT_P_Y);
}

}} // namespace nvimgcdcs::test