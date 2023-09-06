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
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/bmp.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class NvbmpExtDecoderTest : public ::testing::Test, public CommonExtDecoderTest
{
  public:
    NvbmpExtDecoderTest() {}

    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcdcsExtensionDesc_t bmp_parser_extension_desc;
        bmp_parser_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_bmp_parser_extension_desc(&bmp_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &bmp_parser_extension_desc);

        nvimgcdcsExtensionDesc_t nvbmp_extension_desc;
        nvbmp_extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvbmp_extension_desc.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvbmp_extension_desc(&nvbmp_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &extensions_.back(), &nvbmp_extension_desc));
    }

    void TearDown() override
    {
        CommonExtDecoderTest::TearDown();
    }
};

TEST_F(NvbmpExtDecoderTest, NVBMP_SingleImage_RGB_I)
{
    TestSingleImage("bmp/cat-111793_640.bmp", NVIMGCDCS_SAMPLEFORMAT_I_RGB);
}

TEST_F(NvbmpExtDecoderTest, NVBMP_SingleImage_RGB_P)
{
    TestSingleImage("bmp/cat-111793_640.bmp", NVIMGCDCS_SAMPLEFORMAT_P_RGB);
}

}} // namespace nvimgcdcs::test
