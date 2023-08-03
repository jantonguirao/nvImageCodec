/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodecs_tests.h"
#include "parsers/parser_test_utils.h"
#include "parsers/pnm.h"

namespace nvimgcdcs { namespace test {

class PNMParserPluginTest : public ::testing::Test
{
  public:
    PNMParserPluginTest() {}

    void SetUp() override
    {
        nvimgcdcsInstanceCreateInfo_t create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
        create_info.message_severity = NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        pnm_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_pnm_parser_extension_desc(&pnm_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &pnm_parser_extension_, &pnm_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(stream_handle_));
        nvimgcdcsExtensionDestroy(pnm_parser_extension_);
        nvimgcdcsInstanceDestroy(instance_);
    }

    void TestComments(const char* data, size_t data_size)
    {
        LoadImageFromHostMemory(instance_, stream_handle_, reinterpret_cast<const uint8_t*>(data), data_size);
        nvimgcdcsImageInfo_t info;
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
        EXPECT_EQ(NVIMGCDCS_SAMPLING_NONE, info.chroma_subsampling);
        EXPECT_EQ(NVIMGCDCS_COLORSPEC_SRGB, info.color_spec);
        EXPECT_EQ(0, info.orientation.rotated);
        EXPECT_EQ(false, info.orientation.flip_x);
        EXPECT_EQ(false, info.orientation.flip_y);
        EXPECT_EQ(1, info.num_planes);
        EXPECT_EQ(1, info.plane_info[0].num_channels);
        EXPECT_EQ(6, info.plane_info[0].width);
        EXPECT_EQ(10, info.plane_info[0].height);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    }

    nvimgcdcsImageInfo_t expected_cat_2184682_640()
    {
        nvimgcdcsImageInfo_t info;
        memset(&info, 0, sizeof(nvimgcdcsImageInfo_t));
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
        info.orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 398;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 0;
        }
        return info;
    }

    nvimgcdcsImageInfo_t expected_cat_1245673_640()
    {
        nvimgcdcsImageInfo_t info;
        memset(&info, 0, sizeof(nvimgcdcsImageInfo_t));
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
        info.orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 0;
        }
        return info;
    }

    nvimgcdcsImageInfo_t expected_cat_111793_640() {
        nvimgcdcsImageInfo_t info;
        memset(&info, 0, sizeof(nvimgcdcsImageInfo_t));
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
        info.orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 426;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 0;
        }
        return info;
    }
    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t pnm_parser_extension_desc_{};
    nvimgcdcsExtension_t pnm_parser_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
};


TEST_F(PNMParserPluginTest, ValidPbm)
{
    auto buffer = read_file(resources_dir + "/pnm/cat-2184682_640.pbm");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected = expected_cat_2184682_640();
    expected.num_planes = 1;
    expected.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_Y;
    expect_eq(expected, info);
}

TEST_F(PNMParserPluginTest, ValidPgm)
{
    auto buffer = read_file(resources_dir + "/pnm/cat-1245673_640.pgm");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected = expected_cat_1245673_640();
    expected.num_planes = 1;
    expected.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_Y;
    expect_eq(expected, info);
}


TEST_F(PNMParserPluginTest, ValidPpm)
{
    auto buffer = read_file(resources_dir + "/pnm/cat-111793_640.ppm");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected = expected_cat_111793_640();
    expected.num_planes = 3;
    expected.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    expect_eq(expected, info);
}

TEST_F(PNMParserPluginTest, ValidPbmComment)
{
    const char data[] =
        "P1\n"
        "#This is an example bitmap of the letter \"J\"\n"
        "6 10\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "1 0 0 0 1 0\n"
        "0 1 1 1 0 0\n"
        "0 0 0 0 0 0\n"
        "0 0 0 0 0 0\n";
    TestComments(data, sizeof(data));
}

TEST_F(PNMParserPluginTest, ValidPbmCommentInsideToken)
{
  const char data[] =
      "P1\n"
      "6 1#Comment can be inside of a token\n"
      "0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
    TestComments(data, sizeof(data));
}

TEST_F(PNMParserPluginTest, ValidPbmCommentInsideWhitespaces)
{
  const char data[] =
      "P1 \n"
      "#Comment can be inside of whitespaces\n"
      " 6 10\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
    TestComments(data, sizeof(data));
}

TEST_F(PNMParserPluginTest, CannotParsePamFormat)
{
    const char data[] = "P7 \n";
    ASSERT_EQ(NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED,
        nvimgcdcsCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                             reinterpret_cast<const uint8_t*>(data), sizeof(data)));
}

TEST_F(PNMParserPluginTest, CanParseAllKindsOfWhitespace)
{
    for (uint8_t whitespace : {' ', '\n', '\f', '\r', '\t', '\v'}) {
        const uint8_t data[] = {'P', '6', whitespace};
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            nvimgcdcsCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                                reinterpret_cast<const uint8_t*>(data), sizeof(data)));
    }
}

TEST_F(PNMParserPluginTest, MissingWhitespace)
{
    const char data[] = "P61\n";
    ASSERT_EQ(NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED,
        nvimgcdcsCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                             reinterpret_cast<const uint8_t*>(data), sizeof(data)));
}

TEST_F(PNMParserPluginTest, LowercaseP)
{
    const char data[] = "p6 \n";
    ASSERT_EQ(NVIMGCDCS_STATUS_CODESTREAM_UNSUPPORTED,
        nvimgcdcsCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                             reinterpret_cast<const uint8_t*>(data), sizeof(data)));
}


}} // namespace nvimgcdcs::test
