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
#include "parsers/jpeg2k.h"
#include "parsers/parser_test_utils.h"
#include "nvimgcodecs_tests.h"
#include <nvimgcodecs.h>
#include <string>
#include <fstream>
#include <vector>

#include <cstring>

namespace nvimgcdcs {
namespace test {

class JPEG2KParserPluginTest : public ::testing::Test
{
  public:
    JPEG2KParserPluginTest()
    {
    }

    void SetUp() override
    {
        nvimgcdcsInstanceCreateInfo_t create_info;
        create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.next = nullptr;
        create_info.device_allocator = nullptr;
        create_info.pinned_allocator = nullptr;
        create_info.load_extension_modules = false;
        create_info.executor = nullptr;
        create_info.num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        jpeg2k_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &jpeg2k_parser_extension_, &jpeg2k_parser_extension_desc_);
    }

    void TearDown() override {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
                nvimgcdcsCodeStreamDestroy(stream_handle_));
        nvimgcdcsExtensionDestroy(jpeg2k_parser_extension_);
        nvimgcdcsInstanceDestroy(instance_);
    }

    void LoadImageFromFilename(const std::string& filename) {
        if (stream_handle_) {
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
                nvimgcdcsCodeStreamDestroy(stream_handle_));
            stream_handle_ = nullptr;
        }
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            nvimgcdcsCodeStreamCreateFromFile(instance_, &stream_handle_, filename.c_str()));
    }

    void LoadImageFromHostMemory(const uint8_t* data, size_t data_size) {
        if (stream_handle_) {
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
                nvimgcdcsCodeStreamDestroy(stream_handle_));
            stream_handle_ = nullptr;
        }
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            nvimgcdcsCodeStreamCreateFromHostMem(instance_, &stream_handle_, data, data_size));
    }

    nvimgcdcsImageInfo_t expected_cat_1046544_640() {
        nvimgcdcsImageInfo_t info;
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x = false;
        info.orientation.flip_y = false;
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 475;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        }
        return info;
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t jpeg2k_parser_extension_desc_{};
    nvimgcdcsExtension_t jpeg2k_parser_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
};

void ExpectEq(nvimgcdcsImageInfo_t expected, nvimgcdcsImageInfo_t actual) {
    EXPECT_EQ(expected.type, actual.type);
    EXPECT_EQ(expected.next, actual.next);
    EXPECT_EQ(expected.sample_format, actual.sample_format);
    EXPECT_EQ(expected.num_planes, actual.num_planes);
    EXPECT_EQ(expected.color_spec, actual.color_spec);
    EXPECT_EQ(expected.chroma_subsampling, actual.chroma_subsampling);
    EXPECT_EQ(expected.orientation.rotated, actual.orientation.rotated);
    EXPECT_EQ(expected.orientation.flip_x, actual.orientation.flip_x);
    EXPECT_EQ(expected.orientation.flip_y, actual.orientation.flip_y);
    for (int p = 0; p < expected.num_planes; p++) {
        EXPECT_EQ(expected.plane_info[p].height, actual.plane_info[p].height);
        EXPECT_EQ(expected.plane_info[p].width, actual.plane_info[p].width);
        EXPECT_EQ(expected.plane_info[p].num_channels, actual.plane_info[p].num_channels);
        EXPECT_EQ(expected.plane_info[p].sample_type, actual.plane_info[p].sample_type);
    }
}


TEST_F(JPEG2KParserPluginTest, Uint8) {
    LoadImageFromFilename(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    ExpectEq(expected_cat_1046544_640(), info);
}

TEST_F(JPEG2KParserPluginTest, Uint8_FromHostMem) {
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    LoadImageFromHostMemory(buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    ExpectEq(expected_cat_1046544_640(), info);
}

TEST_F(JPEG2KParserPluginTest, Uint8_CodeStreamOnly) {
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    std::vector<uint8_t> JP2_header_until_SOC = {
        0x0, 0x0, 0x0, 0xc, 0x6a, 0x50, 0x20, 0x20, 0xd, 0xa, 0x87, 0xa,
        0x0, 0x0, 0x0, 0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x70, 0x32,
        0x20, 0x0, 0x0, 0x0, 0x0, 0x6a, 0x70, 0x32, 0x20, 0x0, 0x0, 0x0,
        0x2d, 0x6a, 0x70, 0x32, 0x68, 0x0, 0x0, 0x0, 0x16, 0x69, 0x68,
        0x64, 0x72, 0x0, 0x0, 0x1, 0xdb, 0x0, 0x0, 0x2, 0x80, 0x0, 0x3,
        0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 0x63, 0x6f, 0x6c, 0x72,
        0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x10, 0x0, 0x8, 0x83, 0xe8, 0x6a,
        0x70, 0x32, 0x63, 0xff, 0x4f};
    std::vector<uint8_t> just_SOC = {0xff, 0x4f};
    buffer = replace(buffer, JP2_header_until_SOC, just_SOC);
    LoadImageFromHostMemory(buffer.data(), buffer.size());

    auto expected_info = expected_cat_1046544_640();
    expected_info.color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;  // don't have such info in codestream

    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    ExpectEq(expected_info, info);
}

TEST_F(JPEG2KParserPluginTest, TiledUint8) {
    LoadImageFromFilename(resources_dir + "/jpeg2k/tiled-cat-1046544_640.jp2");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    ExpectEq(expected_cat_1046544_640(), info);
}

TEST_F(JPEG2KParserPluginTest, TiledUint16) {
    LoadImageFromFilename(resources_dir + "/jpeg2k/cat-1046544_640-16bit.jp2");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    for (int p = 0; p < expected_info.num_planes; p++)
        expected_info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
    ExpectEq(expected_info, info);
}

TEST_F(JPEG2KParserPluginTest, ErrorUnexpectedEnd) {
    const std::array<uint8_t, 12> just_the_signatures = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
    LoadImageFromHostMemory(just_the_signatures.data(), just_the_signatures.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_NE(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
}


}  // namespace test
}  // namespace nvimgcdcs