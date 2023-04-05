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

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t jpeg2k_parser_extension_desc_{};
    nvimgcdcsExtension_t jpeg2k_parser_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
};

TEST_F(JPEG2KParserPluginTest, Uint8) {
    LoadImageFromFilename(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_NONE, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(475, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEG2KParserPluginTest, TiledUint8) {
    LoadImageFromFilename(resources_dir + "/jpeg2k/tiled-cat-1046544_640.jp2");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_NONE, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(475, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEG2KParserPluginTest, TiledUint16) {
    LoadImageFromFilename(resources_dir + "/jpeg2k/cat-1046544_640-16bit.jp2");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_NONE, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(475, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEG2KParserPluginTest, ErrorUnexpectedEnd) {
    std::vector<uint8_t> data = {0, 0, 0, 8, 'j', 'P', ' ', ' '};
    LoadImageFromHostMemory(data.data(), data.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_NE(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
}


}  // namespace test
}  // namespace nvimgcdcs