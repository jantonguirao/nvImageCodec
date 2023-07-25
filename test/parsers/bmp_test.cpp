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
#include "parsers/bmp.h"
#include "parsers/parser_test_utils.h"
#include "nvimgcodecs_tests.h"
#include <nvimgcodecs.h>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>

namespace nvimgcdcs {
namespace test {

class BMPParserPluginTest : public ::testing::Test
{
  public:
    BMPParserPluginTest()
    {
    }

    void SetUp() override
    {
        nvimgcdcsInstanceCreateInfo_t create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
        create_info.num_cpu_threads = 1;
        create_info.message_severity = NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            nvimgcdcsInstanceCreate(&instance_, create_info));

        bmp_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
         ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            get_bmp_parser_extension_desc(&bmp_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &bmp_parser_extension_, &bmp_parser_extension_desc_);
    }

    void TearDown() override {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
                nvimgcdcsCodeStreamDestroy(stream_handle_));
        nvimgcdcsExtensionDestroy(bmp_parser_extension_);
        nvimgcdcsInstanceDestroy(instance_);
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
    nvimgcdcsExtensionDesc_t bmp_parser_extension_desc_{};
    nvimgcdcsExtension_t bmp_parser_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
};

TEST_F(BMPParserPluginTest, RGB) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640.bmp");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_111793_640(), info);
}

TEST_F(BMPParserPluginTest, RGB_FromHostMem) {
    auto buffer = read_file(resources_dir + "/bmp/cat-111793_640.bmp");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_111793_640(), info);
}

TEST_F(BMPParserPluginTest, Grayscale) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640_grayscale.bmp");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_111793_640();
    expected_info.color_spec = NVIMGCDCS_COLORSPEC_GRAY;
    expected_info.num_planes = 1;
    expected_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_Y;
    expect_eq(expected_info, info);
}


TEST_F(BMPParserPluginTest, Palette_1Bit) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640_palette_1bit.bmp");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_111793_640();
    expected_info.color_spec = NVIMGCDCS_COLORSPEC_GRAY;
    expected_info.num_planes = 1;
    expected_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_Y;
    expect_eq(expected_info, info);
}


TEST_F(BMPParserPluginTest, Palette_8Bit) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640_palette_8bit.bmp");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_111793_640();
    expected_info.num_planes = 3;
    expected_info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
    expect_eq(expected_info, info);
}

}  // namespace test
}  // namespace nvimgcdcs
