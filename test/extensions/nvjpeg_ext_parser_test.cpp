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
#include <parsers/jpeg.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodecs_tests.h"

namespace nvimgcdcs { namespace test {

class NvJpegExtParserTest : public ::testing::Test
{
  public:
    NvJpegExtParserTest() {}

    void SetUp() override
    {
        nvimgcdcsInstanceCreateInfo_t create_info;
        create_info.type = NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.next = nullptr;
        create_info.device_allocator = nullptr;
        create_info.pinned_allocator = nullptr;
        create_info.load_builtin_modules = false;
        create_info.load_extension_modules = false;
        create_info.executor = nullptr;
        create_info.num_cpu_threads = 1;
        create_info.message_severity = NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_type = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;


        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        nvjpeg_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvjpeg_extension_desc(&nvjpeg_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &nvjpeg_extension_, &nvjpeg_extension_desc_);

        jpeg_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &jpeg_parser_extension_, &jpeg_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(stream_handle_));
        if (out_stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(out_stream_handle_));
        nvimgcdcsExtensionDestroy(nvjpeg_extension_);
        nvimgcdcsInstanceDestroy(instance_);
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t jpeg_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t nvjpeg_extension_desc_{};
    nvimgcdcsExtension_t jpeg_parser_extension_;
    nvimgcdcsExtension_t nvjpeg_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
    nvimgcdcsCodeStream_t out_stream_handle_ = nullptr;
};

TEST_F(NvJpegExtParserTest, Parse_CSS_410)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_410.jpg");
    nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_UNKNOWN, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_410, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    EXPECT_EQ(426, info.plane_info[0].height);
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(1, info.plane_info[0].num_channels);
    EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    for (int p = 1; p < info.num_planes; p++) {
        EXPECT_EQ(426 / 2, info.plane_info[p].height);
        EXPECT_EQ(640 / 4, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(NvJpegExtParserTest, Parse_CSS_410_Extended_Jpeg_info_Baseline_DCT)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_410.jpg");
    nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsJpegImageInfo_t jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    info.next = &jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_UNKNOWN, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_410, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    EXPECT_EQ(426, info.plane_info[0].height);
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(1, info.plane_info[0].num_channels);
    EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    for (int p = 1; p < info.num_planes; p++) {
        EXPECT_EQ(426 / 2, info.plane_info[p].height);
        EXPECT_EQ(640 / 4, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }

    EXPECT_EQ(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, jpeg_info.encoding);
}

TEST_F(NvJpegExtParserTest, Parse_CSS_420_Extended_Jpeg_info_Progressive_DCT)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg");
    nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    nvimgcdcsJpegImageInfo_t jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
    info.next = &jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_UNKNOWN, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_420, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    EXPECT_EQ(213, info.plane_info[0].height);
    EXPECT_EQ(200, info.plane_info[0].width);
    EXPECT_EQ(1, info.plane_info[0].num_channels);
    EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    for (int p = 1; p < info.num_planes; p++) {
        EXPECT_EQ((213 + 1) / 2, info.plane_info[p].height);
        EXPECT_EQ((200 + 1) / 2, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }

    EXPECT_EQ(NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, jpeg_info.encoding);
}

}} // namespace nvimgcdcs::test
