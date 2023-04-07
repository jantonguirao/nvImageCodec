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
#include "parsers/tiff.h"

namespace nvimgcdcs { namespace test {

class TIFFParserPluginTest : public ::testing::Test
{
  public:
    TIFFParserPluginTest() {}

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

        tiff_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &tiff_parser_extension_, &tiff_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(stream_handle_));
        nvimgcdcsExtensionDestroy(tiff_parser_extension_);
        nvimgcdcsInstanceDestroy(instance_);
    }

    nvimgcdcsImageInfo_t expected_cat_1245673_640()
    {
        nvimgcdcsImageInfo_t info;
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
        info.chroma_subsampling = NVIMGCDCS_SAMPLING_NONE;
        info.orientation = {NVIMGCDCS_STRUCTURE_TYPE_ORIENTATION, nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        }
        return info;
    }

    nvimgcdcsImageInfo_t expected_cat_1046544_640()
    {
        nvimgcdcsImageInfo_t info;
        info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
        info.next = nullptr;
        info.sample_format = NVIMGCDCS_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCDCS_COLORSPEC_UNKNOWN;
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
    nvimgcdcsExtensionDesc_t tiff_parser_extension_desc_{};
    nvimgcdcsExtension_t tiff_parser_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
};

TEST_F(TIFFParserPluginTest, RGB)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_640.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

TEST_F(TIFFParserPluginTest, RGB_FromHostMem)
{
    auto buffer = read_file(resources_dir + "/tiff/cat-1245673_640.tiff");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Horizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_horizontal.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = false;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorHorizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_horizontal.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x = true;
    expected_info.orientation.flip_y = false;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Rotate180)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_rotate_180.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 180;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = false;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorVertical)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_vertical.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = true;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate270)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_horizontal_rotate_270.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = true;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Rotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_rotate_90.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = false;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate90)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_horizontal_rotate_90.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = true;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Rotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_rotate_270.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = false;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_NoOrientation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_no_orientation.tiff");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x = false;
    expected_info.orientation.flip_y = false;
    expect_eq(expected_info, info);
}

}} // namespace nvimgcdcs::test