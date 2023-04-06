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
#include "parsers/jpeg.h"
#include "parsers/parser_test_utils.h"
#include "nvimgcodecs_tests.h"
#include <nvimgcodecs.h>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>

namespace nvimgcdcs {
namespace test {

static int NormalizeAngle(int degrees)
{
    return (degrees % 360 + 360) % 360;
}

class JPEGParserPluginTest : public ::testing::Test
{
  public:
    JPEGParserPluginTest()
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

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            nvimgcdcsInstanceCreate(&instance_, create_info));

        jpeg_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
         ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
            get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &jpeg_parser_extension_, &jpeg_parser_extension_desc_);
    }

    void TearDown() override {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
                nvimgcdcsCodeStreamDestroy(stream_handle_));
        nvimgcdcsExtensionDestroy(jpeg_parser_extension_);
        nvimgcdcsInstanceDestroy(instance_);
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t jpeg_parser_extension_desc_{};
    nvimgcdcsExtension_t jpeg_parser_extension_;
    nvimgcdcsCodeStream_t stream_handle_ = nullptr;
};

TEST_F(JPEGParserPluginTest, YCC_410) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_410.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_410, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, YCC_410_Extended_JPEG_info) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_410.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    nvimgcdcsJpegImageInfo_t jpeg_info;
    jpeg_info.type = NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO;
    info.next = &jpeg_info;

    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_410, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }

    EXPECT_EQ(NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, jpeg_info.encoding);
}


TEST_F(JPEGParserPluginTest, YCC_411) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_411.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_411, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_420) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_420, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_422) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_422.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_422, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_440) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_440.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_440, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_444) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_444.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_444, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, Gray) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_gray.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_Y, info.sample_format);
    EXPECT_EQ(1, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_GRAY, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_NONE, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, CMYK) {  // TODO(janton) : get a permissive license free image
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/cmyk-dali.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(4, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_CMYK, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_UNSUPPORTED, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(616, info.plane_info[p].height);
        EXPECT_EQ(792, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, YCCK) {  // TODO(janton) : get a permissive license free image
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/ycck_colorspace.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(4, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_YCCK, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_UNSUPPORTED, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(512, info.plane_info[p].height);
        EXPECT_EQ(512, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, File_vs_MemoryStream)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));

    auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcdcsImageInfo_t info2;
    info2.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info2.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info2));

    EXPECT_EQ(0, std::memcmp(&info, &info2, sizeof(info)));
}

TEST_F(JPEGParserPluginTest, Error_CreateStream_Empty)
{
    std::vector<uint8_t> empty;
    ASSERT_NE(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamCreateFromHostMem(instance_, &stream_handle_, empty.data(), empty.size()));
}

TEST_F(JPEGParserPluginTest, Error_CreateStream_BadSOI) {
  auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
  EXPECT_EQ(0xd8, buffer[1]);  // A valid JPEG starts with ff d8 (Start Of Image marker)...
  buffer[1] = 0xc0;            // ...but we make it ff c0, which is Start Of Frame
  EXPECT_NE(NVIMGCDCS_STATUS_SUCCESS,
    nvimgcdcsCodeStreamCreateFromHostMem(instance_, &stream_handle_, buffer.data(), buffer.size()));
}

TEST_F(JPEGParserPluginTest, Error_GetInfo_NoSOF) {
    auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    // We change Start Of Frame marker into a Comment marker
    auto bad = replace(buffer, {0xff, 0xc0}, {0xff, 0xfe});
    // It can match the JPEG parser
    EXPECT_EQ(NVIMGCDCS_STATUS_SUCCESS,
        nvimgcdcsCodeStreamCreateFromHostMem(instance_, &stream_handle_, bad.data(), bad.size()));
    // Fails to GetInfo (actual parsing) because there's no valid SOF marker
    nvimgcdcsImageInfo_t info;
    ASSERT_NE(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(JPEGParserPluginTest, Padding)
{
    /* https://www.w3.org/Graphics/JPEG/itu-t81.pdf section B.1.1.2 Markers
   * Any marker may optionally be preceded by any number of fill bytes,
   * which are bytes assigned code X’FF’ */
    auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    auto padded = replace(buffer, {0xff, 0xe0}, {0xff, 0xff, 0xff, 0xff, 0xe0});
    padded = replace(padded, {0xff, 0xe1}, {0xff, 0xff, 0xe1});
    padded = replace(padded, {0xff, 0xdb}, {0xff, 0xff, 0xff, 0xdb});
    padded = replace(padded, {0xff, 0xc0}, {0xff, 0xff, 0xff, 0xff, 0xff, 0xc0});
    EXPECT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamCreateFromHostMem(instance_, &stream_handle_, padded.data(), padded.size()));
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCDCS_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCDCS_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCDCS_SAMPLING_420, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, EXIF_NoOrientation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_no_orientation.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Horizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_horizontal.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorHorizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_horizontal.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(true, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Rotate180)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_rotate_180.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(180, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorVertical)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_vertical.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(true, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorHorizontalRotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 270, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(true, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Rotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 90, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorHorizontalRotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 90, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(true, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Rotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_rotate_270.jpg");
    nvimgcdcsImageInfo_t info;
    info.type = NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO;
    info.next = nullptr;
    ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 270, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

}  // namespace test
}  // namespace nvimgcdcs