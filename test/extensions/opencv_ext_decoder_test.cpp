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
#include <parsers/jpeg.h>
#include <parsers/jpeg2k.h>
#include <parsers/png.h>
#include <parsers/bmp.h>
#include <parsers/tiff.h>
#include <parsers/webp.h>
#include <parsers/pnm.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include "nvimgcodecs_tests.h"

#define DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcdcs { namespace test {

class OpenCVExtDecoderTest : public ::testing::Test
{
  public:
    OpenCVExtDecoderTest() {}

    void SetUp() override
    {
        nvimgcdcsInstanceCreateInfo_t create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
        create_info.num_cpu_threads = 3;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        jpeg_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &jpeg_parser_extension_, &jpeg_parser_extension_desc_);

        jpeg2k_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &jpeg2k_parser_extension_, &jpeg2k_parser_extension_desc_);

        png_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_png_parser_extension_desc(&png_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &png_parser_extension_, &png_parser_extension_desc_);

        bmp_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_bmp_parser_extension_desc(&bmp_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &bmp_parser_extension_, &bmp_parser_extension_desc_);

        pnm_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_pnm_parser_extension_desc(&pnm_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &pnm_parser_extension_, &pnm_parser_extension_desc_);

        tiff_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &tiff_parser_extension_, &tiff_parser_extension_desc_);

        webp_parser_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_webp_parser_extension_desc(&webp_parser_extension_desc_));
        nvimgcdcsExtensionCreate(instance_, &webp_parser_extension_, &webp_parser_extension_desc_);

        opencv_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        opencv_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_opencv_extension_desc(&opencv_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &opencv_extension_, &opencv_extension_desc_));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        images_.clear();
        streams_.clear();

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder_, NVIMGCDCS_DEVICE_CURRENT));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
        params_.enable_orientation = true;
        params_.enable_color_conversion = true;
    }

    void TearDown() override
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDestroy(decoder_));
        if (future_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureDestroy(future_));
        if (image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(image_));
        if (in_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(in_code_stream_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(opencv_extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(jpeg_parser_extension_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_));
    }

    void TestSingleImage(const std::string& rel_path, nvimgcdcsSampleFormat_t sample_format,
        nvimgcdcsRegion_t region = {NVIMGCDCS_STRUCTURE_TYPE_REGION, nullptr, 0})
    {
        std::string filename = resources_dir + "/" + rel_path;
        std::string reference_filename = std::filesystem::path(resources_dir + "/ref/" + rel_path).replace_extension(".ppm");
        int num_channels = sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y ? 1 : 3;
        auto cv_type = num_channels == 1 ? CV_8UC1 : CV_8UC3;
        int cv_flags = num_channels == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
        cv::Mat ref;
        if (region.ndim == 0) {
            ref = cv::imread(reference_filename, cv_flags);
        } else {
            int start_x = region.start[1];
            int start_y = region.start[0];
            int crop_w = region.end[1] - region.start[1];
            int crop_h = region.end[0] - region.start[0];
            cv::Mat tmp = cv::imread(reference_filename, cv_flags);
            cv::Rect roi(start_x, start_y, crop_w, crop_h);
            tmp(roi).copyTo(ref);
        }

        bool planar = sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB ||sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR;
        bool rgb = sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB || sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB;
        if (rgb)
            ref = bgr2rgb(ref);

        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        uint32_t& width = image_info_.plane_info[0].width;
        uint32_t& height = image_info_.plane_info[0].height;
        bool swap_xy = params_.enable_orientation && image_info_.orientation.rotated % 180 == 90;
        if (swap_xy) {
            std::swap(width, height);
        }
        if (region.ndim == 2) {
            width = region.end[1] - region.start[1];
            height = region.end[0] - region.start[0];
        }

        image_info_.region = region;
        image_info_.num_planes = planar ? num_channels : 1;
        int plane_nchannels = planar ? 1 : num_channels;
        for (int p = 0; p < image_info_.num_planes; p++) {
            image_info_.plane_info[p].width = width;
            image_info_.plane_info[p].height = height;
            image_info_.plane_info[p].row_stride = width * plane_nchannels;
            image_info_.plane_info[p].num_channels = plane_nchannels;
            image_info_.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        }
        image_info_.buffer_size = height * width *num_channels;
        out_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &image_, &image_info_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));

        nvimgcdcsProcessingStatus_t status;
        size_t status_size;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &status, &status_size));
        ASSERT_EQ(NVIMGCDCS_PROCESSING_STATUS_SUCCESS, status);

        ASSERT_EQ(ref.size[0], height);
        ASSERT_EQ(ref.size[1], width);
        ASSERT_EQ(ref.type(), cv_type);
#ifdef DEBUG_DUMP_DECODE_OUTPUT
        cv::Mat decoded_image(height, width, cv_type, static_cast<void*>(out_buffer_.data()));
        cv::imwrite("./decode_out.pnm", rgb2bgr(decoded_image));
        cv::imwrite("./ref.pnm", rgb2bgr(ref));
#endif

        uint8_t eps = 1;
        if (rel_path.find("exif") != std::string::npos) {
            eps = 4;
        }
        else if (rel_path.find("cmyk") != std::string::npos) {
            eps = 2;
        }
        if (planar) {
            size_t out_pos = 0;
            for (size_t c = 0; c < num_channels; c++) {
                for (size_t i = 0; i < height; i++) {
                    for (size_t j = 0; j < width; j++, out_pos++) {
                        auto out_val = out_buffer_[out_pos];
                        size_t ref_pos = i * width * num_channels + j * num_channels + c;
                        auto ref_val = ref.data[ref_pos];
                        ASSERT_NEAR(out_val, ref_val, eps)
                            << "@" << i << "x" << j << "x" << c << " : " << (int)out_val << " != " << (int)ref_val << "\n";
                    }
                }
            }
        } else {
            size_t pos = 0;
            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    for (size_t c = 0; c < num_channels; c++, pos++) {
                        ASSERT_NEAR(out_buffer_.data()[pos], ref.data[pos], eps)
                            << "@" << i << "x" << j << "x" << c << " : " << (int)out_buffer_.data()[pos] << " != " << (int)ref.data[pos]
                            << "\n";
                    }
                }
            }
        }
    }

    void TestNotSupported(const std::string& rel_path, nvimgcdcsSampleFormat_t sample_format, nvimgcdcsSampleDataType_t sample_type,
        nvimgcdcsProcessingStatus_t expected_status)
    {
        std::string filename = resources_dir + "/" + rel_path;

        int num_channels = sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y ? 1 : 3;
        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        // image_info_.plane_info[0].height = image_info_.plane_info[0].height;
        // image_info_.plane_info[0].width = image_info_.plane_info[0].width;
        image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width * num_channels;
        image_info_.plane_info[0].num_channels = num_channels;
        image_info_.plane_info[0].sample_type = sample_type;
        image_info_.buffer_size =
            image_info_.plane_info[0].height * image_info_.plane_info[0].width * image_info_.plane_info[0].num_channels;
        out_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageCreate(instance_, &image_, &image_info_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureWaitForAll(future_));

        nvimgcdcsProcessingStatus_t status;
        size_t status_size;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureGetProcessingStatus(future_, &status, &status_size));
        ASSERT_EQ(expected_status, status);
    }

    nvimgcdcsInstance_t instance_;
    nvimgcdcsExtensionDesc_t jpeg_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t jpeg2k_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t png_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t bmp_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t pnm_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t tiff_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t webp_parser_extension_desc_{};
    nvimgcdcsExtensionDesc_t opencv_extension_desc_{};
    nvimgcdcsExtension_t jpeg_parser_extension_;
    nvimgcdcsExtension_t jpeg2k_parser_extension_;
    nvimgcdcsExtension_t png_parser_extension_;
    nvimgcdcsExtension_t bmp_parser_extension_;
    nvimgcdcsExtension_t pnm_parser_extension_;
    nvimgcdcsExtension_t tiff_parser_extension_;
    nvimgcdcsExtension_t webp_parser_extension_;
    nvimgcdcsExtension_t opencv_extension_;

    nvimgcdcsDecoder_t decoder_;
    nvimgcdcsDecodeParams_t params_;
    nvimgcdcsImageInfo_t image_info_;
    nvimgcdcsCodeStream_t in_code_stream_ = nullptr;
    nvimgcdcsImage_t image_ = nullptr;
    std::vector<nvimgcdcsImage_t> images_;
    std::vector<nvimgcdcsCodeStream_t> streams_;
    nvimgcdcsFuture_t future_ = nullptr;
    std::vector<uint8_t> out_buffer_;
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

TEST_F(OpenCVExtDecoderTest, JPEG_SampleTypeUnsupported)
{
    for (auto sample_type : {NVIMGCDCS_SAMPLE_DATA_TYPE_FLOAT32, NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16, NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8,
             NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16}) {
        TestNotSupported("jpeg/padlock-406986_640_444.jpg", NVIMGCDCS_SAMPLEFORMAT_I_RGB, sample_type,
            NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED);
    }
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

}} // namespace nvimgcdcs::test
