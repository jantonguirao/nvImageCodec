/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <gtest/gtest.h>
#include <nvimgcodecs.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "nvimgcodecs_tests.h"

#define DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcdcs { namespace test {

class CommonExtDecoderTest
{
  public:
    CommonExtDecoderTest() {}


    void SetUp()
    {
        nvimgcdcsInstanceCreateInfo_t create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
        create_info.num_cpu_threads = 3;
        create_info.message_severity = NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ALL;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        images_.clear();
        streams_.clear();

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderCreate(instance_, &decoder_, NVIMGCDCS_DEVICE_CURRENT, 0, nullptr, nullptr));
        params_ = {NVIMGCDCS_STRUCTURE_TYPE_DECODE_PARAMS, 0};
        params_.apply_exif_orientation= 1;
        params_.enable_color_conversion= 1;
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsDecoderDestroy(decoder_));
        if (future_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureDestroy(future_));
        if (image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(image_));
        if (in_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(in_code_stream_));
        for (auto& ext : extensions_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(ext));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_));
    }

    void TestSingleImage(const std::string& rel_path, nvimgcdcsSampleFormat_t sample_format,
        nvimgcdcsRegion_t region = {NVIMGCDCS_STRUCTURE_TYPE_REGION, nullptr, 0})
    {
        std::string filename = resources_dir + "/" + rel_path;
        std::string reference_filename = std::filesystem::path(resources_dir + "/ref/" + rel_path).replace_extension(".ppm").string();
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

        bool planar = sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB || sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR ||
                      sample_format == NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED;
        bool bgr = sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR || sample_format == NVIMGCDCS_SAMPLEFORMAT_I_BGR;
        if (!bgr && num_channels >= 3)
            ref = bgr2rgb(ref);

        LoadImageFromFilename(instance_, in_code_stream_, filename);
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = NVIMGCDCS_COLORSPEC_SRGB;
        image_info_.num_planes = 1;
        uint32_t& width = image_info_.plane_info[0].width;
        uint32_t& height = image_info_.plane_info[0].height;
        bool swap_xy = params_.apply_exif_orientation && image_info_.orientation.rotated % 180 == 90;
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
        image_info_.buffer_size = height * width * num_channels;
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
#if DEBUG_DUMP_DECODE_OUTPUT
        cv::Mat decoded_image(height, width, cv_type, static_cast<void*>(out_buffer_.data()));
        cv::imwrite("./decode_out.ppm", rgb2bgr(decoded_image));
        cv::imwrite("./ref.ppm", rgb2bgr(ref));
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
    nvimgcdcsDecoder_t decoder_;
    nvimgcdcsDecodeParams_t params_;
    nvimgcdcsImageInfo_t image_info_;
    nvimgcdcsCodeStream_t in_code_stream_ = nullptr;
    nvimgcdcsImage_t image_ = nullptr;
    std::vector<nvimgcdcsImage_t> images_;
    std::vector<nvimgcdcsCodeStream_t> streams_;
    nvimgcdcsFuture_t future_ = nullptr;
    std::vector<uint8_t> out_buffer_;
    std::vector<nvimgcdcsExtension_t> extensions_;
};

}} // namespace nvimgcdcs::test
