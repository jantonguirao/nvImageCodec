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

#include <vector>

#include <nvimgcodecs.h>

#include <gtest/gtest.h>

namespace nvimgcdcs { namespace test {

class ExtensionTestBase
{
  public:
    virtual ~ExtensionTestBase() = default;
    virtual void SetUp()
    {
        nvimgcdcsInstanceCreateInfo_t create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
        create_info.num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceCreate(&instance_, create_info));

        images_.clear();
        streams_.clear();
    }

    virtual void TearDown() { ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsInstanceDestroy(instance_)); }

    virtual void TearDownCodecResources()
    {
        if (future_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsFutureDestroy(future_));
        if (in_image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(in_image_));
        if (out_image_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsImageDestroy(out_image_));
        if (in_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(in_code_stream_));
        if (out_code_stream_)
            ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsCodeStreamDestroy(out_code_stream_));
    }

    void PrepareImageForPlanarFormat(int num_planes = 3)
    {
        image_info_.num_planes = num_planes;
        for (int p = 0; p < image_info_.num_planes; p++) {
            image_info_.plane_info[p].height = image_info_.plane_info[0].height;
            image_info_.plane_info[p].width = image_info_.plane_info[0].width;
            image_info_.plane_info[p].row_stride = image_info_.plane_info[0].width;
            image_info_.plane_info[p].num_channels = 1;
            image_info_.plane_info[p].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
            image_info_.plane_info[p].precision = 0;
        }
        image_info_.buffer_size = image_info_.plane_info[0].height * image_info_.plane_info[0].width * image_info_.num_planes;
        image_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = image_buffer_.data();
        image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
    }

    void PrepareImageForInterleavedFormat()
    {
        image_info_.num_planes = 1;
        image_info_.plane_info[0].num_channels = 3;
        image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width * image_info_.plane_info[0].num_channels;
        image_info_.plane_info[0].sample_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
        image_info_.buffer_size = image_info_.plane_info[0].height * image_info_.plane_info[0].row_stride * image_info_.num_planes;
        image_buffer_.resize(image_info_.buffer_size);
        image_info_.buffer = image_buffer_.data();
        image_info_.buffer_kind = NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST;
    }

    void PrepareImageForFormat()
    {
        image_info_.color_spec = color_spec_;
        image_info_.sample_format = sample_format_;
        image_info_.chroma_subsampling = chroma_subsampling_;

        switch (sample_format_) {
        case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
        case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_P_RGB: {
            PrepareImageForPlanarFormat();
            break;
        }
        case NVIMGCDCS_SAMPLEFORMAT_P_Y: {
            PrepareImageForPlanarFormat(1);
            break;
        }
        case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED: {
            PrepareImageForPlanarFormat(image_info_.num_planes);
            break;
        }
        case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_I_RGB: {
            PrepareImageForInterleavedFormat();
            break;
        }
        default: {
            assert(!"TODO");
        }
        }
    }

    void Convert_P_RGB_to_I_RGB(std::vector<uint8_t>& out_buffer, const std::vector<uint8_t>& in_buffer, nvimgcdcsImageInfo_t image_info)
    {
        out_buffer.resize(in_buffer.size());
        for (int y = 0; y < image_info_.plane_info[0].height; y++) {
            for (int x = 0; x < image_info_.plane_info[0].width; x++) {
                for (int c = 0; c < image_info_.plane_info[0].num_channels; ++c) {
                    *(static_cast<uint8_t*>(image_info_.buffer) + y * image_info_.plane_info[0].row_stride +
                        x * image_info_.plane_info[0].num_channels + c) =
                        in_buffer[c * image_info_.plane_info[0].height * image_info_.plane_info[0].width +
                                  y * image_info_.plane_info[0].width + x];
                }
            }
        }
    }

    void Convert_I_RGB_to_P_RGB()
    {
        planar_out_buffer_.resize(image_buffer_.size());
        for (int y = 0; y < image_info_.plane_info[0].height; y++) {
            for (int x = 0; x < image_info_.plane_info[0].width; x++) {
                for (int c = 0; c < image_info_.plane_info[0].num_channels; ++c) {
                    planar_out_buffer_[c * image_info_.plane_info[0].height * image_info_.plane_info[0].width +
                                       y * image_info_.plane_info[0].width + x] =
                        *(static_cast<char*>(image_info_.buffer) + y * image_info_.plane_info[0].row_stride +
                            x * image_info_.plane_info[0].num_channels + c);
                }
            }
        }
    }

    void Convert_P_BGR_to_P_RGB()
    {
        planar_out_buffer_.resize(image_buffer_.size());
        auto plane_size = image_info_.plane_info[0].height * image_info_.plane_info[0].row_stride;
        memcpy(planar_out_buffer_.data(), static_cast<char*>(image_info_.buffer) + 2 * plane_size, plane_size);
        memcpy(planar_out_buffer_.data() + plane_size, static_cast<char*>(image_info_.buffer) + plane_size, plane_size);
        memcpy(planar_out_buffer_.data() + 2 * plane_size, static_cast<char*>(image_info_.buffer), plane_size);
    }

    void Convert_I_BGR_to_P_RGB()
    {
        planar_out_buffer_.resize(image_buffer_.size());
        for (int y = 0; y < image_info_.plane_info[0].height; y++) {
            for (int x = 0; x < image_info_.plane_info[0].width; x++) {
                for (int c = 0; c < image_info_.plane_info[0].num_channels; ++c) {
                    planar_out_buffer_[(image_info_.plane_info[0].num_channels - c - 1) * image_info_.plane_info[0].height *
                                           image_info_.plane_info[0].width +
                                       y * image_info_.plane_info[0].width + x] =
                        *(static_cast<char*>(image_info_.buffer) + y * image_info_.plane_info[0].row_stride +
                            x * image_info_.plane_info[0].num_channels + c);
                }
            }
        }
    }

    void ConvertToPlanar()
    {
        switch (sample_format_) {
        case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCDCS_SAMPLEFORMAT_P_Y:
        case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
        case NVIMGCDCS_SAMPLEFORMAT_P_RGB: {
            planar_out_buffer_.resize(image_buffer_.size());
            memcpy(planar_out_buffer_.data(), image_buffer_.data(), image_buffer_.size());
            break;
        }
        case NVIMGCDCS_SAMPLEFORMAT_I_RGB: {
            Convert_I_RGB_to_P_RGB();
            break;
        }
        case NVIMGCDCS_SAMPLEFORMAT_P_BGR: {
            Convert_P_BGR_to_P_RGB();
            break;
        }
        case NVIMGCDCS_SAMPLEFORMAT_I_BGR: {
            Convert_I_BGR_to_P_RGB();
            break;
        }
        default: {
            assert(!"TODO");
        }
        }
    }

    unsigned char* GetBuffer(size_t bytes)
    {
        code_stream_buffer_.resize(bytes);
        return code_stream_buffer_.data();
    }

    static unsigned char* GetBufferStatic(void* ctx, size_t bytes)
    {
        auto handle = reinterpret_cast<ExtensionTestBase*>(ctx);
        return handle->GetBuffer(bytes);
    }

    nvimgcdcsInstance_t instance_;
    std::string image_file_;
    nvimgcdcsCodeStream_t in_code_stream_ = nullptr;
    nvimgcdcsCodeStream_t out_code_stream_ = nullptr;
    nvimgcdcsImage_t in_image_ = nullptr;
    nvimgcdcsImage_t out_image_ = nullptr;
    std::vector<nvimgcdcsImage_t> images_;
    std::vector<nvimgcdcsCodeStream_t> streams_;
    nvimgcdcsFuture_t future_ = nullptr;

    nvimgcdcsImageInfo_t image_info_;
    nvimgcdcsSampleFormat_t reference_output_format_;
    std::vector<unsigned char> planar_out_buffer_;
    nvimgcdcsColorSpec_t color_spec_;
    nvimgcdcsSampleFormat_t sample_format_;
    nvimgcdcsChromaSubsampling_t chroma_subsampling_;
    std::vector<unsigned char> image_buffer_;
    std::vector<unsigned char> code_stream_buffer_;
};
}} // namespace nvimgcdcs::test
