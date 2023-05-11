
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

#include <fstream>

#include <gtest/gtest.h>

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <nvjpeg.h>
#include "common.h"

namespace nvimgcdcs { namespace test {

class NvJpegExtTestBase : public ExtensionTestBase
{
  public:
    virtual ~NvJpegExtTestBase() = default;

    virtual void SetUp()
    {
        ExtensionTestBase::SetUp();
        nvjpeg_extension_desc_.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg_extension_desc_.next = nullptr;
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, get_nvjpeg_extension_desc(&nvjpeg_extension_desc_));
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionCreate(instance_, &nvjpeg_extension_, &nvjpeg_extension_desc_));

        image_info_ = {NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        jpeg_info_ = {NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, 0};
        image_info_.next = &jpeg_info_;
    }

    virtual void TearDown()
    {
        TearDownCodecResources();
        ASSERT_EQ(NVIMGCDCS_STATUS_SUCCESS, nvimgcdcsExtensionDestroy(nvjpeg_extension_));
        ExtensionTestBase::TearDown();
    }

    nvimgcdcsExtensionDesc_t nvjpeg_extension_desc_{};
    nvimgcdcsExtension_t nvjpeg_extension_;
    nvimgcdcsJpegImageInfo_t jpeg_info_;
};

constexpr bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

constexpr int format_to_num_components(nvjpegOutputFormat_t format, int num_planes)
{
    switch (format) {
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI:
    case NVJPEG_OUTPUT_BGR:
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_YUV:
        return 3;
    case NVJPEG_OUTPUT_UNCHANGED:
        return num_planes;
    case NVJPEG_OUTPUT_Y:
        return 1;
    default:
        return 3;
    }
}

class NvJpegTestBase
{
  public:
    virtual ~NvJpegTestBase() = default;

    virtual void SetUp()
    {
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegCreateSimple(&nvjpeg_handle_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecoderCreate(nvjpeg_handle_, backend_, &nvjpeg_decoder_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decode_state_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &nvjpeg_pinned_buffer_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &nvjpeg_device_buffer_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamCreate(nvjpeg_handle_, &nvjpeg_jpeg_stream_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegStateAttachPinnedBuffer(nvjpeg_decode_state_, nvjpeg_pinned_buffer_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegStateAttachDeviceBuffer(nvjpeg_decode_state_, nvjpeg_device_buffer_));

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncoderStateCreate(nvjpeg_handle_, &nvjpeg_encode_state_, NULL));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncoderParamsCreate(nvjpeg_handle_, &nvjpeg_encode_params_, NULL));
    }

    virtual void TearDown()
    {
        if (nvjpeg_encode_params_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncoderParamsDestroy(nvjpeg_encode_params_));
            nvjpeg_encode_params_ = nullptr;
        }
        if (nvjpeg_encode_state_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncoderStateDestroy(nvjpeg_encode_state_));
            nvjpeg_encode_state_ = nullptr;
        }

        if (nvjpeg_decode_params_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsDestroy(nvjpeg_decode_params_));
            nvjpeg_decode_params_ = nullptr;
        }
        if (nvjpeg_jpeg_stream_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamDestroy(nvjpeg_jpeg_stream_));
            nvjpeg_jpeg_stream_ = nullptr;
        }
        if (nvjpeg_pinned_buffer_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferPinnedDestroy(nvjpeg_pinned_buffer_));
            nvjpeg_pinned_buffer_ = nullptr;
        }
        if (nvjpeg_device_buffer_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegBufferDeviceDestroy(nvjpeg_device_buffer_));
            nvjpeg_device_buffer_ = nullptr;
        }
        if (nvjpeg_decode_state_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStateDestroy(nvjpeg_decode_state_));
            nvjpeg_decode_state_ = nullptr;
        }
        if (nvjpeg_decoder_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecoderDestroy(nvjpeg_decoder_));
            nvjpeg_decoder_ = nullptr;
        }
        if (nvjpeg_handle_) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDestroy(nvjpeg_handle_));
            nvjpeg_handle_ = nullptr;
        }
    };

    virtual void DecodeReference(const std::string& resources_dir, const std::string& file_name, nvimgcdcsSampleFormat_t output_format,
        bool enable_color_convert, nvimgcdcsImageInfo_t* cs_image_info = nullptr)
    {
        std::string file_path(resources_dir + '/' + file_name);
        auto nvimgcdcs_to_nvjpeg_format = [](nvimgcdcsSampleFormat_t nvimgcdcs_format) -> nvjpegOutputFormat_t {
            switch (nvimgcdcs_format) {
            case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
                return NVJPEG_OUTPUT_UNCHANGED;
            case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:
                return NVJPEG_OUTPUT_UNCHANGED;
            case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
                return NVJPEG_OUTPUT_RGB;
            case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
                return NVJPEG_OUTPUT_RGBI;
            case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
                return NVJPEG_OUTPUT_BGR;
            case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
                return NVJPEG_OUTPUT_BGRI;
            case NVIMGCDCS_SAMPLEFORMAT_P_Y:
                return NVJPEG_OUTPUT_Y;
            case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
                return NVJPEG_OUTPUT_YUV;
            default:
                return NVJPEG_OUTPUT_UNCHANGED;
            }
        };

        nvjpegOutputFormat_t jpeg_output_format = nvimgcdcs_to_nvjpeg_format(output_format);
        std::ifstream input_stream(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        ASSERT_EQ(true, input_stream.is_open());
        std::streamsize file_size = input_stream.tellg();
        input_stream.seekg(0, std::ios::beg);
        std::vector<unsigned char> compressed_buffer(file_size);
        input_stream.read(reinterpret_cast<char*>(compressed_buffer.data()), file_size);
        ASSERT_EQ(true, input_stream.good());
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegJpegStreamParse(nvjpeg_handle_, compressed_buffer.data(), static_cast<size_t>(file_size), 0, 0, nvjpeg_jpeg_stream_));

        unsigned int nComponent = 0;
        nvjpegChromaSubsampling_t subsampling;
        unsigned int frame_width, frame_height;
        unsigned int widths[NVJPEG_MAX_COMPONENT];
        unsigned int heights[NVJPEG_MAX_COMPONENT];

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetFrameDimensions(nvjpeg_jpeg_stream_, &frame_width, &frame_height));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetComponentsNum(nvjpeg_jpeg_stream_, &nComponent));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetChromaSubsampling(nvjpeg_jpeg_stream_, &subsampling));
        for (unsigned int i = 0; i < nComponent; i++) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetComponentDimensions(nvjpeg_jpeg_stream_, i, &widths[i], &heights[i]));
        }
        nvjpegExifOrientation_t orientation_flag = NVJPEG_ORIENTATION_UNKNOWN;
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegJpegStreamGetExifOrientation(nvjpeg_jpeg_stream_, &orientation_flag));

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsSetExifOrientation(nvjpeg_decode_params_, orientation_flag));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params_, jpeg_output_format));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegDecodeParamsSetAllowCMYK(nvjpeg_decode_params_, enable_color_convert));

        if (orientation_flag >= NVJPEG_ORIENTATION_TRANSPOSE) {
            std::swap(frame_width, frame_height);
        }
        if (cs_image_info) {
            cs_image_info->plane_info[0].width = frame_width;
            cs_image_info->plane_info[0].height = frame_height;
        }

        unsigned int output_format_num_components = format_to_num_components(jpeg_output_format, nComponent);

        unsigned char* pBuffer = NULL;
        size_t buffer_size = frame_width * frame_height * output_format_num_components;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&pBuffer), buffer_size));
        nvjpegImage_t imgdesc;
        auto plane_buf = pBuffer;
        for (unsigned int i = 0; i < output_format_num_components; i++) {
            imgdesc.channel[i] = plane_buf;
            imgdesc.pitch[i] = (unsigned int)(is_interleaved(jpeg_output_format) ? frame_width * 3 : frame_width);
            plane_buf += frame_width * frame_height;
        }

        cudaDeviceSynchronize();
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegDecodeJpegHost(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decode_state_, nvjpeg_decode_params_, nvjpeg_jpeg_stream_));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decode_state_, nvjpeg_jpeg_stream_, NULL));
        nvjpegDecodeJpegDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decode_state_, &imgdesc, NULL);
        cudaDeviceSynchronize();

        ref_buffer_.resize(buffer_size);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(pBuffer), buffer_size,
                                   ::cudaMemcpyDeviceToHost));

        cudaFree(pBuffer);
    }

    virtual void EncodeReference(const nvimgcdcsImageInfo_t& input_image_info, const nvimgcdcsEncodeParams_t& params,
        const nvimgcdcsJpegEncodeParams_t& jpeg_enc_params, const nvimgcdcsImageInfo_t& output_image_info,
        const nvimgcdcsJpegImageInfo_t& out_jpeg_image_info, std::vector<unsigned char>* out_buffer)
    {
        auto nvimgcdcs2nvjpeg_css = [](nvimgcdcsChromaSubsampling_t nvimgcdcs_css) -> nvjpegChromaSubsampling_t {
            switch (nvimgcdcs_css) {
            case NVIMGCDCS_SAMPLING_UNSUPPORTED:
                return NVJPEG_CSS_UNKNOWN;
            case NVIMGCDCS_SAMPLING_444:
                return NVJPEG_CSS_444;
            case NVIMGCDCS_SAMPLING_422:
                return NVJPEG_CSS_422;
            case NVIMGCDCS_SAMPLING_420:
                return NVJPEG_CSS_420;
            case NVIMGCDCS_SAMPLING_440:
                return NVJPEG_CSS_440;
            case NVIMGCDCS_SAMPLING_411:
                return NVJPEG_CSS_411;
            case NVIMGCDCS_SAMPLING_410:
                return NVJPEG_CSS_410;
            case NVIMGCDCS_SAMPLING_GRAY:
                return NVJPEG_CSS_GRAY;
            case NVIMGCDCS_SAMPLING_410V:
                return NVJPEG_CSS_410V;
            default:
                return NVJPEG_CSS_UNKNOWN;
            }
        };

        auto nvimgcdcs2nvjpeg_encoding = [](nvimgcdcsJpegEncoding_t nvimgcdcs_encoding) -> nvjpegJpegEncoding_t {
            switch (nvimgcdcs_encoding) {
            case NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT:
                return NVJPEG_ENCODING_BASELINE_DCT;
            case NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN:
                return NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN;
            case NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN:
                return NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;
            default:
                return NVJPEG_ENCODING_UNKNOWN;
            }
        };

        auto nvimgcdcs2nvjpeg_format = [](nvimgcdcsSampleFormat_t nvimgcdcs_format) -> nvjpegInputFormat_t {
            switch (nvimgcdcs_format) {
            case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
                return NVJPEG_INPUT_RGB;
            case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
                return NVJPEG_INPUT_RGBI;
            case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
                return NVJPEG_INPUT_BGR;
            case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
                return NVJPEG_INPUT_BGRI;
            default:
                return NVJPEG_INPUT_RGB;
            }
        };

        auto is_interleaved = [](nvjpegInputFormat_t format) -> bool {
            return format == NVJPEG_INPUT_RGBI || format == NVJPEG_INPUT_BGRI;
        };

        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncoderParamsSetQuality(nvjpeg_encode_params_, params.quality, NULL));
        ASSERT_EQ(
            NVJPEG_STATUS_SUCCESS, nvjpegEncoderParamsSetOptimizedHuffman(nvjpeg_encode_params_, jpeg_enc_params.optimized_huffman, NULL));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegEncoderParamsSetSamplingFactors(nvjpeg_encode_params_, nvimgcdcs2nvjpeg_css(output_image_info.chroma_subsampling), NULL));
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
            nvjpegEncoderParamsSetEncoding(nvjpeg_encode_params_, nvimgcdcs2nvjpeg_encoding(out_jpeg_image_info.encoding), NULL));

        auto input_format = nvimgcdcs2nvjpeg_format(input_image_info.sample_format);

        unsigned char* dev_buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&dev_buffer, input_image_info.buffer_size));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dev_buffer, input_image_info.buffer, input_image_info.buffer_size, cudaMemcpyHostToDevice));

        nvjpegImage_t img_desc = {{dev_buffer, dev_buffer + input_image_info.plane_info[0].width * input_image_info.plane_info[0].height,
                                      dev_buffer + input_image_info.plane_info[0].width * input_image_info.plane_info[0].height * 2,
                                      dev_buffer + input_image_info.plane_info[0].width * input_image_info.plane_info[0].height * 3},
            {(unsigned int)(is_interleaved(input_format) ? input_image_info.plane_info[0].width * 3 : input_image_info.plane_info[0].width),
                (unsigned int)input_image_info.plane_info[0].width, (unsigned int)input_image_info.plane_info[0].width,
                (unsigned int)input_image_info.plane_info[0].width}};
        if (NVIMGCDCS_SAMPLEFORMAT_P_Y == input_image_info.sample_format || NVIMGCDCS_SAMPLEFORMAT_P_YUV == input_image_info.sample_format) {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncodeYUV(nvjpeg_handle_, nvjpeg_encode_state_, nvjpeg_encode_params_, &img_desc,
                                                 nvimgcdcs2nvjpeg_css(input_image_info.chroma_subsampling),
                                                 input_image_info.plane_info[0].width, input_image_info.plane_info[0].height, NULL));
        } else {
            ASSERT_EQ(NVJPEG_STATUS_SUCCESS,
                nvjpegEncodeImage(nvjpeg_handle_, nvjpeg_encode_state_, nvjpeg_encode_params_, &img_desc, input_format,
                    input_image_info.plane_info[0].width, input_image_info.plane_info[0].height, NULL));
        }

        size_t length;
        ASSERT_EQ(NVJPEG_STATUS_SUCCESS, nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, nvjpeg_encode_state_, NULL, &length, NULL));
        out_buffer->resize(length);
        ASSERT_EQ(
            NVJPEG_STATUS_SUCCESS, nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, nvjpeg_encode_state_, out_buffer->data(), &length, NULL));
        ASSERT_EQ(cudaSuccess, cudaFree(dev_buffer));
    }

    nvjpegBackend_t backend_ = NVJPEG_BACKEND_DEFAULT;
    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    nvjpegJpegDecoder_t nvjpeg_decoder_ = nullptr;
    nvjpegBufferPinned_t nvjpeg_pinned_buffer_ = nullptr;
    nvjpegBufferDevice_t nvjpeg_device_buffer_ = nullptr;
    nvjpegJpegState_t nvjpeg_decode_state_ = nullptr;
    nvjpegJpegStream_t nvjpeg_jpeg_stream_ = nullptr;
    nvjpegDecodeParams_t nvjpeg_decode_params_ = nullptr;
    nvjpegImage_t decoded_image_;
    std::vector<unsigned char> ref_buffer_;

    nvjpegEncoderState_t nvjpeg_encode_state_ = nullptr;
    nvjpegEncoderParams_t nvjpeg_encode_params_ = nullptr;
};
}} // namespace nvimgcdcs::test
