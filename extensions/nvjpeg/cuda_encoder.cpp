/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuda_encoder.h"
#include <nvimgcodecs.h>
#include <cstring>
#include <future>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nvjpeg.h>
#include <nvtx3/nvtx3.hpp>

#include "errors_handling.h"
#include "log.h"
#include "type_convert.h"

namespace nvjpeg {

NvJpegCudaEncoderPlugin::NvJpegCudaEncoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : encoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC, NULL,
          this,             // instance
          "nvjpeg_encoder", // id
          0x00000100,       // version
          "jpeg",           // codec_type
          static_create, Encoder::static_destroy, Encoder::static_get_capabilities, Encoder::static_can_encode,
          Encoder::static_encode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_INPUT}
    , framework_(framework)
{
}

nvimgcdcsEncoderDesc_t NvJpegCudaEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        (*code_stream)->getCodecName((*code_stream)->instance, codec_name);

        if (strcmp(codec_name, "jpeg") != 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }
        if (params->backends != nullptr) {
            *result = NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED;
            for (int b = 0; b < params->num_backends; ++b) {
                if (params->backends[b].kind == NVIMGCDCS_BACKEND_KIND_HYBRID_CPU_GPU) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
        }
        nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

        nvimgcdcsJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(cs_image_info.next);
        while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
        if (jpeg_image_info) {

            static const std::set<nvimgcdcsJpegEncoding_t> supported_encoding{
                NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN};
            if (supported_encoding.find(jpeg_image_info->encoding) == supported_encoding.end()) {
                *result = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                continue;
            }
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        nvimgcdcsImageInfo_t out_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &out_image_info);

        static const std::set<nvimgcdcsColorSpec_t> supported_color_space{
            NVIMGCDCS_COLORSPEC_SRGB, NVIMGCDCS_COLORSPEC_GRAY, NVIMGCDCS_COLORSPEC_SYCC};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        static const std::set<nvimgcdcsChromaSubsampling_t> supported_css{NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_422,
            NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (supported_css.find(out_image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }

        static const std::set<nvimgcdcsSampleFormat_t> supported_sample_format{
            NVIMGCDCS_SAMPLEFORMAT_P_RGB,
            NVIMGCDCS_SAMPLEFORMAT_I_RGB,
            NVIMGCDCS_SAMPLEFORMAT_P_BGR,
            NVIMGCDCS_SAMPLEFORMAT_I_BGR,
            NVIMGCDCS_SAMPLEFORMAT_P_YUV,
            NVIMGCDCS_SAMPLEFORMAT_P_Y,
        };
        if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y) {
            if ((image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_GRAY) ||
                (out_image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_GRAY)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if ((image_info.color_spec != NVIMGCDCS_COLORSPEC_GRAY) && (image_info.color_spec != NVIMGCDCS_COLORSPEC_SYCC)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg_can_encode");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not check if nvjpge can encode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegCudaEncoderPlugin::Encoder::Encoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id)
    : capabilities_(capabilities)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , device_id_(device_id)
{
    if (framework->device_allocator && framework->device_allocator->device_malloc && framework->device_allocator->device_free) {
        device_allocator_.dev_ctx = framework->device_allocator->device_ctx;
        device_allocator_.dev_malloc = framework->device_allocator->device_malloc;
        device_allocator_.dev_free = framework->device_allocator->device_free;
    }

    if (framework->pinned_allocator && framework->pinned_allocator->pinned_malloc && framework->pinned_allocator->pinned_free) {
        pinned_allocator_.pinned_ctx = framework->pinned_allocator->pinned_ctx;
        pinned_allocator_.pinned_malloc = framework->pinned_allocator->pinned_malloc;
        pinned_allocator_.pinned_free = framework->pinned_allocator->pinned_free;
    }

    if (device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, 0, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, 0, &handle_));
    }

    if (framework->device_allocator && (framework->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(framework->device_allocator->device_mem_padding, handle_));
    }
    if (framework->pinned_allocator && (framework->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(framework->pinned_allocator->pinned_mem_padding, handle_));
    }

    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    int num_threads = executor->get_num_threads(executor->instance);

    encode_state_batch_ = std::make_unique<NvJpegCudaEncoderPlugin::EncodeState>(handle_, num_threads);
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::create(nvimgcdcsEncoder_t* encoder, int device_id)
{
    *encoder = reinterpret_cast<nvimgcdcsEncoder_t>(new NvJpegCudaEncoderPlugin::Encoder(capabilities_, framework_, device_id));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::static_create(void* instance, nvimgcdcsEncoder_t* encoder, int device_id)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg_create_encoder");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(encoder);
        NvJpegCudaEncoderPlugin* handle = reinterpret_cast<NvJpegCudaEncoderPlugin*>(instance);
        handle->create(encoder, device_id);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not create nvjpeg encoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegCudaEncoderPlugin::Encoder::~Encoder()
{
    try {
        encode_state_batch_.reset();
        XM_CHECK_NVJPEG(nvjpegDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not properly destroy nvjpeg encoder - " << e.what());
    }
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_destroy(nvimgcdcsEncoder_t encoder)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg_destroy_encoder");
        XM_CHECK_NULL(encoder);
        NvJpegCudaEncoderPlugin::Encoder* handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not properly destroy nvjpeg encoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    if (capabilities) {
        *capabilities = capabilities_.data();
    }

    if (size) {
        *size = capabilities_.size();
    } else {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_get_capabilities(
    nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg_get_capabilities");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        NvJpegCudaEncoderPlugin::Encoder* handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not retrive nvjpeg encoder capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegCudaEncoderPlugin::EncodeState::EncodeState(nvjpegHandle_t handle, int num_threads)
    : handle_(handle)
{
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG(nvjpegEncoderStateCreate(handle_, &res.state_, res.stream_));
    }
}

NvJpegCudaEncoderPlugin::EncodeState::~EncodeState()
{
    for (auto& res : per_thread_) {
        try {
            if (res.event_) {
                XM_CHECK_CUDA(cudaEventDestroy(res.event_));
            }

            if (res.stream_) {
                XM_CHECK_CUDA(cudaStreamDestroy(res.stream_));
            }

            if (res.state_) {
                XM_CHECK_NVJPEG(nvjpegEncoderStateDestroy(res.state_));
            }

        } catch (const std::runtime_error& e) {
            NVIMGCDCS_E_LOG_ERROR("Could not destroy encode state - " << e.what());
        }
    }
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::encode(int sample_idx)
{
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    executor->launch(
        executor->instance, device_id_, sample_idx, encode_state_batch_.get(), [](int tid, int sample_idx, void* task_context) -> void {
            nvtx3::scoped_range marker{"encode " + std::to_string(sample_idx)};
            auto encode_state = reinterpret_cast<NvJpegCudaEncoderPlugin::EncodeState*>(task_context);
            nvimgcdcsCodeStreamDesc_t code_stream = encode_state->samples_[sample_idx].code_stream_;
            nvimgcdcsImageDesc_t image = encode_state->samples_[sample_idx].image_;
            const nvimgcdcsEncodeParams_t* params = encode_state->samples_[sample_idx].params;
            auto& handle_ = encode_state->handle_;
            auto& t = encode_state->per_thread_[tid];
            auto& jpeg_state_ = t.state_;
            try {
                nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
                image->getImageInfo(image->instance, &image_info);

                nvimgcdcsImageInfo_t out_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
                code_stream->getImageInfo(code_stream->instance, &out_image_info);

                if (image_info.plane_info[0].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED);
                    NVIMGCDCS_E_LOG_ERROR("Unsupported sample data type. Only UINT8 is supported.");
                    return;
                }

                unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

                nvjpegEncoderParams_t encode_params_;
                XM_CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &encode_params_, t.stream_));
                std::unique_ptr<std::remove_pointer<nvjpegEncoderParams_t>::type, decltype(&nvjpegEncoderParamsDestroy)> encode_params(
                    encode_params_, &nvjpegEncoderParamsDestroy);
                int nvjpeg_format = nvimgcdcs_to_nvjpeg_format(image_info.sample_format);
                if (nvjpeg_format < 0) {
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
                    return;
                }
                nvjpegInputFormat_t input_format = static_cast<nvjpegInputFormat_t>(nvjpeg_format);

                nvjpegImage_t input_image;
                unsigned char* ptr = device_buffer;
                for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                    input_image.channel[p] = ptr;
                    input_image.pitch[p] = image_info.plane_info[p].row_stride;
                    ptr += input_image.pitch[p] * image_info.plane_info[p].height;
                }

                XM_CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encode_params.get(), static_cast<int>(params->quality), t.stream_));
                NVIMGCDCS_E_LOG_DEBUG(" - quality: " << static_cast<int>(params->quality));

                auto jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(out_image_info.next);
                while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
                    jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
                if (jpeg_image_info) {
                    nvjpegJpegEncoding_t encoding = nvimgcdcs_to_nvjpeg_encoding(jpeg_image_info->encoding);
                    NVIMGCDCS_E_LOG_DEBUG(" - encoding: " << encoding);
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encode_params.get(), encoding, t.stream_));
                }

                auto jpeg_encode_params = static_cast<nvimgcdcsJpegEncodeParams_t*>(params->next);
                while (jpeg_encode_params && jpeg_encode_params->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS)
                    jpeg_encode_params = static_cast<nvimgcdcsJpegEncodeParams_t*>(jpeg_encode_params->next);
                if (jpeg_encode_params) {
                    NVIMGCDCS_E_LOG_DEBUG(" - optimized huffman: " << jpeg_encode_params->optimized_huffman);
                    XM_CHECK_NVJPEG(
                        nvjpegEncoderParamsSetOptimizedHuffman(encode_params.get(), jpeg_encode_params->optimized_huffman, t.stream_));
                } else {
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encode_params.get(), 0, t.stream_));
                }
                nvjpegChromaSubsampling_t chroma_subsampling = nvimgcdcs_to_nvjpeg_css(out_image_info.chroma_subsampling);
                if (chroma_subsampling != NVJPEG_CSS_UNKNOWN) {
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encode_params.get(), chroma_subsampling, NULL));
                }
                if (((image_info.color_spec == NVIMGCDCS_COLORSPEC_SYCC) &&
                        ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_YUV) ||
                            (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y))) ||
                    ((image_info.color_spec == NVIMGCDCS_COLORSPEC_GRAY) && (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y))) {
                    nvjpegChromaSubsampling_t input_chroma_subsampling = nvimgcdcs_to_nvjpeg_css(image_info.chroma_subsampling);
                    XM_CHECK_NVJPEG(nvjpegEncodeYUV(handle_, jpeg_state_, encode_params.get(), &input_image, input_chroma_subsampling,
                        image_info.plane_info[0].width, image_info.plane_info[0].height, t.stream_));
                } else {
                    XM_CHECK_NVJPEG(nvjpegEncodeImage(handle_, jpeg_state_, encode_params.get(), &input_image, input_format,
                        image_info.plane_info[0].width, image_info.plane_info[0].height, t.stream_));
                }

                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

                size_t length;
                XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, jpeg_state_, NULL, &length, t.stream_));

                t.compressed_data_.resize(length);
                XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, jpeg_state_, t.compressed_data_.data(), &length, t.stream_));

                nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
                size_t output_size;
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                io_stream->write(io_stream->instance, &output_size, static_cast<void*>(&t.compressed_data_[0]), t.compressed_data_.size());

                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            } catch (const std::runtime_error& e) {
                NVIMGCDCS_D_LOG_ERROR("Could not encode jpeg code stream - " << e.what());
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
        });
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::encodeBatch()
{
    int batch_size = encode_state_batch_->samples_.size();
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        this->encode(sample_idx);
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaEncoderPlugin::Encoder::static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg_encode_batch");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_E_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        auto handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        handle->encode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->encode_state_batch_->samples_.push_back(
                NvJpegCudaEncoderPlugin::EncodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->encodeBatch();
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not encode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg
