/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "lossless_decoder.h"
#include <library_types.h>
#include <nvimgcodecs.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <vector>
#include "errors_handling.h"
#include "log.h"
#include "nvjpeg_utils.h"
#include "type_convert.h"

namespace nvjpeg {

NvJpegLosslessDecoderPlugin::NvJpegLosslessDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,                // instance
          "nvjpeg_hw_decoder", // id
          "jpeg",              // codec_type
          static_create, Decoder::static_destroy, Decoder::static_get_capabilities, Decoder::static_can_decode,
          Decoder::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT, NVIMGCDCS_CAPABILITY_ROI, NVIMGCDCS_CAPABILITY_LAYOUT_PLANAR,
          NVIMGCDCS_CAPABILITY_LAYOUT_INTERLEAVED}
    , framework_(framework)
{
}

nvimgcdcsDecoderDesc_t NvJpegLosslessDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status, nvjpegHandle_t handle, 
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg") != 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }
        if (params->backends != nullptr) {
            *result = NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED;
            for (int b = 0; b < params->num_backends; ++b) {
                if (params->backends[b].kind == NVIMGCDCS_BACKEND_KIND_GPU_ONLY) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
            if (*result == NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED)
                continue;
        }
                
        nvimgcdcsJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(cs_image_info.next);
        while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
        if (jpeg_image_info) {            
            if (jpeg_image_info->encoding != NVIMGCDCS_JPEG_ENCODING_LOSSLESS_HUFFMAN) {
                *result = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                continue;
            }
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info); // TODO this image info is for decoded output, correct? 
        
        /* TODO: Color spec is not well defined for lossless jpeg, should this be removed? */
        static const std::set<nvimgcdcsColorSpec_t> supported_color_space{NVIMGCDCS_COLORSPEC_SRGB, NVIMGCDCS_COLORSPEC_GRAY,
            NVIMGCDCS_COLORSPEC_SYCC, NVIMGCDCS_COLORSPEC_CMYK, NVIMGCDCS_COLORSPEC_YCCK};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
                
        if (image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_444) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }       

        if (image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED)
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        
        if (!(image_info.num_planes == 1 && image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16))
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        
        nvimgcdcsIoStreamDesc_t io_stream = (*code_stream)->io_stream;
        const void* encoded_stream_data = nullptr;
        size_t encoded_stream_data_size = 0;
        io_stream->raw_data(io_stream->instance, &encoded_stream_data);
        io_stream->size(io_stream->instance, &encoded_stream_data_size);

        if (!encoded_stream_data) {
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            size_t read_nbytes = 0;
            io_stream->read(io_stream->instance, &read_nbytes, &parse_state_->buffer_[0], encoded_stream_data_size);
            if (read_nbytes != encoded_stream_data_size) {
                NVIMGCDCS_P_LOG_ERROR("Unexpected end-of-stream");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }
            encoded_stream_data = &parse_state_->buffer_[0];
        }

        XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, 0, 0, parse_state_->nvjpeg_stream_));                
        int isSupported = -1;
        XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle, parse_state_->nvjpeg_stream_, &isSupported));
        if (isSupported == 0)
        {
            *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
            NVIMGCDCS_D_LOG_INFO("decoding this lossless jpeg image is supported");
        } else {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            NVIMGCDCS_D_LOG_INFO("decoding this lossless jpeg image is NOT supported");
        }                        
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, handle->handle_, code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if nvjpeg can decode - " << e.info());
        return e.nvimgcdcsStatus();
    }
}

NvJpegLosslessDecoderPlugin::ParseState::ParseState(nvjpegHandle_t handle)
{
    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &nvjpeg_stream_));
}

NvJpegLosslessDecoderPlugin::ParseState::~ParseState()
{
    if (nvjpeg_stream_) {
        XM_NVJPEG_D_LOG_DESTROY(nvjpegJpegStreamDestroy(nvjpeg_stream_));
    }
}


NvJpegLosslessDecoderPlugin::Decoder::Decoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id, const char* options)
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

    unsigned int nvjpeg_flags = get_nvjpeg_flags("nvjpeg_lossless_decoder", options);
    if (device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_LOSSLESS_JPEG, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_LOSSLESS_JPEG, nullptr, nullptr, nvjpeg_flags, &handle_));
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

    decode_state_batch_ =
        std::make_unique<NvJpegLosslessDecoderPlugin::DecodeState>(handle_, &device_allocator_, &pinned_allocator_, num_threads);
    parse_state_ = std::make_unique<NvJpegLosslessDecoderPlugin::ParseState>(handle_);
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new NvJpegLosslessDecoderPlugin::Decoder(capabilities_, framework_, device_id, options));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        if (device_id == NVIMGCDCS_DEVICE_CPU_ONLY)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        NvJpegLosslessDecoderPlugin* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin*>(instance);
        return handle->create(decoder, device_id, options);
    } catch (const NvJpegException& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create nvjpeg lossless decoder - " << e.info());
        return e.nvimgcdcsStatus();
    }
}

NvJpegLosslessDecoderPlugin::Decoder::~Decoder()
{
    parse_state_.reset();
    decode_state_batch_.reset();
    if (handle_)
        XM_NVJPEG_D_LOG_DESTROY(nvjpegDestroy(handle_));
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_destroy");
        XM_CHECK_NULL(decoder);
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const NvJpegException& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg lossless decoder - " << e.info());
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const NvJpegException& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve nvjpeg lossless decoder capabilites - " << e.info());
        return e.nvimgcdcsStatus();
    }
}

NvJpegLosslessDecoderPlugin::DecodeState::DecodeState(
    nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads)
    : handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
{
    XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    XM_CHECK_CUDA(cudaEventCreate(&event_));
}

NvJpegLosslessDecoderPlugin::DecodeState::~DecodeState()
{
    if (event_)
        XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
    if (stream_)
        XM_CUDA_LOG_DESTROY(cudaStreamDestroy(stream_));
    if (state_)
        XM_NVJPEG_D_LOG_DESTROY(nvjpegJpegStateDestroy(state_));
    samples_.clear();
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::decodeBatch()
{
    NVTX3_FUNC_RANGE();
    /* lossless decoder only has sampling 444
    auto subsampling_score = [](nvimgcdcsChromaSubsampling_t subsampling) -> uint32_t {
        switch (subsampling) {
        case NVIMGCDCS_SAMPLING_444:
            return 8;
        case NVIMGCDCS_SAMPLING_422:
            return 7;
        case NVIMGCDCS_SAMPLING_420:
            return 6;
        case NVIMGCDCS_SAMPLING_440:
            return 5;
        case NVIMGCDCS_SAMPLING_411:
            return 4;
        case NVIMGCDCS_SAMPLING_410:
            return 3;
        case NVIMGCDCS_SAMPLING_GRAY:
            return 2;
        case NVIMGCDCS_SAMPLING_410V:
        default:
            return 1;
        }
    };
    */
    int nsamples = decode_state_batch_->samples_.size();
    
    std::vector<const unsigned char*> batched_bitstreams;
    std::vector<size_t> batched_bitstreams_size;
    std::vector<nvjpegImage_t> batched_output;
    std::vector<nvimgcdcsImageInfo_t> batched_image_info;

    nvjpegOutputFormat_t nvjpeg_format;
    
    for (int sample_idx = 0; sample_idx < nsamples; sample_idx++)
    {
        nvimgcdcsCodeStreamDesc_t code_stream = decode_state_batch_->samples_[sample_idx].code_stream;
        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[sample_idx].image;

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        image->getImageInfo(image->instance, &image_info);
        unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);
     
        nvjpegImage_t nvjpeg_image;
        unsigned char* ptr = device_buffer;
        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            nvjpeg_image.channel[c] = ptr;
            nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
            ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
        }
                
        if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED && image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)
            nvjpeg_format = NVJPEG_OUTPUT_UNCHANGEDI_U16;
        else
            return NVIMGCDCS_EXTENSION_STATUS_IMPLEMENTATION_NOT_SUPPORTED;

        const void* encoded_stream_data = nullptr;
        size_t encoded_stream_data_size = 0;
        io_stream->raw_data(io_stream->instance, &encoded_stream_data);
        io_stream->size(io_stream->instance, &encoded_stream_data_size);

        batched_bitstreams.push_back(static_cast<const unsigned char*>(encoded_stream_data));
        batched_bitstreams_size.push_back(encoded_stream_data_size);
        batched_output.push_back(nvjpeg_image);
        batched_image_info.push_back(image_info);
    }

    try {
        auto& state = decode_state_batch_->state_;
        auto& handle = decode_state_batch_->handle_;

        if (batched_bitstreams.size() > 0) {
            XM_CHECK_CUDA(cudaEventSynchronize(decode_state_batch_->event_));

            XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle, state, batched_bitstreams.size(), 1, nvjpeg_format));

            XM_CHECK_NVJPEG(nvjpegDecodeBatched(handle, state, batched_bitstreams.data(), batched_bitstreams_size.data(),
                batched_output.data(), decode_state_batch_->stream_));

            XM_CHECK_CUDA(cudaEventRecord(decode_state_batch_->event_, decode_state_batch_->stream_));
        }

        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {            
            nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[sample_idx].image;
            nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
            image->getImageInfo(image->instance, &image_info);
            XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, decode_state_batch_->event_));
            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
        }
    } catch (const NvJpegException& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode lossless jpeg code stream - " << e.info());
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {            
            nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[sample_idx].image;
            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcdcsStatus();
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}
nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{

    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_decode_batch");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        handle->decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->decode_state_batch_->samples_.push_back(
                NvJpegLosslessDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->decodeBatch();
    } catch (const NvJpegException& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode lossless jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcdcsStatus();
    }
}
} // namespace nvjpeg
