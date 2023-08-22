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

#if WITH_DYNAMIC_NVJPEG_ENABLED
    #include "dynlink/dynlink_nvjpeg.h"
#else
    #define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

NvJpegLosslessDecoderPlugin::NvJpegLosslessDecoderPlugin(const nvimgcdcsFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL, this, plugin_id_, "jpeg", NVIMGCDCS_BACKEND_KIND_HYBRID_CPU_GPU,
          static_create, Decoder::static_destroy, Decoder::static_can_decode, Decoder::static_decode_batch}
    , framework_(framework)
{
}

bool NvJpegLosslessDecoderPlugin::isPlatformSupported()
{
    return true;
}

nvimgcdcsDecoderDesc_t* NvJpegLosslessDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);

        *status = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        nvimgcdcsImageInfo_t jpeg_info{NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO, nullptr};
        nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, static_cast<void*>(&jpeg_info)};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg") != 0) {
            *status = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            return NVIMGCDCS_STATUS_SUCCESS;
        }

        nvimgcdcsJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(cs_image_info.next);
        while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
        if (jpeg_image_info) {
            if (jpeg_image_info->encoding != NVIMGCDCS_JPEG_ENCODING_LOSSLESS_HUFFMAN) {
                *status = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                return NVIMGCDCS_STATUS_SUCCESS;
            }
        } else {
            *status = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
            return NVIMGCDCS_STATUS_SUCCESS;
        }
        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, nullptr};
        image->getImageInfo(image->instance, &image_info);

        if (image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_444) {
            *status |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED)
            *status |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;

        if (!(image_info.num_planes <= 2 && image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16))
            *status |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;

        nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t encoded_stream_data_size = 0;
        io_stream->size(io_stream->instance, &encoded_stream_data_size);

        void* encoded_stream_data = nullptr;
        void* mapped_encoded_stream_data = nullptr;
        io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size);

        if (!mapped_encoded_stream_data) {
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            size_t read_nbytes = 0;
            io_stream->read(io_stream->instance, &read_nbytes, &parse_state_->buffer_[0], encoded_stream_data_size);
            if (read_nbytes != encoded_stream_data_size) {
                NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Unexpected end-of-stream");
                return NVIMGCDCS_STATUS_BAD_CODESTREAM;
            }
            encoded_stream_data = &parse_state_->buffer_[0];
        } else {
            encoded_stream_data = mapped_encoded_stream_data;
        }

        XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size,
            0, 0, parse_state_->nvjpeg_stream_));
        int isSupported = -1;
        XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, parse_state_->nvjpeg_stream_, &isSupported));
        if (isSupported == 0) {
            *status = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
            NVIMGCDCS_LOG_INFO(framework_, plugin_id_, "decoding this lossless jpeg image is supported");
        } else {
            *status = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            NVIMGCDCS_LOG_INFO(framework_, plugin_id_, "decoding this lossless jpeg image is NOT supported");
        }
        if (mapped_encoded_stream_data) {
            io_stream->unmap(io_stream->instance, &mapped_encoded_stream_data, encoded_stream_data_size);
        }
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if lossless nvjpeg can decode - " << e.info());
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        nvtx3::scoped_range marker{"nvjpeg_lossless_can_decode"};
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        if (batch_size < (num_threads + 1)) {  // not worth parallelizing
            for (int i = 0; i < batch_size; i++)
                canDecode(&status[i], code_streams[i], images[i], params);
        } else {
            int num_blocks = num_threads + 1;  // the last block is processed in the current thread
            CanDecodeCtx canDecodeCtx{this, status, code_streams, images, params, batch_size, num_blocks};
            canDecodeCtx.promise.resize(num_threads);
            std::vector<std::future<void>> fut;
            fut.reserve(num_threads);
            for (auto& pr : canDecodeCtx.promise)
                fut.push_back(pr.get_future());
            auto task = [](int tid, int block_idx, void* context) -> void {
                auto* ctx = reinterpret_cast<CanDecodeCtx*>(context);
                int64_t i_start = ctx->num_samples * block_idx / ctx->num_blocks;
                int64_t i_end = ctx->num_samples * (block_idx + 1) / ctx->num_blocks;
                for (int i = i_start; i < i_end; i++) {
                    ctx->this_ptr->canDecode(&ctx->status[i], ctx->code_streams[i], ctx->images[i], ctx->params);
                }
                if (block_idx < static_cast<int>(ctx->promise.size()))
                    ctx->promise[block_idx].set_value();
            };
            int block_idx = 0;
            for (; block_idx < num_threads; ++block_idx) {
                executor->launch(executor->instance, exec_params_->device_id, block_idx, &canDecodeCtx, task);
            }
            task(-1, block_idx, &canDecodeCtx);

            // wait for it to finish
            for (auto& f : fut)
                f.wait();
        }
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if lossless nvjpeg can decode - " << e.info());
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcdcsStatus();
    }
}

NvJpegLosslessDecoderPlugin::ParseState::ParseState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, nvjpegHandle_t handle)
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
    const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
{
    bool use_nvjpeg_create_ex_v2 = false;
    if (nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
        if (exec_params->device_allocator && exec_params->device_allocator->device_malloc && exec_params->device_allocator->device_free) {
            device_allocator_.dev_ctx = exec_params->device_allocator->device_ctx;
            device_allocator_.dev_malloc = exec_params->device_allocator->device_malloc;
            device_allocator_.dev_free = exec_params->device_allocator->device_free;
        }

        if (exec_params->pinned_allocator && exec_params->pinned_allocator->pinned_malloc && exec_params->pinned_allocator->pinned_free) {
            pinned_allocator_.pinned_ctx = exec_params->pinned_allocator->pinned_ctx;
            pinned_allocator_.pinned_malloc = exec_params->pinned_allocator->pinned_malloc;
            pinned_allocator_.pinned_free = exec_params->pinned_allocator->pinned_free;
        }
        use_nvjpeg_create_ex_v2 =
            device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free;
    }

    unsigned int nvjpeg_flags = get_nvjpeg_flags("nvjpeg_lossless_decoder", options);
    if (use_nvjpeg_create_ex_v2) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_LOSSLESS_JPEG, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_LOSSLESS_JPEG, nullptr, nullptr, nvjpeg_flags, &handle_));
    }

    if (exec_params->device_allocator && (exec_params->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(exec_params->device_allocator->device_mem_padding, handle_));
    }
    if (exec_params->pinned_allocator && (exec_params->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(exec_params->pinned_allocator->pinned_mem_padding, handle_));
    }

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    decode_state_batch_ = std::make_unique<NvJpegLosslessDecoderPlugin::DecodeState>(
        plugin_id_, framework_, handle_, &device_allocator_, &pinned_allocator_, num_threads);
    parse_state_ = std::make_unique<NvJpegLosslessDecoderPlugin::ParseState>(plugin_id_, framework_, handle_);
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::create(
    nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCDCS_DEVICE_CPU_ONLY)
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        *decoder =
            reinterpret_cast<nvimgcdcsDecoder_t>(new NvJpegLosslessDecoderPlugin::Decoder(plugin_id_, framework_, exec_params, options));
        return NVIMGCDCS_STATUS_SUCCESS;
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg lossless decoder - " << e.info());
        return e.nvimgcdcsStatus();
    }
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::static_create(
    void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        NvJpegLosslessDecoderPlugin* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    } catch (const NvJpegException& e) {
        return e.nvimgcdcsStatus();
    }
}

NvJpegLosslessDecoderPlugin::Decoder::~Decoder()
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_destroy");
        parse_state_.reset();
        decode_state_batch_.reset();
        if (handle_)
            XM_NVJPEG_D_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg lossless decoder - " << e.info());
    }
}

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const NvJpegException& e) {
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegLosslessDecoderPlugin::DecodeState::DecodeState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
    nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , handle_(handle)
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

nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::decodeBatch(
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVTX3_FUNC_RANGE();
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            decode_state_batch_->samples_.push_back(
                NvJpegLosslessDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }

        int nsamples = decode_state_batch_->samples_.size();

        std::vector<const unsigned char*> batched_bitstreams;
        std::vector<size_t> batched_bitstreams_size;
        std::vector<nvjpegImage_t> batched_output;
        std::vector<nvimgcdcsImageInfo_t> batched_image_info;

        nvjpegOutputFormat_t nvjpeg_format;

        std::vector<int> sample_idxs;
        sample_idxs.reserve(nsamples);

        std::set<cudaStream_t> sync_streams;

        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
            nvimgcdcsCodeStreamDesc_t* code_stream = decode_state_batch_->samples_[sample_idx].code_stream;
            nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, nullptr};
            code_stream->getImageInfo(code_stream->instance, &cs_image_info);

            nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
            nvimgcdcsImageDesc_t* image = decode_state_batch_->samples_[sample_idx].image;

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

            if ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED ||
                    (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y && image_info.num_planes == 1)) &&
                image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16) {
                nvjpeg_format = NVJPEG_OUTPUT_UNCHANGEDI_U16;

                size_t encoded_stream_data_size = 0;
                io_stream->size(io_stream->instance, &encoded_stream_data_size);
                void* encoded_stream_data = nullptr;
                io_stream->map(io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size);
                batched_bitstreams.push_back(static_cast<const unsigned char*>(encoded_stream_data));
                batched_bitstreams_size.push_back(encoded_stream_data_size);
                batched_output.push_back(nvjpeg_image);
                batched_image_info.push_back(image_info);
                sample_idxs.push_back(sample_idx);
                sync_streams.insert(image_info.cuda_stream);
            } else {
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
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

            for (cudaStream_t stream : sync_streams) {
                XM_CHECK_CUDA(cudaStreamWaitEvent(stream, decode_state_batch_->event_));
            }

            for (size_t i = 0; i < sample_idxs.size(); i++) {
                auto sample_idx = sample_idxs[i];
                nvimgcdcsImageDesc_t* image = decode_state_batch_->samples_[sample_idx].image;
                nvimgcdcsCodeStreamDesc_t* code_stream = decode_state_batch_->samples_[sample_idx].code_stream;
                nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
                io_stream->unmap(io_stream->instance, &batched_bitstreams[i], batched_bitstreams_size[i]);
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            }
        } catch (const NvJpegException& e) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not decode lossless jpeg code stream - " << e.info());
            for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
                nvimgcdcsImageDesc_t* image = decode_state_batch_->samples_[sample_idx].image;
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
            return e.nvimgcdcsStatus();
        }
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not decode lossless jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}
nvimgcdcsStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_decode_batch(nvimgcdcsDecoder_t decoder,
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {

        return e.nvimgcdcsStatus();
    }
}
} // namespace nvjpeg
