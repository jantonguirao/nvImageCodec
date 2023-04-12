#include "hw_decoder.h"

#include <nvimgcodecs.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include "errors_handling.h"
#include "log.h"
#include "parser.h"
#include "type_convert.h"

namespace nvjpeg {

NvJpegHwDecoderPlugin::NvJpegHwDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,                // instance
          "nvjpeg_hw_decoder", // id
          0x00000100,          // version
          "jpeg",              // codec_type
          static_create, Decoder::static_destroy, Decoder::static_get_capabilities, Decoder::static_can_decode,
          Decoder::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT, NVIMGCDCS_CAPABILITY_ROI, NVIMGCDCS_CAPABILITY_LAYOUT_PLANAR,
          NVIMGCDCS_CAPABILITY_LAYOUT_INTERLEAVED}
    , framework_(framework)
{
}

bool NvJpegHwDecoderPlugin::isPlatformSupported()
{
    nvjpegHandle_t handle;
    nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &handle);
    if (status == NVJPEG_STATUS_SUCCESS) {
        XM_CHECK_NVJPEG(nvjpegDestroy(handle));
        return true;
    } else {
        return false;
    }
}

nvimgcdcsDecoderDesc_t NvJpegHwDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status, nvjpegHandle_t handle,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
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
                if (params->backends[b].use_hw_eng) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
        }

        nvjpegDecodeParams_t nvjpeg_params;
        XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &nvjpeg_params));
        nvimgcdcsIoStreamDesc_t io_stream = (*code_stream)->io_stream;
        const void* encoded_stream_data = nullptr;
        size_t encoded_stream_data_size = 0;
        io_stream->raw_data(io_stream->instance, &encoded_stream_data);
        io_stream->size(io_stream->instance, &encoded_stream_data_size);

        // TODO - does this condition need to be checked again when accessing encoded_stream_data
        // in decodeBatch function?
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

        XM_CHECK_NVJPEG(nvjpegJpegStreamParseHeader(
            handle, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, parse_state_->nvjpeg_stream_));
        int isSupported = -1;
        XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupportedEx(handle, parse_state_->nvjpeg_stream_, nvjpeg_params, &isSupported));
        if (isSupported == 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
            NVIMGCDCS_D_LOG_INFO("decoding image on HW is supported");
        } else {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            NVIMGCDCS_D_LOG_INFO("decoding image on HW is NOT supported");
        }
        XM_CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_params));
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, handle->handle_, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if nvjpeg can decode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegHwDecoderPlugin::ParseState::ParseState(nvjpegHandle_t handle)
{
    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &nvjpeg_stream_));
}

NvJpegHwDecoderPlugin::ParseState::~ParseState()
{
    try {
        if (nvjpeg_stream_) {
            XM_CHECK_NVJPEG(nvjpegJpegStreamDestroy(nvjpeg_stream_));
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not destroy nvjpeg stream - " << e.what());
    }
}

NvJpegHwDecoderPlugin::Decoder::Decoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, const nvimgcdcsDecodeParams_t* params)
    : capabilities_(capabilities)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
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
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_HARDWARE, &device_allocator_, &pinned_allocator_, 0, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &handle_));
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
        std::make_unique<NvJpegHwDecoderPlugin::DecodeState>(handle_, &device_allocator_, &pinned_allocator_, num_threads);
    parse_state_ = std::make_unique<NvJpegHwDecoderPlugin::ParseState>(handle_);
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, const nvimgcdcsDecodeParams_t* params)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new NvJpegHwDecoderPlugin::Decoder(capabilities_, framework_, params));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(params);
        NvJpegHwDecoderPlugin* handle = reinterpret_cast<NvJpegHwDecoderPlugin*>(instance);
        handle->create(decoder, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create nvjpeg decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegHwDecoderPlugin::Decoder::~Decoder()
{
    try {
        decode_state_batch_.reset();
        XM_CHECK_NVJPEG(nvjpegDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg decoder - " << e.what());
    }
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_destroy");
        XM_CHECK_NULL(decoder);
        NvJpegHwDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::static_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        NvJpegHwDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve nvjpeg decoder capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegHwDecoderPlugin::DecodeState::DecodeState(
    nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads)
    : handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
{
    XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    XM_CHECK_CUDA(cudaEventCreate(&event_));
}

NvJpegHwDecoderPlugin::DecodeState::~DecodeState()
{
    try {
        if (event_)
            XM_CHECK_CUDA(cudaEventDestroy(event_));
        if (stream_)
            XM_CHECK_CUDA(cudaStreamDestroy(stream_));
        if (state_)
            XM_CHECK_NVJPEG(nvjpegJpegStateDestroy(state_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Error destroying decode state object - " << e.what());
    }
    samples_.clear();
}

nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::decodeBatch()
{
    NVTX3_FUNC_RANGE();
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
    int nsamples = decode_state_batch_->samples_.size();
    using sort_elem_t = std::tuple<uint32_t, uint64_t, int>;
    std::vector<sort_elem_t> sample_meta(nsamples);
    for (int i = 0; i < nsamples; i++) {
        nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[i].image;
        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        image->getImageInfo(image->instance, &image_info);
        uint64_t area = image_info.plane_info[0].height * image_info.plane_info[0].width;
        sample_meta[i] = sort_elem_t{subsampling_score(image_info.chroma_subsampling), area, i};
    }
    auto order = [](const sort_elem_t& lhs, const sort_elem_t& rhs) { return lhs > rhs; };
    std::sort(sample_meta.begin(), sample_meta.end(), order);

    std::vector<const unsigned char*> batched_bitstreams;
    std::vector<size_t> batched_bitstreams_size;
    std::vector<nvjpegImage_t> batched_output;
    std::vector<nvimgcdcsImageInfo_t> batched_image_info;

    nvjpegOutputFormat_t nvjpeg_format;

    for (auto& elem : sample_meta) {
        int sample_idx = std::get<2>(elem);
        nvimgcdcsCodeStreamDesc_t code_stream = decode_state_batch_->samples_[sample_idx].code_stream;
        nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
        nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[sample_idx].image;

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        image->getImageInfo(image->instance, &image_info);
        unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        // get output image
        nvjpegImage_t nvjpeg_image;
        unsigned char* ptr = device_buffer;
        for (uint32_t c = 0; c < image_info.num_planes; ++c) {
            nvjpeg_image.channel[c] = ptr;
            nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
            ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
        }

        nvjpeg_format = nvimgcdcs_to_nvjpeg_format(image_info.sample_format);

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

        for (auto& elem : sample_meta) {
            int sample_idx = std::get<2>(elem);
            nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[sample_idx].image;
            nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
            image->getImageInfo(image->instance, &image_info);
            XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, decode_state_batch_->event_));
            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg code stream - " << e.what());
        for (auto& elem : sample_meta) {
            int sample_idx = std::get<2>(elem);
            nvimgcdcsImageDesc_t image = decode_state_batch_->samples_[sample_idx].image;
            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}
nvimgcdcsStatus_t NvJpegHwDecoderPlugin::Decoder::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
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
        NvJpegHwDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        handle->decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->decode_state_batch_->samples_.push_back(
                NvJpegHwDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->decodeBatch();
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}
} // namespace nvjpeg
