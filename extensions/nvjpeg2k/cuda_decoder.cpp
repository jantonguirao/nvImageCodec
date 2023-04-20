#include <nvimgcodecs.h>
#include <nvjpeg2k.h>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cmath>
#include "log.h"
#include <nvtx3/nvtx3.hpp>

#include "cuda_decoder.h"
#include "error_handling.h"

namespace nvjpeg2k {

NvJpeg2kDecoderPlugin::NvJpeg2kDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,               // instance
          "nvjpeg2k_decoder", // id
          0x00000100,         // version
          "jpeg2k",           // codec_type
          static_create, Decoder::static_destroy, Decoder::static_get_capabilities, Decoder::static_can_decode,
          Decoder::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT}
    , framework_(framework)
{
}

nvimgcdcsDecoderDesc_t NvJpeg2kDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        (*code_stream)->getCodecName((*code_stream)->instance, codec_name);

        if (strcmp(codec_name, "jpeg2k") != 0) {
            *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }
        if (params->backends != nullptr) {
            *result = NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED;
            for (int b = 0; b < params->num_backends; ++b) {
                if (params->backends[b].use_gpu) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("jpeg2k_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if nvjpeg2k can decode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpeg2kDecoderPlugin::Decoder::Decoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id)
    : capabilities_(capabilities)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , device_id_(device_id)
{

    if (framework->device_allocator && framework->device_allocator->device_malloc && framework->device_allocator->device_free) {
        device_allocator_.device_ctx = framework->device_allocator->device_ctx;
        device_allocator_.device_malloc = framework->device_allocator->device_malloc;
        device_allocator_.device_free = framework->device_allocator->device_free;
    }

    if (framework->pinned_allocator && framework->pinned_allocator->pinned_malloc && framework->pinned_allocator->pinned_free) {
        pinned_allocator_.pinned_ctx = framework->pinned_allocator->pinned_ctx;
        pinned_allocator_.pinned_malloc = framework->pinned_allocator->pinned_malloc;
        pinned_allocator_.pinned_free = framework->pinned_allocator->pinned_free;
    }

    if (device_allocator_.device_malloc && device_allocator_.device_free && pinned_allocator_.pinned_malloc &&
        pinned_allocator_.pinned_free) {
        XM_CHECK_NVJPEG2K(nvjpeg2kCreateV2(NVJPEG2K_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, &handle_));
    } else {
        XM_CHECK_NVJPEG2K(nvjpeg2kCreateSimple(&handle_));
    }

    if (framework->device_allocator && (framework->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG2K(nvjpeg2kSetDeviceMemoryPadding(framework->device_allocator->device_mem_padding, handle_));
    }
    if (framework->pinned_allocator && (framework->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG2K(nvjpeg2kSetPinnedMemoryPadding(framework->pinned_allocator->pinned_mem_padding, handle_));
    }
    
    // create resources per thread
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    int num_threads = executor->get_num_threads(executor->instance);
        
    decode_state_batch_ = std::make_unique<NvJpeg2kDecoderPlugin::DecodeState>(handle_, num_threads);
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new NvJpeg2kDecoderPlugin::Decoder(capabilities_, framework_, device_id));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("jpeg2k_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin*>(instance);
        handle->create(decoder, device_id);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create nvjpeg2k decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpeg2kDecoderPlugin::Decoder::~Decoder()
{
    try {
        decode_state_batch_.reset();
        XM_CHECK_NVJPEG2K(nvjpeg2kDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg2k decoder");
    }
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("jpeg2k_destroy");
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg2k decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::static_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("jpeg2k_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve nvjpeg2k decoder capabilites " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpeg2kDecoderPlugin::DecodeState::DecodeState(nvjpeg2kHandle_t handle, int num_threads)
    : handle_(handle)
{
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &res.state_));
        res.parse_state_ = std::make_unique<NvJpeg2kDecoderPlugin::ParseState>();
    }
}

NvJpeg2kDecoderPlugin::DecodeState::~DecodeState()
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
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(res.state_));
            }
        } catch (const std::runtime_error& e) {
            NVIMGCDCS_D_LOG_ERROR("Coud not destroy jpeg2k decode state - " << e.what());
        }
    }
}

NvJpeg2kDecoderPlugin::ParseState::ParseState()
{
    XM_CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&nvjpeg2k_stream_));
}

NvJpeg2kDecoderPlugin::ParseState::~ParseState()
{
    try {
        if (nvjpeg2k_stream_) {
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(nvjpeg2k_stream_));
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not destroy nvjpeg2k stream - " << e.what());
    }
}
        
nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::decode(int sample_idx)
{
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    executor->launch(
        executor->instance, device_id_, sample_idx, decode_state_batch_.get(), [](int tid, int sample_idx, void* context) -> void {
            nvtx3::scoped_range marker{"decode " + std::to_string(sample_idx)};
            auto* decode_state = reinterpret_cast<NvJpeg2kDecoderPlugin::DecodeState*>(context);
            auto& t = decode_state->per_thread_[tid];
            auto* parse_state = t.parse_state_.get();
            auto jpeg2k_state = t.state_;
            nvimgcdcsCodeStreamDesc_t code_stream = decode_state->samples_[sample_idx].code_stream;
            nvimgcdcsImageDesc_t image = decode_state->samples_[sample_idx].image;
            const nvimgcdcsDecodeParams_t* params = decode_state->samples_[sample_idx].params;
            auto handle_ = decode_state->handle_;
            try {
                nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
                image->getImageInfo(image->instance, &image_info);
                unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

                nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
                size_t encoded_stream_data_size = 0;
                io_stream->size(io_stream->instance, &encoded_stream_data_size);
                const void* encoded_stream_data = nullptr;
                io_stream->raw_data(io_stream->instance, &encoded_stream_data);
                if (!encoded_stream_data) {
                    if (parse_state->buffer_.size() != encoded_stream_data_size) {
                        parse_state->buffer_.resize(encoded_stream_data_size);
                        io_stream->seek(io_stream->instance, 0, SEEK_SET);
                        size_t read_nbytes = 0;
                        io_stream->read(io_stream->instance, &read_nbytes, &parse_state->buffer_[0], encoded_stream_data_size);
                        if (read_nbytes != encoded_stream_data_size) {
                            NVIMGCDCS_D_LOG_ERROR("Unexpected end-of-stream");
                            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
                            return;
                        }
                    }
                    encoded_stream_data = &parse_state->buffer_[0];
                }
                
                
                XM_CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, false,
                    false, parse_state->nvjpeg2k_stream_));
                std::vector<unsigned char*> decode_output(image_info.num_planes);
                std::vector<size_t> pitch_in_bytes(image_info.num_planes);
                nvjpeg2kImage_t output_image;

                size_t bytes_per_sample;
                if (image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                    bytes_per_sample = 1;
                    output_image.pixel_type = NVJPEG2K_UINT8;
                } else if (image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16 ||
                        image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16) {
                    bytes_per_sample = 2;
                    output_image.pixel_type = NVJPEG2K_UINT16;
                } else {
                    NVIMGCDCS_D_LOG_ERROR("bit depth not supported");
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
                    return;
                }

                nvjpeg2kDecodeParams_t decode_params;
                nvjpeg2kDecodeParamsCreate(&decode_params);
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, params->enable_color_conversion));
                size_t row_nbytes;
                size_t component_nbytes;
                if (params->enable_roi && image_info.region.ndim > 0) {
                    auto region = image_info.region;
                    NVIMGCDCS_D_LOG_DEBUG(
                        "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
                    auto roi_width = region.end[1] - region.start[1];
                    auto roi_height = region.end[0] - region.start[0];
                    XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetDecodeArea(decode_params, region.start[1], region.end[1], region.start[0], region.end[0]));
                    row_nbytes = roi_width * bytes_per_sample;
                    component_nbytes = roi_height * row_nbytes;
                } else {
                    row_nbytes = image_info.plane_info[0].width * bytes_per_sample;
                    component_nbytes = image_info.plane_info[0].height * row_nbytes;
                }
                for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                    decode_output[p] = device_buffer + p * component_nbytes;
                    pitch_in_bytes[p] = row_nbytes;
                }

                output_image.num_components = image_info.num_planes; //assumed planar
                output_image.pixel_data = (void**)&decode_output[0];
                output_image.pitch_in_bytes = &pitch_in_bytes[0];

                std::unique_ptr<std::remove_pointer<nvjpeg2kDecodeParams_t>::type, decltype(&nvjpeg2kDecodeParamsDestroy)> decode_params_raii(
                    decode_params, &nvjpeg2kDecodeParamsDestroy);

                // Waits for GPU stage from previous iteration (on this thread)
                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));
                
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeImage(
                    handle_, jpeg2k_state, parse_state->nvjpeg2k_stream_, decode_params_raii.get(), &output_image, t.stream_));

                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, t.event_));

                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            } catch (const std::runtime_error& e) {
                NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg2k code stream - " << e.what());
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
            }
    });
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::decodeBatch()
{
    //NVTX3_FUNC_RANGE();
    
    /*
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
    for (auto& elem : sample_meta) {
        int sample_idx = std::get<2>(elem);
        this->decode(sample_idx);
    }
    */
    int batch_size = decode_state_batch_->samples_.size();
    for (int i = 0; i < batch_size; i++) {
        this->decode(i);
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{

    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg2k_decode_batch");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        auto *handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        handle->decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->decode_state_batch_->samples_.push_back(
                NvJpeg2kDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->decodeBatch();
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg2k batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg2k
