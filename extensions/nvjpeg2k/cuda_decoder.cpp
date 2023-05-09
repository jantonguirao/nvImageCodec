#include <nvimgcodecs.h>
#include <nvjpeg2k.h>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <nvtx3/nvtx3.hpp>
#include "log.h"

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
{}

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
            if (*result == NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED)
                continue;
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        static const std::set<nvimgcdcsColorSpec_t> supported_color_space{
            NVIMGCDCS_COLORSPEC_SRGB, NVIMGCDCS_COLORSPEC_GRAY, NVIMGCDCS_COLORSPEC_SYCC};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_YUV) {
            static const std::set<nvimgcdcsChromaSubsampling_t> supported_css{
                NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420};
            if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
        }

        static const std::set<nvimgcdcsSampleFormat_t> supported_sample_format{
            NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED,
            NVIMGCDCS_SAMPLEFORMAT_P_RGB,
            NVIMGCDCS_SAMPLEFORMAT_P_Y,
            NVIMGCDCS_SAMPLEFORMAT_P_YUV,
        };
        if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        static const std::set<nvimgcdcsSampleDataType_t> supported_sample_type{
            NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8, NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16, NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16};
        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (supported_sample_type.find(sample_type) == supported_sample_type.end()) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
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

                if (image_info.buffer_kind != NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                    NVIMGCDCS_D_LOG_ERROR("Unexpected buffer kind");
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                    return;
                }
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
                            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                            return;
                        }
                    }
                    encoded_stream_data = &parse_state->buffer_[0];
                }

                XM_CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data),
                    encoded_stream_data_size, false, false, parse_state->nvjpeg2k_stream_));

                nvjpeg2kImageInfo_t jpeg2k_info;
                XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(parse_state->nvjpeg2k_stream_, &jpeg2k_info));

                nvjpeg2kImageComponentInfo_t comp;
                XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(parse_state->nvjpeg2k_stream_, &comp, 0));
                auto height = comp.component_height;
                auto width = comp.component_width;
                auto bpp = comp.precision;
                auto num_components = jpeg2k_info.num_components;
                if(bpp > 16) {
                    NVIMGCDCS_D_LOG_ERROR("Unexpected bitdepth");
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                    return;
                }

                std::vector<unsigned char*> decode_output(num_components);
                std::vector<size_t> pitch_in_bytes(num_components);
                nvjpeg2kImage_t output_image;

                size_t bytes_per_sample;
                nvimgcdcsSampleDataType_t orig_data_type;
                if (bpp <= 8) {
                    output_image.pixel_type = NVJPEG2K_UINT8;
                    orig_data_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8;
                    bytes_per_sample = 1;
                } else if (bpp <= 16) {
                    output_image.pixel_type = NVJPEG2K_UINT16;
                    orig_data_type = NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16;
                    bytes_per_sample = 2;
                } else {
                    NVIMGCDCS_D_LOG_ERROR("bit depth not supported");
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                    return;
                }

                nvjpeg2kDecodeParams_t decode_params;
                nvjpeg2kDecodeParamsCreate(&decode_params);
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, params->enable_color_conversion));
                size_t row_nbytes;
                size_t component_nbytes;
                if (num_components != image_info.num_planes) {
                    NVIMGCDCS_D_LOG_ERROR("Unexpected number of planes");
                    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                }
                for (size_t p = 0; p < num_components; p++) {
                    if (image_info.plane_info[p].sample_type != orig_data_type) {
                        NVIMGCDCS_D_LOG_ERROR("Unexpected sample data type");
                        image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                    }
                }
                if (params->enable_roi && image_info.region.ndim > 0) {
                    auto region = image_info.region;
                    NVIMGCDCS_D_LOG_DEBUG(
                        "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
                    uint32_t roi_width = region.end[1] - region.start[1];
                    uint32_t roi_height = region.end[0] - region.start[0];
                    XM_CHECK_NVJPEG2K(
                        nvjpeg2kDecodeParamsSetDecodeArea(decode_params, region.start[1], region.end[1], region.start[0], region.end[0]));
                    for (size_t p = 0; p < num_components; p++) {
                        if (roi_height != image_info.plane_info[p].height ||
                            roi_width != image_info.plane_info[p].width) {
                            NVIMGCDCS_D_LOG_ERROR("Unexpected plane info dimensions");
                            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                        }
                    }
                    row_nbytes = roi_width * bytes_per_sample;
                    component_nbytes = roi_height * row_nbytes;
                } else {
                    for (size_t p = 0; p < num_components; p++) {
                        if (height != image_info.plane_info[p].height ||
                            width != image_info.plane_info[p].width) {
                            NVIMGCDCS_D_LOG_ERROR("Unexpected plane info dimensions");
                            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                        }
                    }
                    row_nbytes = width * bytes_per_sample;
                    component_nbytes =height * row_nbytes;
                }

                if (image_info.buffer_size < component_nbytes * num_components ||
                    image_info.num_planes != num_components) {
                    NVIMGCDCS_D_LOG_ERROR("The provided buffer can't hold the decoded image : " << num_components);
                    return;
                }

                for (uint32_t p = 0; p < num_components; ++p) {
                    decode_output[p] = device_buffer + p * component_nbytes;
                    pitch_in_bytes[p] = row_nbytes;
                }
                output_image.num_components = num_components;
                output_image.pixel_data = (void**)&decode_output[0];
                output_image.pitch_in_bytes = &pitch_in_bytes[0];

                std::unique_ptr<std::remove_pointer<nvjpeg2kDecodeParams_t>::type, decltype(&nvjpeg2kDecodeParamsDestroy)>
                    decode_params_raii(decode_params, &nvjpeg2kDecodeParamsDestroy);

                // Waits for GPU stage from previous iteration (on this thread)
                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeImage(
                    handle_, jpeg2k_state, parse_state->nvjpeg2k_stream_, decode_params_raii.get(), &output_image, t.stream_));

                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, t.event_));

                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            } catch (const std::runtime_error& e) {
                NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg2k code stream - " << e.what());
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
        });
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kDecoderPlugin::Decoder::decodeBatch()
{
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
        auto* handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        handle->decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->decode_state_batch_->samples_.push_back(
                NvJpeg2kDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->decodeBatch();
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg2k batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg2k
