#include "cuda_decoder.h"
#include <nvimgcodecs.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <sstream>
#include <vector>
#include <library_types.h>
#include "errors_handling.h"
#include "log.h"
#include "parser.h"
#include "type_convert.h"

namespace nvjpeg {

NvJpegCudaDecoderPlugin::NvJpegCudaDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,                  // instance
          "nvjpeg_cuda_decoder", // id
          0x00000100,            // version
          "jpeg",                // codec_type
          static_create, Decoder::static_destroy, Decoder::static_get_capabilities, Decoder::static_can_decode,
          Decoder::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_OUTPUT}
    , framework_(framework)
{
}

nvimgcdcsDecoderDesc_t NvJpegCudaDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
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
                if (params->backends[b].use_gpu) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
            if (*result == NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED)
                continue;
        }
        nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

        nvimgcdcsJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(cs_image_info.next);
        while (jpeg_image_info && jpeg_image_info->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcdcsJpegImageInfo_t*>(jpeg_image_info->next);
        if (jpeg_image_info) {
            static const std::set<nvimgcdcsJpegEncoding_t> supported_encoding{NVIMGCDCS_JPEG_ENCODING_BASELINE_DCT,
                NVIMGCDCS_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN, NVIMGCDCS_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN,
                NVIMGCDCS_JPEG_ENCODING_LOSSLESS_HUFFMAN};
            if (supported_encoding.find(jpeg_image_info->encoding) == supported_encoding.end()) {
                *result = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                continue;
            }
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        static const std::set<nvimgcdcsColorSpec_t> supported_color_space{NVIMGCDCS_COLORSPEC_SRGB, NVIMGCDCS_COLORSPEC_GRAY,
            NVIMGCDCS_COLORSPEC_SYCC, NVIMGCDCS_COLORSPEC_CMYK, NVIMGCDCS_COLORSPEC_YCCK};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        static const std::set<nvimgcdcsChromaSubsampling_t> supported_css{NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_422,
            NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_440, NVIMGCDCS_SAMPLING_411, NVIMGCDCS_SAMPLING_410, NVIMGCDCS_SAMPLING_GRAY,
            NVIMGCDCS_SAMPLING_410V};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        static const std::set<nvimgcdcsSampleFormat_t> supported_sample_format{
            NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED,
            NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCDCS_SAMPLEFORMAT_P_RGB,
            NVIMGCDCS_SAMPLEFORMAT_I_RGB,
            NVIMGCDCS_SAMPLEFORMAT_P_BGR,
            NVIMGCDCS_SAMPLEFORMAT_I_BGR,
            NVIMGCDCS_SAMPLEFORMAT_P_Y,
            NVIMGCDCS_SAMPLEFORMAT_P_YUV,
        };
        if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if ((sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) ||
                ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED) && (sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16))) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if nvjpeg can decode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegCudaDecoderPlugin::Decoder::Decoder(
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

    decode_state_batch_ =
        std::make_unique<NvJpegCudaDecoderPlugin::DecodeState>(handle_, &device_allocator_, &pinned_allocator_, num_threads);
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new NvJpegCudaDecoderPlugin::Decoder(capabilities_, framework_, device_id));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        NvJpegCudaDecoderPlugin* handle = reinterpret_cast<NvJpegCudaDecoderPlugin*>(instance);
        handle->create(decoder, device_id);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create nvjpeg decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpegCudaDecoderPlugin::Decoder::~Decoder()
{
    try {
        decode_state_batch_.reset();
        XM_CHECK_NVJPEG(nvjpegDestroy(handle_));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg decoder - " << e.what());
    }
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_destroy");
        XM_CHECK_NULL(decoder);
        NvJpegCudaDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy nvjpeg decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::static_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("nvjpeg_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        NvJpegCudaDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve nvjpeg decoder capabilites - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpegCudaDecoderPlugin::DecodeState::DecodeState(
    nvjpegHandle_t handle, nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads)
    : handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
{
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        const int npages = res.pages_.size();
        for (int page_idx = 0; page_idx < npages; page_idx++) {
            auto& p = res.pages_[page_idx];
            for (auto backend : {NVJPEG_BACKEND_HYBRID, NVJPEG_BACKEND_GPU_HYBRID}) {
                auto& decoder = p.decoder_data[backend].decoder;
                auto& state = p.decoder_data[backend].state;
                XM_CHECK_NVJPEG(nvjpegDecoderCreate(handle_, backend, &decoder));
                XM_CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder, &state));
            }
            if (pinned_allocator_ && pinned_allocator_->pinned_malloc && pinned_allocator_->pinned_free) {
                XM_CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(handle, pinned_allocator_, &p.pinned_buffer_));
            } else {
                XM_CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle, nullptr, &p.pinned_buffer_));
            }
            XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &p.parse_state_.nvjpeg_stream_));
        }
        if (device_allocator_ && device_allocator_->dev_malloc && device_allocator_->dev_free) {
            XM_CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(handle, device_allocator_, &res.device_buffer_));
        } else {
            XM_CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle, nullptr, &res.device_buffer_));
        }
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
    }
}

NvJpegCudaDecoderPlugin::DecodeState::~DecodeState()
{
    for (auto& res : per_thread_) {
        const int npages = res.pages_.size();
        for (int page_idx = 0; page_idx < npages; page_idx++) {
            auto& p = res.pages_[page_idx];
            try {
                if (p.parse_state_.nvjpeg_stream_) {
                    XM_CHECK_NVJPEG(nvjpegJpegStreamDestroy(p.parse_state_.nvjpeg_stream_));
                }
                if (p.pinned_buffer_) {
                    XM_CHECK_NVJPEG(nvjpegBufferPinnedDestroy(p.pinned_buffer_));
                }
                for (auto& decoder_data : p.decoder_data) {
                    if (decoder_data.state) {
                        XM_CHECK_NVJPEG(nvjpegJpegStateDestroy(decoder_data.state));
                    }
                    if (decoder_data.decoder) {
                        XM_CHECK_NVJPEG(nvjpegDecoderDestroy(decoder_data.decoder));
                    }
                }
            } catch (const std::runtime_error& e) {
                NVIMGCDCS_D_LOG_ERROR("Error destroying decode state - " << e.what());
            }
        }
        try {
            if (res.device_buffer_) {
                XM_CHECK_NVJPEG(nvjpegBufferDeviceDestroy(res.device_buffer_));
            }
            if (res.event_) {
                XM_CHECK_CUDA(cudaEventDestroy(res.event_));
            }
            if (res.stream_) {
                XM_CHECK_CUDA(cudaStreamDestroy(res.stream_));
            }
        } catch (const std::runtime_error& e) {
            NVIMGCDCS_D_LOG_ERROR("Error destroying decode state - " << e.what());
        }
    }
    per_thread_.clear();
    samples_.clear();
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::decode(int sample_idx)
{
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    executor->launch(
        executor->instance, device_id_, sample_idx, decode_state_batch_.get(), [](int tid, int sample_idx, void* context) -> void {
            nvtx3::scoped_range marker{"decode " + std::to_string(sample_idx)};
            auto* decode_state = reinterpret_cast<NvJpegCudaDecoderPlugin::DecodeState*>(context);
            nvimgcdcsCodeStreamDesc_t code_stream = decode_state->samples_[sample_idx].code_stream;
            nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
            nvimgcdcsImageDesc_t image = decode_state->samples_[sample_idx].image;
            const nvimgcdcsDecodeParams_t* params = decode_state->samples_[sample_idx].params;
            auto& handle = decode_state->handle_;
            auto& t = decode_state->per_thread_[tid];
            t.current_page_idx = (t.current_page_idx + 1) % 2;
            int page_idx = t.current_page_idx;
            auto& p = t.pages_[page_idx];
            try {
                nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
                image->getImageInfo(image->instance, &image_info);
                unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

                nvjpegDecodeParams_t nvjpeg_params_;
                XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &nvjpeg_params_));
                std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params(
                    nvjpeg_params_, &nvjpegDecodeParamsDestroy);
                nvjpegOutputFormat_t nvjpeg_format = nvimgcdcs_to_nvjpeg_format(image_info.sample_format);
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_params.get(), nvjpeg_format));
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetAllowCMYK(nvjpeg_params.get(), params->enable_color_conversion));

                if (params->enable_orientation) {
                    nvjpegExifOrientation_t orientation = nvimgcdcs_to_nvjpeg_orientation(image_info.orientation);

                    // This is a workaround for a known bug in nvjpeg.
                    int major, minor, ver;
                    if (NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MAJOR_VERSION, &major) &&
                        NVJPEG_STATUS_SUCCESS == nvjpegGetProperty(MINOR_VERSION, &minor)) {
                        ver = major * 1000 + minor;
                        if (ver < 12001) {  // TODO(janton): double check the version that includes the fix
                            if (orientation == NVJPEG_ORIENTATION_ROTATE_90)
                                orientation = NVJPEG_ORIENTATION_ROTATE_270;
                            else if (orientation == NVJPEG_ORIENTATION_ROTATE_270)
                                orientation = NVJPEG_ORIENTATION_ROTATE_90;
                        }
                    }

                    if (orientation == NVJPEG_ORIENTATION_UNKNOWN) {
                        image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                        return;
                    }
                    NVIMGCDCS_D_LOG_DEBUG("Setting up EXIF orientation " << orientation);
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetExifOrientation(nvjpeg_params.get(), orientation));
                }

                if (params->enable_roi && image_info.region.ndim > 0) {
                    auto region = image_info.region;
                    NVIMGCDCS_D_LOG_DEBUG(
                        "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
                    auto roi_width = region.end[1] - region.start[1];
                    auto roi_height = region.end[0] - region.start[0];
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), region.start[1], region.start[0], roi_width, roi_height));
                } else {
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), 0, 0, -1, -1));
                }

                size_t encoded_stream_data_size = 0;
                io_stream->size(io_stream->instance, &encoded_stream_data_size);
                const void* encoded_stream_data = nullptr;
                io_stream->raw_data(io_stream->instance, &encoded_stream_data);
                if (!encoded_stream_data) {
                    if (p.parse_state_.buffer_.size() != encoded_stream_data_size) {
                        p.parse_state_.buffer_.resize(encoded_stream_data_size);
                        io_stream->seek(io_stream->instance, 0, SEEK_SET);
                        size_t read_nbytes = 0;
                        io_stream->read(io_stream->instance, &read_nbytes, &p.parse_state_.buffer_[0], encoded_stream_data_size);
                        if (read_nbytes != encoded_stream_data_size) {
                            NVIMGCDCS_P_LOG_ERROR("Unexpected end-of-stream");
                            image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                        }
                    }
                    encoded_stream_data = &p.parse_state_.buffer_[0];
                }
                XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle, static_cast<const unsigned char*>(encoded_stream_data),
                    encoded_stream_data_size, false, false, p.parse_state_.nvjpeg_stream_));

                nvjpegJpegEncoding_t jpeg_encoding;
                nvjpegJpegStreamGetJpegEncoding(p.parse_state_.nvjpeg_stream_, &jpeg_encoding);

                int is_gpu_hybrid_supported = -1; // zero means is supported
                if (jpeg_encoding == NVJPEG_ENCODING_BASELINE_DCT) { //gpu hybrid is not supported for progressive
                    XM_CHECK_NVJPEG(nvjpegDecoderJpegSupported(p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID].decoder,
                        p.parse_state_.nvjpeg_stream_, nvjpeg_params.get(), &is_gpu_hybrid_supported));
                }
    
                auto& decoder_data =
                    (image_info.plane_info[0].height * image_info.plane_info[0].width) > (256u * 256u) && is_gpu_hybrid_supported == 0
                        ? p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID]
                        : p.decoder_data[NVJPEG_BACKEND_HYBRID];
                auto& decoder = decoder_data.decoder;
                auto& state = decoder_data.state;

                XM_CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(state, p.pinned_buffer_));

                XM_CHECK_NVJPEG(nvjpegDecodeJpegHost(handle, decoder, state, nvjpeg_params.get(), p.parse_state_.nvjpeg_stream_));

                nvjpegImage_t nvjpeg_image;
                unsigned char* ptr = device_buffer;
                for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                    nvjpeg_image.channel[c] = ptr;
                    nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
                    ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
                }
                // Waits for GPU stage from previous iteration (on this thread)
                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

                XM_CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(state, t.device_buffer_));
                XM_CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle, decoder, state, p.parse_state_.nvjpeg_stream_, t.stream_));
                XM_CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle, decoder, state, &nvjpeg_image, t.stream_));

                // this captures the state of t.stream_ in the cuda event t.event_
                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                // this is so that any post processing on image waits for t.event_ i.e. decoding to finish,
                // without this the post-processing tasks such as encoding, would not know that deocding has finished on this
                // particular image
                XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, t.event_));

                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            } catch (const std::runtime_error& e) {
                NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg code stream - " << e.what());
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
        });
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::decodeBatch()
{
    NVTX3_FUNC_RANGE();
    nvjpegDecodeParams_t nvjpeg_params;
    XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &nvjpeg_params));
    std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params_raii(
        nvjpeg_params, &nvjpegDecodeParamsDestroy);

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
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpegCudaDecoderPlugin::Decoder::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
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
        NvJpegCudaDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        handle->decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->decode_state_batch_->samples_.push_back(
                NvJpegCudaDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->decodeBatch();
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}
} // namespace nvjpeg
