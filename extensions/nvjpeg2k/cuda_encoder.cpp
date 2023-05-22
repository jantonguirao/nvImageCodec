
#include <nvimgcodecs.h>
#include <cstring>
#include <future>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_encoder.h"
#include "error_handling.h"
#include "log.h"
#include "nvjpeg2k.h"

namespace nvjpeg2k {

NvJpeg2kEncoderPlugin::NvJpeg2kEncoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : encoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC, NULL,
          this,               // instance
          "nvjpeg2k_encoder", // id
          0x00000100,         // version
          "jpeg2k",           // codec_type
          static_create, Encoder::static_destroy, Encoder::static_get_capabilities, Encoder::static_can_encode,
          Encoder::static_encode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_DEVICE_INPUT}
    , framework_(framework)
{
}

nvimgcdcsEncoderDesc_t NvJpeg2kEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
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
                if (params->backends[b].kind == NVIMGCDCS_BACKEND_KIND_HYBRID_CPU_GPU) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
        }

        nvimgcdcsJpeg2kEncodeParams_t* j2k_encode_params = static_cast<nvimgcdcsJpeg2kEncodeParams_t*>(params->next);
        while (j2k_encode_params && j2k_encode_params->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS)
            j2k_encode_params = static_cast<nvimgcdcsJpeg2kEncodeParams_t*>(j2k_encode_params->next);
        if (j2k_encode_params) {
            if ((j2k_encode_params->code_block_w != 32 || j2k_encode_params->code_block_h != 32) &&
                (j2k_encode_params->code_block_w != 64 || j2k_encode_params->code_block_h != 64)) {
                *result = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                NVIMGCDCS_E_LOG_WARNING("Unsupported block size: " << j2k_encode_params->code_block_w << "x"
                                                                   << j2k_encode_params->code_block_h << "(Valid values: 32, 64)");
            }
            if (j2k_encode_params->num_resolutions > NVJPEG2K_MAXRES) {
                NVIMGCDCS_E_LOG_WARNING(
                    "Unsupported number of resolutions: " << j2k_encode_params->num_resolutions << " (max = " << NVJPEG2K_MAXRES << ") ");
                *result = NVIMGCDCS_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
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
        static const std::set<nvimgcdcsChromaSubsampling_t> supported_css{
            NVIMGCDCS_SAMPLING_444, NVIMGCDCS_SAMPLING_422, NVIMGCDCS_SAMPLING_420, NVIMGCDCS_SAMPLING_GRAY};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (out_image_info.chroma_subsampling != image_info.chroma_subsampling) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
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

        if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_Y) {
            if ((image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_GRAY) ||
                (out_image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_GRAY)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if (image_info.color_spec != NVIMGCDCS_COLORSPEC_GRAY) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
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

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg2k_can_encode");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not check if nvjpeg2k can encode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpeg2kEncoderPlugin::Encoder::Encoder(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id)
    : capabilities_(capabilities)
    , framework_(framework)
    , device_id_(device_id)
{
    XM_CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&handle_));

    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    int num_threads = executor->get_num_threads(executor->instance);

    encode_state_batch_ = std::make_unique<NvJpeg2kEncoderPlugin::EncodeState>(handle_, num_threads);
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::create(nvimgcdcsEncoder_t* encoder, int device_id)
{
    *encoder = reinterpret_cast<nvimgcdcsEncoder_t>(new NvJpeg2kEncoderPlugin::Encoder(capabilities_, framework_, device_id));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::static_create(void* instance, nvimgcdcsEncoder_t* encoder, int device_id)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg2k_create_encoder");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin*>(instance);
        handle->create(encoder, device_id);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not create nvjpeg2k encoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

NvJpeg2kEncoderPlugin::Encoder::~Encoder()
{
    try {
        encode_state_batch_.reset();
        XM_CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(handle_));

    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not properly destroy nvjpeg2k encoder");
    }
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::static_destroy(nvimgcdcsEncoder_t encoder)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg2k_destroy_encoder");
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not properly destroy nvjpeg2k encoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::static_get_capabilities(
    nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("jpeg2k_get_capabilities");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not retrieve nvjpeg2k encoder capabilites " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

NvJpeg2kEncoderPlugin::EncodeState::EncodeState(nvjpeg2kEncoder_t handle, int num_threads)
    : handle_(handle)
{
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(handle_, &res.state_));
    }
}

NvJpeg2kEncoderPlugin::EncodeState::~EncodeState()
{
    for (auto& res : per_thread_) {
        try {
            if (res.state_) {
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeStateDestroy(res.state_));
            }
            if (res.event_) {
                XM_CHECK_CUDA(cudaEventDestroy(res.event_));
            }
            if (res.stream_) {
                XM_CHECK_CUDA(cudaStreamDestroy(res.stream_));
            }
        } catch (const std::runtime_error& e) {
            NVIMGCDCS_E_LOG_ERROR("Could not destroy nvjpeg2k decode state - " << e.what());
        }
    }
}

static void fill_encode_config(nvjpeg2kEncodeConfig_t* encode_config, const nvimgcdcsEncodeParams_t* params)
{
    nvimgcdcsJpeg2kEncodeParams_t* j2k_encode_params = static_cast<nvimgcdcsJpeg2kEncodeParams_t*>(params->next);
    while (j2k_encode_params && j2k_encode_params->type != NVIMGCDCS_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS)
        j2k_encode_params = static_cast<nvimgcdcsJpeg2kEncodeParams_t*>(j2k_encode_params->next);
    if (j2k_encode_params) {
        encode_config->stream_type = static_cast<nvjpeg2kBitstreamType>(j2k_encode_params->stream_type);
        encode_config->code_block_w = j2k_encode_params->code_block_w;
        encode_config->code_block_h = j2k_encode_params->code_block_h;
        encode_config->irreversible = j2k_encode_params->irreversible;
        encode_config->prog_order = static_cast<nvjpeg2kProgOrder>(j2k_encode_params->prog_order);
        encode_config->num_resolutions = j2k_encode_params->num_resolutions;
    }
}

static nvjpeg2kColorSpace_t nvimgcdcs_to_nvjpeg2k_color_spec(nvimgcdcsColorSpec_t nvimgcdcs_color_spec)
{
    switch (nvimgcdcs_color_spec) {
    case NVIMGCDCS_COLORSPEC_UNKNOWN:
        return NVJPEG2K_COLORSPACE_UNKNOWN;
    case NVIMGCDCS_COLORSPEC_SRGB:
        return NVJPEG2K_COLORSPACE_SRGB;
    case NVIMGCDCS_COLORSPEC_GRAY:
        return NVJPEG2K_COLORSPACE_GRAY;
    case NVIMGCDCS_COLORSPEC_SYCC:
        return NVJPEG2K_COLORSPACE_SYCC;
    case NVIMGCDCS_COLORSPEC_CMYK:
        return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
    case NVIMGCDCS_COLORSPEC_YCCK:
        return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
    case NVIMGCDCS_COLORSPEC_UNSUPPORTED:
        return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
    default:
        return NVJPEG2K_COLORSPACE_UNKNOWN;
    }
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::encode(int sample_idx)
{
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);

    executor->launch(
        executor->instance, device_id_, sample_idx, encode_state_batch_.get(), [](int tid, int sample_idx, void* task_context) -> void {
            nvtx3::scoped_range marker{"decode " + std::to_string(sample_idx)};
            auto encode_state = reinterpret_cast<NvJpeg2kEncoderPlugin::EncodeState*>(task_context);
            auto& t = encode_state->per_thread_[tid];
            auto state_handle = t.state_;
            auto handle = encode_state->handle_;
            nvimgcdcsCodeStreamDesc_t code_stream = encode_state->samples_[sample_idx].code_stream;
            nvimgcdcsImageDesc_t image = encode_state->samples_[sample_idx].image;
            const nvimgcdcsEncodeParams_t* params = encode_state->samples_[sample_idx].params;
            try {
                nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
                image->getImageInfo(image->instance, &image_info);

                unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

                nvjpeg2kEncodeParams_t encode_params_;
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&encode_params_));
                std::unique_ptr<std::remove_pointer<nvjpeg2kEncodeParams_t>::type, decltype(&nvjpeg2kEncodeParamsDestroy)> encode_params(
                    encode_params_, &nvjpeg2kEncodeParamsDestroy);

                std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info(image_info.num_planes); //assumed planar

                for (uint32_t c = 0; c < image_info.num_planes; c++) {
                    image_comp_info[c].component_width = image_info.plane_info[c].width;
                    image_comp_info[c].component_height = image_info.plane_info[c].height;
                    image_comp_info[c].precision = image_info.plane_info[c].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16;
                    image_comp_info[c].sgn = (image_info.plane_info[c].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT8) ||
                                             (image_info.plane_info[c].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_SINT16);
                }

                nvjpeg2kEncodeConfig_t encode_config;
                memset(&encode_config, 0, sizeof(encode_config));
                encode_config.color_space = nvimgcdcs_to_nvjpeg2k_color_spec(image_info.color_spec);
                encode_config.image_width = image_info.plane_info[0].width;
                encode_config.image_height = image_info.plane_info[0].height;
                encode_config.num_components = image_info.num_planes; //assumed planar
                encode_config.image_comp_info = image_comp_info.data();
                encode_config.mct_mode = params->mct_mode;

                //Defaults
                encode_config.stream_type = NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
                encode_config.code_block_w = 64;
                encode_config.code_block_h = 64;
                encode_config.irreversible = 0;
                encode_config.prog_order = NVJPEG2K_LRCP;
                encode_config.num_resolutions = 6;
                encode_config.num_layers = 1;
                encode_config.enable_tiling = 0;
                encode_config.enable_SOP_marker = 0;
                encode_config.enable_EPH_marker = 0;
                encode_config.encode_modes = 0;
                encode_config.enable_custom_precincts = 0;

                fill_encode_config(&encode_config, params);

                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetEncodeConfig(encode_params.get(), &encode_config));
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetQuality(encode_params.get(), params->target_psnr));

                std::vector<unsigned char*> encode_input;
                std::vector<size_t> pitch_in_bytes;
                nvjpeg2kImage_t input_image;
                for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                    encode_input.push_back(device_buffer + c * image_info.plane_info[c].row_stride * image_info.plane_info[c].height);
                    pitch_in_bytes.push_back(image_info.plane_info[c].row_stride);
                }

                input_image.num_components = image_info.num_planes; //assumed planar
                input_image.pixel_data = reinterpret_cast<void**>(&encode_input[0]);
                input_image.pitch_in_bytes = pitch_in_bytes.data();
                input_image.pixel_type =
                    image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? NVJPEG2K_UINT8 : NVJPEG2K_UINT16;
                NVIMGCDCS_E_LOG_DEBUG("before encode ");
                XM_CHECK_NVJPEG2K(nvjpeg2kEncode(handle, state_handle, encode_params.get(), &input_image, t.stream_));
                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                NVIMGCDCS_E_LOG_DEBUG("after encode ");

                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));
                size_t length;
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(handle, state_handle, NULL, &length, t.stream_));

                t.compressed_data_.resize(length);

                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(handle, state_handle, t.compressed_data_.data(), &length, t.stream_));

                nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
                size_t output_size;
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                io_stream->write(io_stream->instance, &output_size, static_cast<void*>(&t.compressed_data_[0]), t.compressed_data_.size());

                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
            } catch (const std::runtime_error& e) {
                NVIMGCDCS_D_LOG_ERROR("Could not encode jpeg2k code stream - " << e.what());
                image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
            }
        });

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::encodeBatch()
{
    int batch_size = encode_state_batch_->samples_.size();
    for (int i = 0; i < batch_size; i++) {
        this->encode(i);
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvJpeg2kEncoderPlugin::Encoder::static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvjpeg2k_encode_batch");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_E_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        handle->encode_state_batch_->samples_.clear();
        NVIMGCDCS_E_LOG_DEBUG("batch size - " << batch_size);
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            handle->encode_state_batch_->samples_.push_back(
                NvJpeg2kEncoderPlugin::EncodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
        return handle->encodeBatch();
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_E_LOG_ERROR("Could not encode jpeg2k batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace nvjpeg2k
