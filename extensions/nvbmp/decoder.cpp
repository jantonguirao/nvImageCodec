#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <cstring>
#include <memory>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include "decoder.h"
#include "error_handling.h"
#include "log.h"

namespace nvbmp {

struct DecodeState
{
    DecodeState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, int num_threads)
        : plugin_id_(plugin_id)
        , framework_(framework)
        , per_thread_(num_threads)
    {
    }
    ~DecodeState() = default;

    struct PerThreadResources
    {
        std::vector<uint8_t> buffer;
    };

    struct Sample
    {
        nvimgcdcsCodeStreamDesc_t* code_stream;
        nvimgcdcsImageDesc_t* image;
        const nvimgcdcsDecodeParams_t* params;
    };
    const char* plugin_id_;
    const nvimgcdcsFrameworkDesc_t* framework_;
    std::vector<PerThreadResources> per_thread_;
    std::vector<Sample> samples_;
};

struct DecoderImpl
{
    DecoderImpl(
        const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, int device_id, const nvimgcdcsBackendParams_t* backend_params);
    ~DecoderImpl();

    nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
        nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
    static nvimgcdcsProcessingStatus_t decode(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
        nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params,
        std::vector<uint8_t>& buffer);
    nvimgcdcsStatus_t decodeBatch(
        nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
    static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
        nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);
    static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
        nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    const char* plugin_id_;
    const nvimgcdcsFrameworkDesc_t* framework_;
    int device_id_;
    const nvimgcdcsBackendParams_t* backend_params_;
    std::unique_ptr<DecodeState> decode_state_batch_;
};

NvBmpDecoderPlugin::NvBmpDecoderPlugin(const nvimgcdcsFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL, this, plugin_id_, "bmp", NVIMGCDCS_BACKEND_KIND_CPU_ONLY, static_create,
          DecoderImpl::static_destroy, DecoderImpl::static_can_decode, DecoderImpl::static_decode_batch}
    , framework_(framework)
{
}

nvimgcdcsDecoderDesc_t* NvBmpDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t DecoderImpl::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t** code_streams,
    nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto result = status;
        auto code_stream = code_streams;
        auto image = images;
        for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
            *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;

            nvimgcdcsImageInfo_t cs_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
            (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

            if (strcmp(cs_image_info.codec_name, "bmp") != 0) {
                *result = NVIMGCDCS_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                continue;
            }

            if (params->enable_roi) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_ROI_UNSUPPORTED;
            }

            nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
            (*image)->getImageInfo((*image)->instance, &image_info);
            if (image_info.color_spec != NVIMGCDCS_COLORSPEC_SRGB) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
            if (image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_NONE) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if ((image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_I_RGB)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }
            if ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.num_planes != 3)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            if ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) && (image_info.num_planes != 1)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                if (image_info.plane_info[p].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                    *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }

                if ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.plane_info[p].num_channels != 1)) {
                    *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }

                if ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) && (image_info.plane_info[p].num_channels != 3)) {
                    *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if nvbmp can decode - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

DecoderImpl::DecoderImpl(
    const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, int device_id, const nvimgcdcsBackendParams_t* backend_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , device_id_(device_id)
    , backend_params_(backend_params)
{
    nvimgcdcsExecutorDesc_t* executor;
    framework_->getExecutor(framework_->instance, &executor);
    int num_threads = executor->get_num_threads(executor->instance);
    decode_state_batch_ = std::make_unique<DecodeState>(plugin_id_, framework_, num_threads);
}

nvimgcdcsStatus_t NvBmpDecoderPlugin::create(
    nvimgcdcsDecoder_t* decoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_create");
        XM_CHECK_NULL(decoder);
        *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new DecoderImpl(plugin_id_, framework_, device_id, backend_params));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not create nvbmp decoder - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvBmpDecoderPlugin::static_create(
    void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const nvimgcdcsBackendParams_t* backend_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<NvBmpDecoderPlugin*>(instance);
        handle->create(decoder, device_id, backend_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

DecoderImpl::~DecoderImpl()
{
    NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_destroy");
}

nvimgcdcsStatus_t DecoderImpl::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsProcessingStatus_t DecoderImpl::decode(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
    nvimgcdcsCodeStreamDesc_t* code_stream, nvimgcdcsImageDesc_t* image, const nvimgcdcsDecodeParams_t* params,
    std::vector<uint8_t>& buffer)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework, plugin_id, "nvbmp_decoder_decode");

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCDCS_STATUS_SUCCESS)
            return NVIMGCDCS_PROCESSING_STATUS_FAIL;

        size_t size = 0;
        size_t output_size = 0;
        nvimgcdcsIoStreamDesc_t* io_stream = code_stream->io_stream;
        io_stream->size(io_stream->instance, &size);
        buffer.resize(size);
        static constexpr int kHeaderStart = 14;
        io_stream->seek(io_stream->instance, kHeaderStart, SEEK_SET);
        uint32_t header_size;
        io_stream->read(io_stream->instance, &output_size, &header_size, sizeof(header_size));
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        io_stream->read(io_stream->instance, &output_size, &buffer[0], size);
        if (output_size != size) {
            return NVIMGCDCS_PROCESSING_STATUS_IMAGE_CORRUPTED;
        }

        unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
            for (size_t p = 0; p < image_info.num_planes; p++) {
                for (size_t y = 0; y < image_info.plane_info[p].height; y++) {
                    for (size_t x = 0; x < image_info.plane_info[p].width; x++) {
                        host_buffer[(image_info.num_planes - p - 1) * image_info.plane_info[p].height * image_info.plane_info[p].width +
                                    (image_info.plane_info[p].height - y - 1) * image_info.plane_info[p].width + x] =
                            buffer[kHeaderStart + header_size + image_info.num_planes * (y * image_info.plane_info[p].width + x) + p];
                    }
                }
            }
        } else if (image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
            for (size_t c = 0; c < image_info.plane_info[0].num_channels; c++) {
                for (size_t y = 0; y < image_info.plane_info[0].height; y++) {
                    for (size_t x = 0; x < image_info.plane_info[0].width; x++) {
                        auto src_idx = kHeaderStart + header_size +
                                       image_info.plane_info[0].num_channels * (y * image_info.plane_info[0].width + x) + c;
                        auto dst_idx = (image_info.plane_info[0].height - y - 1) * image_info.plane_info[0].width *
                                           image_info.plane_info[0].num_channels +
                                       x * image_info.plane_info[0].num_channels + (image_info.plane_info[0].num_channels - c - 1);
                        host_buffer[dst_idx] = buffer[src_idx];
                    }
                }
            }
        } else {
            return NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Could not decode bmp code stream - " << e.what());
        return NVIMGCDCS_PROCESSING_STATUS_FAIL;
    }
    return NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::decodeBatch(
    nvimgcdcsCodeStreamDesc_t** code_streams, nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }

        decode_state_batch_->samples_.resize(batch_size);
        for (int i = 0; i < batch_size; i++) {
            decode_state_batch_->samples_[i].code_stream = code_streams[i];
            decode_state_batch_->samples_[i].image = images[i];
            decode_state_batch_->samples_[i].params = params;
        }

        nvimgcdcsExecutorDesc_t* executor;
        framework_->getExecutor(framework_->instance, &executor);
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            auto task = [](int tid, int sample_idx, void* context) -> void {
                nvtx3::scoped_range marker{"nvbmp decode " + std::to_string(sample_idx)};
                auto* decode_state = reinterpret_cast<DecodeState*>(context);
                auto& sample = decode_state->samples_[sample_idx];
                auto& thread_resources = decode_state->per_thread_[tid];
                auto& plugin_id = decode_state->plugin_id_;
                auto& framework = decode_state->framework_;

                auto result = decode(plugin_id, framework, sample.code_stream, sample.image, sample.params, thread_resources.buffer);
                sample.image->imageReady(sample.image->instance, result);
            };
            if (batch_size == 1) {
                task(0, sample_idx, decode_state_batch_.get());
            } else {
                executor->launch(executor->instance, NVIMGCDCS_DEVICE_CPU_ONLY, sample_idx, decode_state_batch_.get(), std::move(task));
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not decode bmp batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t** code_streams,
    nvimgcdcsImageDesc_t** images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

} // namespace nvbmp
