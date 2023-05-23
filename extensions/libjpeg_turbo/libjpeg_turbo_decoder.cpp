#include "nvimgcodecs.h"
#include "libjpeg_turbo_decoder.h"
#include "jpeg_mem.h"
#include "log.h"
#include <cstring>
#include <nvtx3/nvtx3.hpp>
#include <iostream>

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

namespace libjpeg_turbo {

struct DecodeState
{
    DecodeState(int num_threads)
        : per_thread_(num_threads) {}
    ~DecodeState() = default;

    struct PerThreadResources
    {
        std::vector<uint8_t> buffer;
    };

    struct Sample
    {
        nvimgcdcsCodeStreamDesc_t code_stream;
        nvimgcdcsImageDesc_t image;
        const nvimgcdcsDecodeParams_t* params;
    };
    std::vector<PerThreadResources> per_thread_;
    std::vector<Sample> samples_;

    // Options
    bool fancy_upsampling_;
    bool fast_idct_;
};

struct DecoderImpl
{
    DecoderImpl(const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id, std::string options);
    ~DecoderImpl();

    nvimgcdcsStatus_t getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size);
    nvimgcdcsStatus_t canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images,
        int batch_size, const nvimgcdcsDecodeParams_t* params);
    nvimgcdcsStatus_t decodeBatch(nvimgcdcsCodeStreamDesc_t* code_streams,
        nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    static nvimgcdcsStatus_t static_destroy(nvimgcdcsDecoder_t decoder);
    static nvimgcdcsStatus_t static_get_capabilities(nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size);
    static nvimgcdcsStatus_t static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
        nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);
    static nvimgcdcsStatus_t static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
        nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params);

    void parseOptions(std::string options);

    const std::vector<nvimgcdcsCapability_t>& capabilities_;
    const nvimgcdcsFrameworkDesc_t framework_;
    int device_id_;
    std::unique_ptr<DecodeState> decode_state_batch_;
};

LibjpegTurboDecoderPlugin::LibjpegTurboDecoderPlugin(const nvimgcdcsFrameworkDesc_t framework)
    : decoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC, NULL,
          this,                    // instance
          "libjpeg_turbo_decoder", // id
          "jpeg",                  // codec_type
          static_create, DecoderImpl::static_destroy, DecoderImpl::static_get_capabilities, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_batch}
    , capabilities_{NVIMGCDCS_CAPABILITY_HOST_OUTPUT}
    , framework_(framework)
{
}

nvimgcdcsDecoderDesc_t LibjpegTurboDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcdcsStatus_t DecoderImpl::canDecode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsCodeStreamDesc_t* code_streams,
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
                if (params->backends[b].kind == NVIMGCDCS_BACKEND_KIND_CPU_ONLY) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
            if (*result == NVIMGCDCS_PROCESSING_STATUS_BACKEND_UNSUPPORTED)
                continue;
        }

        nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &info);

        switch(info.sample_format) {
            case NVIMGCDCS_SAMPLEFORMAT_P_YUV:
            case NVIMGCDCS_SAMPLEFORMAT_I_UNCHANGED:  // TODO(janton): support?
            case NVIMGCDCS_SAMPLEFORMAT_P_UNCHANGED:
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                break;
            case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
            case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
            case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
            case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
            case NVIMGCDCS_SAMPLEFORMAT_P_Y:
            default:
                break;  // supported
        }

        if (info.num_planes != 1 && info.num_planes != 3) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }
        if (info.plane_info[0].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }
        if (info.plane_info[0].num_channels != 3 && info.plane_info[0].num_channels != 1) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        }

        // This codec doesn't apply EXIF orientation
        if (params->enable_orientation && (info.orientation.flip_x || info.orientation.flip_y || info.orientation.rotated != 0)) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libjpeg_turbo_can_decode");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not check if libjpeg_turbo can decode - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

DecoderImpl::DecoderImpl(
    const std::vector<nvimgcdcsCapability_t>& capabilities, const nvimgcdcsFrameworkDesc_t framework, int device_id, std::string options)
    : capabilities_(capabilities)
    , framework_(framework)
    , device_id_(device_id)
{
    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    int num_threads = executor->get_num_threads(executor->instance);
    decode_state_batch_ = std::make_unique<DecodeState>(num_threads);

    parseOptions(std::move(options));
}

void DecoderImpl::parseOptions(std::string options) {
    // defaults
    decode_state_batch_->fancy_upsampling_ = true;
    decode_state_batch_->fast_idct_ = false;

    std::istringstream iss(options);
    std::string token;
    while (std::getline(iss, token, ' ')) {
        std::string::size_type colon = token.find(':');
        std::string::size_type equal = token.find('=');
        if (colon == std::string::npos || equal == std::string::npos || colon > equal)
            continue;
        std::string module = token.substr(0, colon);
        if (module != "" && module != "libjpeg_turbo_decoder")
            continue;
        std::string option = token.substr(colon + 1, equal - colon - 1);
        std::string value_str = token.substr(equal + 1);

        std::istringstream value(value_str);
        if (option == "fancy_upsampling") {
            value >> decode_state_batch_->fancy_upsampling_;
        } else if (option == "fast_idct") {
            value >> decode_state_batch_->fast_idct_;
        }
    }
}


nvimgcdcsStatus_t LibjpegTurboDecoderPlugin::create(nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    *decoder = reinterpret_cast<nvimgcdcsDecoder_t>(new DecoderImpl(capabilities_, framework_, device_id, options));
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t LibjpegTurboDecoderPlugin::static_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libjpeg_turbo_create");
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<LibjpegTurboDecoderPlugin*>(instance);
        handle->create(decoder, device_id, options);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not create libjpeg_turbo decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

DecoderImpl::~DecoderImpl()
{
    try {
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy libjpeg_turbo decoder");
    }
}

nvimgcdcsStatus_t DecoderImpl::static_destroy(nvimgcdcsDecoder_t decoder)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libjpeg_turbo_destroy");
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not properly destroy libjpeg_turbo decoder - " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::getCapabilities(const nvimgcdcsCapability_t** capabilities, size_t* size)
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

nvimgcdcsStatus_t DecoderImpl::static_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libjpeg_turbo_get_capabilities");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(capabilities);
        XM_CHECK_NULL(size);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->getCapabilities(capabilities, size);
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not retrieve libjpeg_turbo decoder capabilites " << e.what());
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

nvimgcdcsStatus_t decodeImpl(
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params,
    std::vector<uint8_t>& buffer, bool fancy_upsampling = true, bool fast_idct = false)
{
    libjpeg_turbo::UncompressFlags flags;
    nvimgcdcsImageInfo_t info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    auto ret = image->getImageInfo(image->instance, &info);
    if (ret != NVIMGCDCS_STATUS_SUCCESS)
        return ret;

    flags.sample_format = info.sample_format;
    switch(flags.sample_format) {
        case NVIMGCDCS_SAMPLEFORMAT_I_RGB:
        case NVIMGCDCS_SAMPLEFORMAT_I_BGR:
        case NVIMGCDCS_SAMPLEFORMAT_P_RGB:
        case NVIMGCDCS_SAMPLEFORMAT_P_BGR:
            flags.components = 3;
            break;
        case NVIMGCDCS_SAMPLEFORMAT_P_Y:
            flags.components = 1;
            break;
        default:
            NVIMGCDCS_D_LOG_ERROR("Unsupported sample_format: " << flags.sample_format);
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    flags.dct_method = fast_idct ? JDCT_FASTEST : JDCT_DEFAULT;
    flags.fancy_upscaling = fancy_upsampling;

    if (info.region.ndim != 0 && info.region.ndim != 2) {
        NVIMGCDCS_D_LOG_ERROR("Invalid region of interest");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    if (info.region.ndim == 2) {
        flags.crop = true;
        flags.crop_y = info.region.start[0];
        flags.crop_x = info.region.start[1];
        flags.crop_height = info.region.end[0] - info.region.start[0];
        flags.crop_width = info.region.end[1] - info.region.start[1];

        if (flags.crop_x < 0 || flags.crop_y < 0 || flags.crop_height != static_cast<int>(info.plane_info[0].height) ||
            flags.crop_width != static_cast<int>(info.plane_info[0].width)) {
            NVIMGCDCS_D_LOG_ERROR("Region of interest is out of bounds");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
    }

    auto io_stream = code_stream->io_stream;
    size_t data_size;
    ret = io_stream->size(io_stream->instance, &data_size);
    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        return ret;
    }

    const void *ptr;
    ret = io_stream->raw_data(io_stream->instance, &ptr);
    if (ret != NVIMGCDCS_STATUS_SUCCESS) {
        return ret;
    }
    const uint8_t* encoded_data = static_cast<const uint8_t*>(ptr);
    if (!ptr && data_size > 0) {
        buffer.resize(data_size);
        size_t read_nbytes = 0;
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        ret = io_stream->read(io_stream->instance, &read_nbytes, buffer.data(), buffer.size());
        if (ret != NVIMGCDCS_STATUS_SUCCESS)
            return ret;
        if (read_nbytes != buffer.size()) {
            return NVIMGCDCS_STATUS_BAD_CODESTREAM;
        }
        encoded_data = buffer.data();
    }

    auto orig_sample_format = flags.sample_format;
    if (orig_sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
        flags.sample_format = NVIMGCDCS_SAMPLEFORMAT_I_RGB;
    } else if (orig_sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR) {
        flags.sample_format = NVIMGCDCS_SAMPLEFORMAT_I_BGR;
    }
    auto decoded_image = libjpeg_turbo::Uncompress(encoded_data, data_size, flags);
    if (decoded_image == nullptr) {
        return NVIMGCDCS_STATUS_INTERNAL_ERROR;
    } else if (info.buffer_kind != NVIMGCDCS_IMAGE_BUFFER_KIND_STRIDED_HOST) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    const uint8_t* src = decoded_image.get();
    uint8_t* dst = reinterpret_cast<uint8_t*>(info.buffer);
    if (orig_sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB || orig_sample_format == NVIMGCDCS_SAMPLEFORMAT_P_BGR) {
        const int num_channels = 3;
        uint32_t plane_size = info.plane_info[0].height * info.plane_info[0].width;
        for (uint32_t i = 0; i < info.plane_info[0].height * info.plane_info[0].width; i++) {
            *(dst + plane_size * 0 + i) = *(src + 0 + i * num_channels);
            *(dst + plane_size * 1 + i) = *(src + 1 + i * num_channels);
            *(dst + plane_size * 2 + i) = *(src + 2 + i * num_channels);
        }
    } else {
        uint32_t row_size_bytes = info.plane_info[0].width * flags.components * sizeof(uint8_t);
        for (uint32_t y = 0; y < info.plane_info[0].height; y++, dst += info.plane_info[0].row_stride, src += row_size_bytes) {
            std::memcpy(dst, src, row_size_bytes);
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::decodeBatch(
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    decode_state_batch_->samples_.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        decode_state_batch_->samples_[i].code_stream = code_streams[i];
        decode_state_batch_->samples_[i].image = images[i];
        decode_state_batch_->samples_[i].params = params;
    }

    nvimgcdcsExecutorDesc_t executor;
    framework_->getExecutor(framework_->instance, &executor);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        executor->launch(executor->instance, NVIMGCDCS_DEVICE_CPU_ONLY, sample_idx, decode_state_batch_.get(),
            [](int tid, int sample_idx, void* context) -> void {
                nvtx3::scoped_range marker{"libjpeg_turbo decode " + std::to_string(sample_idx)};
                auto* decode_state = reinterpret_cast<DecodeState*>(context);
                auto& sample = decode_state->samples_[sample_idx];
                auto& thread_resources = decode_state->per_thread_[tid];
                auto result = decodeImpl(sample.code_stream, sample.image, sample.params, thread_resources.buffer,
                                         decode_state->fancy_upsampling_, decode_state->fast_idct_);
                if (result == NVIMGCDCS_STATUS_SUCCESS) {
                    sample.image->imageReady(sample.image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
                } else {
                    sample.image->imageReady(sample.image->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
                }
            });
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t DecoderImpl::static_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_D_LOG_TRACE("libjpeg_turbo_decode_batch");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        nvimgcdcsStatus_t result = handle->decodeBatch(code_streams, images, batch_size, params);
        return result;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

} // namespace libjpeg_turbo
