#include <cuda_runtime_api.h>
#include <nvimgcodecs.h>
#include <cstring>
#include <vector>
#include "error_handling.h"
#include "log.h"

struct nvimgcdcsDecoder
{
    std::vector<nvimgcdcsCapability_t> capabilities_ = {NVIMGCDCS_CAPABILITY_HOST_OUTPUT};
};

struct nvimgcdcsDecodeState
{};

static nvimgcdcsStatus_t nvbmp_can_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsCodeStreamDesc_t* code_streams, nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        (*code_stream)->getCodecName((*code_stream)->instance, codec_name);

        if (strcmp(codec_name, "bmp") != 0) {
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
        if (params->enable_roi) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_ROI_UNSUPPORTED;
        }
        if (params->enable_color_conversion) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_MCT_UNSUPPORTED;
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        if (image_info.color_spec != NVIMGCDCS_COLORSPEC_SRGB) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        if (image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_NONE) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
        if (image_info.num_planes != 3) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            if (image_info.plane_info[p].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
            if (image_info.plane_info[p].num_channels != 1) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_decoder_create(void* instance, nvimgcdcsDecoder_t* decoder, int device_id, const char* options)
{
    NVIMGCDCS_D_LOG_TRACE("nvbmp_decoder_create");
    *decoder = new nvimgcdcsDecoder();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_decoder_destroy(nvimgcdcsDecoder_t decoder)
{
    NVIMGCDCS_D_LOG_TRACE("nvbmp_decoder_destroy");
    delete decoder;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvbmp_get_capabilities(nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_D_LOG_TRACE("nvbmp_get_capabilities");
    if (decoder == 0)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;

    if (capabilities) {
        *capabilities = decoder->capabilities_.data();
    }

    if (size) {
        *size = decoder->capabilities_.size();
    } else {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_decoder_decode(nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t decode_state,
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsImageDesc_t image, const nvimgcdcsDecodeParams_t* params)
{
    NVIMGCDCS_D_LOG_TRACE("nvbmp_decoder_decode");
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    image->getImageInfo(image->instance, &image_info);
    size_t size = 0;
    size_t output_size = 0;
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    io_stream->size(io_stream->instance, &size);
    std::vector<unsigned char> buffer(size);
    static constexpr int kHeaderStart = 14;
    io_stream->seek(io_stream->instance, kHeaderStart, SEEK_SET);
    uint32_t header_size;
    io_stream->read(io_stream->instance, &output_size, &header_size, sizeof(header_size));
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, &buffer[0], size);
    if (output_size != size) {
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);
    for (size_t p = 0; p < image_info.num_planes; p++) {
        for (size_t y = 0; y < image_info.plane_info[p].height; y++) {
            for (size_t x = 0; x < image_info.plane_info[p].width; x++) {
                host_buffer[(image_info.num_planes - p - 1) * image_info.plane_info[p].height * image_info.plane_info[p].width +
                            (image_info.plane_info[p].height - y - 1) * image_info.plane_info[p].width + x] =
                    buffer[kHeaderStart + header_size + image_info.num_planes * (y * image_info.plane_info[p].width + x) + p];
            }
        }
    }
    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_decoder_decode_batch(nvimgcdcsDecoder_t decoder, nvimgcdcsCodeStreamDesc_t* code_streams,
    nvimgcdcsImageDesc_t* images, int batch_size, const nvimgcdcsDecodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvbmp_decoder_decode_batch");

        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        nvimgcdcsStatus_t result = NVIMGCDCS_STATUS_SUCCESS;
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            result = nvbmp_decoder_decode(decoder, nullptr, code_streams[sample_idx], images[sample_idx], params);
            if (result != NVIMGCDCS_STATUS_SUCCESS) {
                return result;
            }
        }
        return result;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not decode bmp batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

// clang-format off
nvimgcdcsDecoderDesc nvbmp_decoder = {
    NVIMGCDCS_STRUCTURE_TYPE_DECODER_DESC,
    NULL,
    NULL,               // instance
    "nvbmp_decoder",    //id
    0x00000100,         // version
    "bmp",              //  codec_type

    nvbmp_decoder_create,
    nvbmp_decoder_destroy,
    nvbmp_get_capabilities,
    nvbmp_can_decode,
    nvbmp_decoder_decode_batch
};
// clang-format on
