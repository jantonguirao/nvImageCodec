#include <nvimgcodecs.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "log.h"

struct nvimgcdcsEncoder
{
    std::vector<nvimgcdcsCapability_t> capabilities_ = {NVIMGCDCS_CAPABILITY_HOST_INPUT};
};

struct nvimgcdcsEncodeState
{};

namespace nvpxm {

#define XM_CHECK_CUDA(call)                                    \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << _e << "'"; \
            std::runtime_error(_error.str());                  \
        }                                                      \
    }

static nvimgcdcsStatus_t pxm_can_encode(void* instance, bool* result, nvimgcdcsImageDesc_t image,
    nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_can_encode");
    *result = true;
    char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
    code_stream->getCodecName(code_stream->instance, codec_name);

    if (strcmp(codec_name, "pxm") != 0) {
        NVIMGCDCS_E_LOG_INFO("cannot encode because it is not pxm codec but " << codec_name);
        *result = false;
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    if (image != nullptr) {
        nvimgcdcsImageInfo_t image_info;
        image->getImageInfo(image->instance, &image_info);

        if ((image_info.num_components > 4 || image_info.num_components == 2)) {
            NVIMGCDCS_E_LOG_INFO("cannot encode because not suppoted number components");
            *result = false;
            return NVIMGCDCS_STATUS_SUCCESS;
        }

        if ((image_info.sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) &&
            image_info.sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16) {
            NVIMGCDCS_E_LOG_INFO(
                "cannot encode because not suppoted data type #" << image_info.sample_type);
            *result = false;
            return NVIMGCDCS_STATUS_SUCCESS;
        }
    }

    if (params->backends != nullptr) {
        *result = false;
        for (int b = 0; b < params->num_backends; ++b) {
            if (params->backends[b].use_cpu) {
                *result = true;
            }
        }
        if (!*result) {
            NVIMGCDCS_E_LOG_INFO("cannot encode because not suppoted backend");
        }
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t pxm_create(
    void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_create_encoder");
    *encoder = new nvimgcdcsEncoder();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t pxm_destroy(nvimgcdcsEncoder_t encoder)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_destroy_encoder");
    delete encoder;
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t pxm_create_encode_state(
    nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t* encode_state, cudaStream_t cuda_stream)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_create_encode_state");
    *encode_state = new nvimgcdcsEncodeState();
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t pxm_destroy_encode_state(nvimgcdcsEncodeState_t encode_state)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_destroy_encode_state");
    delete encode_state;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t pxm_get_capabilities(
    nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_get_capabilities");
    if (encoder == 0)
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;

    if (capabilities) {
        *capabilities = encoder->capabilities_.data();
    }

    if (size) {
        *size = encoder->capabilities_.size();
    } else {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

template <typename D, int SAMPLE_FORMAT = NVIMGCDCS_SAMPLEFORMAT_P_RGB>
int write_pxm(nvimgcdcsIoStreamDesc_t io_stream, const D* chanR, size_t pitchR, const D* chanG,
    size_t pitchG, const D* chanB, size_t pitchB, const D* chanA, size_t pitchA, int width,
    int height, int num_components, uint8_t precision)
{
    size_t written_size;
    int red, green, blue, alpha;
    std::stringstream ss{};
    if (num_components == 4) {
        ss << "P7\n";
        ss << "#nvImageCodecs\n";
        ss << "WIDTH " << width << "\n";
        ss << "HEIGHT " << height << "\n";
        ss << "DEPTH " << num_components << "\n";
        ss << "MAXVAL " << (1 << precision) - 1 << "\n";
        ss << "TUPLTYPE RGB_ALPHA\n";
        ss << "ENDHDR\n";
    } else if (num_components == 1) {
        ss << "P5\n";
        ss << "#nvImageCodecs\n";
        ss << width << " " << height << "\n";
        ss << (1 << precision) - 1 << "\n";
    } else {
        ss << "P6\n";
        ss << "#nvImageCodecs\n";
        ss << width << " " << height << "\n";
        ss << (1 << precision) - 1 << "\n";
    }
    std::string header = ss.str();
    io_stream->write(
        io_stream->instance, &written_size, static_cast<void*>(header.data()), header.size());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
                red = chanR[y * pitchR + x];
                if (num_components > 1) {
                    green = chanG[y * pitchG + x];
                    blue = chanB[y * pitchB + x];
                    if (num_components == 4) {
                        alpha = chanA[y * pitchA + x];
                    }
                }
            } else if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
                red = chanR[(y * pitchR + 3 * x)];
                if (num_components > 1) {
                    green = chanR[(y * pitchR + 3 * x) + 1];
                    blue = chanR[(y * pitchR + 3 * x) + 2];
                    if (num_components == 4) {
                        alpha = chanR[y * pitchR + x];
                    }
                }
            }
            if (precision == 8) {
                io_stream->putc(
                    io_stream->instance, &written_size, static_cast<unsigned char>(red));
                if (num_components > 1) {
                    io_stream->putc(
                        io_stream->instance, &written_size, static_cast<unsigned char>(green));
                    io_stream->putc(
                        io_stream->instance, &written_size, static_cast<unsigned char>(blue));
                    if (num_components == 4) {
                        io_stream->putc(
                            io_stream->instance, &written_size, static_cast<unsigned char>(alpha));
                    }
                }
            } else {
                io_stream->putc(
                    io_stream->instance, &written_size, static_cast<unsigned char>(red >> 8));
                io_stream->putc(
                    io_stream->instance, &written_size, static_cast<unsigned char>(red & 0xFF));
                if (num_components > 1) {
                    io_stream->putc(
                        io_stream->instance, &written_size, static_cast<unsigned char>(green >> 8));
                    io_stream->putc(io_stream->instance, &written_size,
                        static_cast<unsigned char>(green & 0xFF));
                    io_stream->putc(
                        io_stream->instance, &written_size, static_cast<unsigned char>(blue >> 8));
                    io_stream->putc(io_stream->instance, &written_size,
                        static_cast<unsigned char>(blue & 0xFF));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size,
                            static_cast<unsigned char>(alpha >> 8));
                        io_stream->putc(io_stream->instance, &written_size,
                            static_cast<unsigned char>(alpha & 0xFF));
                    }
                }
            }
        }
    }
    return 0;
}

static nvimgcdcsStatus_t pxm_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t encode_state,
    nvimgcdcsImageDesc_t image, nvimgcdcsCodeStreamDesc_t code_stream,
    const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("pxm_encode");
    nvimgcdcsImageInfo_t image_info;
    image->getImageInfo(image->instance, &image_info);
    unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.host_buffer);

    if (NVIMGCDCS_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
        write_pxm<unsigned char, NVIMGCDCS_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer,
            image_info.plane_info[0].host_pitch_in_bytes, NULL, 0, NULL, 0, NULL, 0,
            image_info.width, image_info.height, image_info.num_components,
            image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
    } else {
        write_pxm<unsigned char>(code_stream->io_stream, host_buffer,
            image_info.plane_info[0].host_pitch_in_bytes,
            host_buffer +
                image_info.plane_info[0].host_pitch_in_bytes * image_info.height,
            image_info.plane_info[1].host_pitch_in_bytes,
            host_buffer +
                +image_info.plane_info[0].host_pitch_in_bytes * image_info.height +
                image_info.plane_info[1].host_pitch_in_bytes * image_info.height,
            image_info.plane_info[2].host_pitch_in_bytes, NULL, 0, image_info.width,
            image_info.height, image_info.num_components,
            image_info.sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
    }
    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
struct nvimgcdcsEncoderDesc ppm_encoder = {
    NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC,
    NULL,
    NULL,                // instance     
    "nvpxm",             // id           
    0x00000100,          // version     
    "pxm",               // codec_type   
    pxm_can_encode,
    pxm_create,
    pxm_destroy,
    pxm_create_encode_state,
    NULL,
    pxm_destroy_encode_state,
    pxm_get_capabilities,
    pxm_encode,
    NULL
};
// clang-format on

nvimgcdcsStatus_t extension_create(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("extension_create");

    framework->registerEncoder(framework->instance, &ppm_encoder);

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t extension_destroy(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("extension_destroy");
    Logger::get().unregisterLogFunc();

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvpxm_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    "nvpxm_extension",  // id
     0x00000100,        // version

    extension_create,
    extension_destroy
};
// clang-format on  

nvimgcdcsStatus_t get_nvpxm_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvpxm_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
} //namespace
