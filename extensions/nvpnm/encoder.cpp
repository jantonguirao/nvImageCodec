#include <nvimgcodecs.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "log.h"

#define XM_CHECK_CUDA(call)                                    \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << _e << "'"; \
            std::runtime_error(_error.str());                  \
        }                                                      \
    }


struct nvimgcdcsEncoder
{
    std::vector<nvimgcdcsCapability_t> capabilities_ = {NVIMGCDCS_CAPABILITY_HOST_INPUT};
};

static nvimgcdcsStatus_t pnm_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("pnm_can_encode");
    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
        char codec_name[NVIMGCDCS_MAX_CODEC_NAME_SIZE];
        (*code_stream)->getCodecName((*code_stream)->instance, codec_name);

        if (strcmp(codec_name, "pnm") != 0) {
            NVIMGCDCS_E_LOG_INFO("cannot encode because it is not pnm codec but " << codec_name);
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
        }
        if (*result != NVIMGCDCS_PROCESSING_STATUS_SUCCESS) {
            continue;
        }
        if (params->mct_mode != NVIMGCDCS_MCT_MODE_RGB) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_MCT_UNSUPPORTED;
        }

        nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        nvimgcdcsImageInfo_t out_image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &out_image_info);

        if (image_info.color_spec != NVIMGCDCS_COLORSPEC_SRGB) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        if (image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_NONE) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (out_image_info.chroma_subsampling != NVIMGCDCS_SAMPLING_NONE) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if ((image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.sample_format != NVIMGCDCS_SAMPLEFORMAT_I_RGB)) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
        if (((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.num_planes != 3)) ||
            ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) && (image_info.num_planes != 1))) {
            *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            if (image_info.plane_info[p].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 &&
                (image_info.plane_info[p].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT16)) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }

            if (((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.plane_info[p].num_channels != 1)) ||
                ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) && (image_info.plane_info[p].num_channels != 3))) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
        }
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t pnm_create(void* instance, nvimgcdcsEncoder_t* encoder, int device_id)
{
    NVIMGCDCS_E_LOG_TRACE("pnm_create_encoder");
    *encoder = new nvimgcdcsEncoder();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t pnm_destroy(nvimgcdcsEncoder_t encoder)
{
    NVIMGCDCS_E_LOG_TRACE("pnm_destroy_encoder");
    delete encoder;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t pnm_get_capabilities(nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_E_LOG_TRACE("pnm_get_capabilities");
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
int write_pnm(nvimgcdcsIoStreamDesc_t io_stream, const D* chanR, size_t pitchR, const D* chanG, size_t pitchG, const D* chanB,
    size_t pitchB, const D* chanA, size_t pitchA, int width, int height, int num_components, uint8_t precision)
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
    size_t length = header.size() + (precision / 8) * num_components * height * width;
    io_stream->reserve(io_stream->instance, length);
    io_stream->write(io_stream->instance, &written_size, static_cast<void*>(header.data()), header.size());

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
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red));
                if (num_components > 1) {
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha));
                    }
                }
            } else {
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red >> 8));
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red & 0xFF));
                if (num_components > 1) {
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green >> 8));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green & 0xFF));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue >> 8));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue & 0xFF));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha >> 8));
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha & 0xFF));
                    }
                }
            }
        }
    }
    return 0;
}

static nvimgcdcsStatus_t pnm_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t image,
    nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("pnm_encode");
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    image->getImageInfo(image->instance, &image_info);
    unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

    if (NVIMGCDCS_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
        write_pnm<unsigned char, NVIMGCDCS_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            NULL, 0, NULL, 0, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height,
            image_info.plane_info[0].num_channels, image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
        image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
    } else if (NVIMGCDCS_SAMPLEFORMAT_P_RGB == image_info.sample_format) {
        write_pnm<unsigned char>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            host_buffer + image_info.plane_info[0].row_stride * image_info.plane_info[0].height, image_info.plane_info[1].row_stride,
            host_buffer + +image_info.plane_info[0].row_stride * image_info.plane_info[0].height +
                image_info.plane_info[1].row_stride * image_info.plane_info[0].height,
            image_info.plane_info[2].row_stride, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height,
            image_info.num_planes, image_info.plane_info[0].sample_type == NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
        image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
    } else {
        image->imageReady(
            image->instance, NVIMGCDCS_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED | NVIMGCDCS_PROCESSING_STATUS_FAIL);

    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t pnm_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams,
    int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("pnm_encode_batch");

        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        nvimgcdcsStatus_t result = NVIMGCDCS_STATUS_SUCCESS;
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            result = pnm_encode(encoder, images[sample_idx], code_streams[sample_idx], params);
            if (result != NVIMGCDCS_STATUS_SUCCESS) {
                return result;
            }
        }
        return result;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not encode pnm batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

// clang-format off
struct nvimgcdcsEncoderDesc nvpnm_encoder = {
    NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC,
    NULL,
    NULL,                // instance     
    "nvpnm",             // id           
    "pnm",               // codec_type   
    pnm_create,
    pnm_destroy,
    pnm_get_capabilities,
    pnm_can_encode,
    pnm_encode_batch
};
// clang-format on
