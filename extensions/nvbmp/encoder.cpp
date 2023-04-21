
#include <nvimgcodecs.h>
#include <cstring>
#include <string>
#include <vector>
#include "exceptions.h"
#include "log.h"

struct nvimgcdcsEncoder
{
    std::vector<nvimgcdcsCapability_t> capabilities_ = {NVIMGCDCS_CAPABILITY_HOST_INPUT};
};

struct nvimgcdcsEncodeState
{};

template <typename D, int SAMPLE_FORMAT = NVIMGCDCS_SAMPLEFORMAT_P_RGB>
int writeBMP(nvimgcdcsIoStreamDesc_t io_stream, const D* chanR, size_t pitchR, const D* chanG, size_t pitchG, const D* chanB, size_t pitchB,
    int width, int height, uint8_t precision, bool verbose)
{

    unsigned int headers[13];
    int extrabytes;
    int paddedsize;
    int x;
    int y;
    int n;
    int red, green, blue;

    extrabytes = 4 - ((width * 3) % 4); // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extrabytes == 4)
        extrabytes = 0;

    paddedsize = ((width * 3) + extrabytes) * height;

    headers[0] = paddedsize + 54; // bfSize (whole file size)
    headers[1] = 0;               // bfReserved (both)
    headers[2] = 54;              // bfOffbits
    headers[3] = 40;              // biSize
    headers[4] = width;           // biWidth
    headers[5] = height;          // biHeight

    headers[7] = 0;          // biCompression
    headers[8] = paddedsize; // biSizeImage
    headers[9] = 0;          // biXPelsPerMeter
    headers[10] = 0;         // biYPelsPerMeter
    headers[11] = 0;         // biClrUsed
    headers[12] = 0;         // biClrImportant

    size_t written_size;
    std::string bm("BM");
    io_stream->write(io_stream->instance, &written_size, static_cast<void*>(bm.data()), 2);

    for (n = 0; n <= 5; n++) {
        io_stream->putc(io_stream->instance, &written_size, headers[n] & 0x000000FF);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x0000FF00) >> 8);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x00FF0000) >> 16);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.
    io_stream->putc(io_stream->instance, &written_size, 1);
    io_stream->putc(io_stream->instance, &written_size, 0);
    io_stream->putc(io_stream->instance, &written_size, 24);
    io_stream->putc(io_stream->instance, &written_size, 0);

    for (n = 7; n <= 12; n++) {
        io_stream->putc(io_stream->instance, &written_size, headers[n] & 0x000000FF);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x0000FF00) >> 8);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & 0x00FF0000) >> 16);
        io_stream->putc(io_stream->instance, &written_size, (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    if (verbose && precision > 8) {
        NVIMGCDCS_E_LOG_WARNING("BMP write - truncating " << (int)precision << " bit data to 8 bit");
    }

    //
    // Headers done, now write the data...
    //
    for (y = height - 1; y >= 0; y--) // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++) {

            if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_P_RGB) {
                red = chanR[y * pitchR + x];
                green = chanG[y * pitchG + x];
                blue = chanB[y * pitchB + x];
            } else if (SAMPLE_FORMAT == NVIMGCDCS_SAMPLEFORMAT_I_RGB) {
                red = chanR[(y * pitchR + 3 * x)];
                green = chanR[(y * pitchR + 3 * x) + 1];
                blue = chanR[(y * pitchR + 3 * x) + 2];
            }
            int scale = precision - 8;
            if (scale > 0) {
                red = ((red >> scale) + ((red >> (scale - 1)) % 2));
                green = ((green >> scale) + ((green >> (scale - 1)) % 2));
                blue = ((blue >> scale) + ((blue >> (scale - 1)) % 2));
            }

            if (red > 255)
                red = 255;
            if (red < 0)
                red = 0;
            if (green > 255)
                green = 255;
            if (green < 0)
                green = 0;
            if (blue > 255)
                blue = 255;
            if (blue < 0)
                blue = 0;
            // Also, it's written in (b,g,r) format...

            io_stream->putc(io_stream->instance, &written_size, blue);
            io_stream->putc(io_stream->instance, &written_size, green);
            io_stream->putc(io_stream->instance, &written_size, red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++) {
                io_stream->putc(io_stream->instance, &written_size, 0);
            }
        }
    }
    return 0;
}

static nvimgcdcsStatus_t nvbmp_encoder_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsImageDesc_t* images, nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("nvbmp_encoder_can_encode");
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
                if (params->backends[b].use_cpu) {
                    *result = NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
                }
            }
        }
    }
    //TODO check format /*&& (NVIMGCDCS_SAMPLEFORMAT_P_RGB)*/;

    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_encoder_create(void* instance, nvimgcdcsEncoder_t* encoder, int device_id)
{
    NVIMGCDCS_E_LOG_TRACE("nvbmp_encoder_create");

    *encoder = new nvimgcdcsEncoder();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_encoder_destroy(nvimgcdcsEncoder_t encoder)
{
    NVIMGCDCS_E_LOG_TRACE("nvbmp_encoder_destroy");
    delete encoder;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvbmp_get_capabilities(nvimgcdcsEncoder_t encoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
    NVIMGCDCS_E_LOG_TRACE("nvbmp_get_capabilities");
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

static nvimgcdcsStatus_t nvbmp_encoder_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsEncodeState_t encode_state, nvimgcdcsImageDesc_t image,
    nvimgcdcsCodeStreamDesc_t code_stream, const nvimgcdcsEncodeParams_t* params)
{
    NVIMGCDCS_E_LOG_TRACE("nvbmp_encoder_encode");
    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
    image->getImageInfo(image->instance, &image_info);
    unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

    if (NVIMGCDCS_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
        writeBMP<unsigned char, NVIMGCDCS_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            NULL, 0, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height, 8, true);
    } else {
        writeBMP<unsigned char>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            host_buffer + image_info.plane_info[0].row_stride * image_info.plane_info[0].height, image_info.plane_info[1].row_stride,
            host_buffer + +image_info.plane_info[0].row_stride * image_info.plane_info[0].height +
                image_info.plane_info[1].row_stride * image_info.plane_info[0].height,
            image_info.plane_info[2].row_stride, image_info.plane_info[0].width, image_info.plane_info[0].height, 8, true);
    }
    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t nvbmp_encoder_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t* images,
    nvimgcdcsCodeStreamDesc_t* code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_E_LOG_TRACE("nvbmp_encode_batch");

        if (batch_size < 1) {
            NVIMGCDCS_D_LOG_ERROR("Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        nvimgcdcsStatus_t result = NVIMGCDCS_STATUS_SUCCESS;
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            result = nvbmp_encoder_encode(encoder, nullptr, images[sample_idx], code_streams[sample_idx], params);
            if (result != NVIMGCDCS_STATUS_SUCCESS) {
                return result;
            }
        }
        return result;
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_D_LOG_ERROR("Could not encode bmp batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_ERROR);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; //TODO specific error
    }
}

// clang-format off
nvimgcdcsEncoderDesc nvbmp_encoder = {
    NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC,
    NULL,
    NULL,               // instance    
    "nvbmp_encoder",    //id
     0x00000100,        // version
    "bmp",              //  codec_type 
    nvbmp_encoder_create,
    nvbmp_encoder_destroy, 
    nvbmp_get_capabilities,
    nvbmp_encoder_can_encode,
    nvbmp_encoder_encode_batch
};
// clang-format on    