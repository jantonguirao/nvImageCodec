
#include <nvimgcodecs.h>
#include <cstring>
#include <string>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include "error_handling.h"
#include "log.h"
#include "encoder.h"

namespace nvbmp {

template <typename D, int SAMPLE_FORMAT = NVIMGCDCS_SAMPLEFORMAT_P_RGB>
int writeBMP(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsIoStreamDesc_t* io_stream, const D* chanR,
    size_t pitchR, const D* chanG, size_t pitchG, const D* chanB, size_t pitchB, int width, int height, uint8_t precision, bool verbose)
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

    headers[7] = 0;               // biCompression
    headers[8] = paddedsize;      // biSizeImage
    headers[9] = 0;               // biXPelsPerMeter
    headers[10] = 0;              // biYPelsPerMeter
    headers[11] = 0;              // biClrUsed
    headers[12] = 0;              // biClrImportant

    size_t written_size;
    std::string bm("BM");
    size_t length = 2 /*BM*/ + sizeof(headers) + paddedsize;
    io_stream->reserve(io_stream->instance, length);

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
        NVIMGCDCS_LOG_WARNING(framework, plugin_id, "BMP write - truncating " << (int)precision << " bit data to 8 bit");
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
    io_stream->flush(io_stream->instance);
    return 0;
}

struct EncodeState
{
    EncodeState(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework)
        : plugin_id_(plugin_id)
        , framework_(framework)
    {
    }
    ~EncodeState() = default;

    struct Sample
    {
        nvimgcdcsCodeStreamDesc_t* code_stream;
        nvimgcdcsImageDesc_t* image;
        const nvimgcdcsEncodeParams_t* params;
    };
    const char* plugin_id_;
    const nvimgcdcsFrameworkDesc_t* framework_;
    std::vector<Sample> samples_;
};

struct EncoderImpl
{
    EncoderImpl(
        const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params);
    ~EncoderImpl();

    nvimgcdcsStatus_t canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t** images,
        nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
    static nvimgcdcsProcessingStatus_t encode(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
      nvimgcdcsImageDesc_t* image,  nvimgcdcsCodeStreamDesc_t* code_stream,  const nvimgcdcsEncodeParams_t* params);
    nvimgcdcsStatus_t encodeBatch(
        nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);

    static nvimgcdcsStatus_t static_destroy(nvimgcdcsEncoder_t encoder);
    static nvimgcdcsStatus_t static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
      nvimgcdcsImageDesc_t** images,   nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params);
    static nvimgcdcsStatus_t static_encode_batch(nvimgcdcsEncoder_t encoder,  nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams,
        int batch_size, const nvimgcdcsEncodeParams_t* params);

    const char* plugin_id_;
    const nvimgcdcsFrameworkDesc_t* framework_;
    const nvimgcdcsExecutionParams_t* exec_params_;
    std::unique_ptr<EncodeState> encode_state_batch_;
};

NvBmpEncoderPlugin::NvBmpEncoderPlugin(const nvimgcdcsFrameworkDesc_t* framework)
    : encoder_desc_{NVIMGCDCS_STRUCTURE_TYPE_ENCODER_DESC, NULL, this, plugin_id_, "bmp", NVIMGCDCS_BACKEND_KIND_CPU_ONLY, static_create,
          EncoderImpl::static_destroy, EncoderImpl::static_can_encode, EncoderImpl::static_encode_batch}
    , framework_(framework)
{
}

nvimgcdcsEncoderDesc_t* NvBmpEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcdcsStatus_t EncoderImpl::canEncode(nvimgcdcsProcessingStatus_t* status, nvimgcdcsImageDesc_t** images,
    nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_can_encode");
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
            if (image_info.plane_info[p].sample_type != NVIMGCDCS_SAMPLE_DATA_TYPE_UINT8) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }

            if (((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_P_RGB) && (image_info.plane_info[p].num_channels != 1)) ||
                ((image_info.sample_format == NVIMGCDCS_SAMPLEFORMAT_I_RGB) && (image_info.plane_info[p].num_channels != 3))) {
                *result |= NVIMGCDCS_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
        }
    }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not check if nvbmp can encode - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t EncoderImpl::static_can_encode(nvimgcdcsEncoder_t encoder, nvimgcdcsProcessingStatus_t* status,
    nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<EncoderImpl*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

EncoderImpl::EncoderImpl(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework, const nvimgcdcsExecutionParams_t* exec_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
    encode_state_batch_ = std::make_unique<EncodeState>(plugin_id_, framework_);
}

nvimgcdcsStatus_t NvBmpEncoderPlugin::create(
    nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_create");
        XM_CHECK_NULL(encoder);
        *encoder = reinterpret_cast<nvimgcdcsEncoder_t>(new EncoderImpl(plugin_id_, framework_, exec_params));
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not create nvbmp encoder - " << e.what());
        return NVIMGCDCS_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t NvBmpEncoderPlugin::static_create(
    void* instance, nvimgcdcsEncoder_t* encoder, const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(exec_params);
        auto handle = reinterpret_cast<NvBmpEncoderPlugin*>(instance);
        handle->create(encoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

EncoderImpl::~EncoderImpl()
{
    NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_destroy");
}

nvimgcdcsStatus_t EncoderImpl::static_destroy(nvimgcdcsEncoder_t encoder)
{
    try {
        auto handle = reinterpret_cast<EncoderImpl*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsProcessingStatus_t EncoderImpl::encode(const char* plugin_id, const nvimgcdcsFrameworkDesc_t* framework,
    nvimgcdcsImageDesc_t* image, nvimgcdcsCodeStreamDesc_t* code_stream, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework, plugin_id, "nvbmp_encoder_encode");

    nvimgcdcsImageInfo_t image_info{NVIMGCDCS_STRUCTURE_TYPE_IMAGE_INFO, 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCDCS_STATUS_SUCCESS)
            return NVIMGCDCS_PROCESSING_STATUS_FAIL;

    unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

    if (NVIMGCDCS_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
            writeBMP<unsigned char, NVIMGCDCS_SAMPLEFORMAT_I_RGB>(plugin_id, framework, code_stream->io_stream, host_buffer,
                image_info.plane_info[0].row_stride, NULL, 0, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height, 8,
                true);
    } else {
            writeBMP<unsigned char>(plugin_id, framework, code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            host_buffer + image_info.plane_info[0].row_stride * image_info.plane_info[0].height, image_info.plane_info[1].row_stride,
            host_buffer + +image_info.plane_info[0].row_stride * image_info.plane_info[0].height +
                image_info.plane_info[1].row_stride * image_info.plane_info[0].height,
            image_info.plane_info[2].row_stride, image_info.plane_info[0].width, image_info.plane_info[0].height, 8, true);
    }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework, plugin_id, "Could not encode bmp code stream - " << e.what());
        return NVIMGCDCS_PROCESSING_STATUS_FAIL;
    }
    return NVIMGCDCS_PROCESSING_STATUS_SUCCESS;
}

nvimgcdcsStatus_t EncoderImpl::encodeBatch(
    nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams, int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        NVIMGCDCS_LOG_TRACE(framework_, plugin_id_, "nvbmp_encode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }

        encode_state_batch_->samples_.resize(batch_size);
        for (int i = 0; i < batch_size; i++) {
            encode_state_batch_->samples_[i].code_stream = code_streams[i];
            encode_state_batch_->samples_[i].image = images[i];
            encode_state_batch_->samples_[i].params = params;
        }

        auto executor = exec_params_->executor;
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            executor->launch(executor->instance, NVIMGCDCS_DEVICE_CPU_ONLY, sample_idx, encode_state_batch_.get(),
                [](int tid, int sample_idx, void* context) -> void {
                    nvtx3::scoped_range marker{"nvbmp encode " + std::to_string(sample_idx)};
                    auto* encode_state = reinterpret_cast<EncodeState*>(context);
                    auto& sample = encode_state->samples_[sample_idx];
                    auto& plugin_id = encode_state->plugin_id_;
                    auto& framework = encode_state->framework_;
                    auto result = encode(plugin_id, framework, sample.image, sample.code_stream, sample.params);
                    sample.image->imageReady(sample.image->instance, result);
                });
            }
    } catch (const std::runtime_error& e) {
        NVIMGCDCS_LOG_ERROR(framework_, plugin_id_, "Could not encode bmp batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCDCS_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCDCS_STATUS_INTERNAL_ERROR; 
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t EncoderImpl::static_encode_batch(nvimgcdcsEncoder_t encoder, nvimgcdcsImageDesc_t** images, nvimgcdcsCodeStreamDesc_t** code_streams,
    int batch_size, const nvimgcdcsEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<EncoderImpl*>(encoder);
        return handle->encodeBatch(images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

} // namespace nvbmp
