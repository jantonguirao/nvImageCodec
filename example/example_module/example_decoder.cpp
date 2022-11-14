#include <cuda_runtime_api.h>
#include <nvimgcdcs_module.h>
#include "example_parser.h"
#include "exceptions.h"

struct nvimgcdcsDecoder
{
    std::vector<nvimgcdcsCapability_t> capabilities_ = {NVIMGCDCS_CAPABILITY_HOST_OUTPUT};
};

struct nvimgcdcsDecodeState
{};

static nvimgcdcsStatus_t example_can_decode(void* instance, bool* result,
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params)
{
    *result = true;
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t example_decoder_create(
    void* instance, nvimgcdcsDecoder_t* decoder, nvimgcdcsDecodeParams_t* params)
{
    *decoder = new nvimgcdcsDecoder();
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t example_decoder_destroy(nvimgcdcsDecoder_t decoder)
{
    delete decoder;
    return NVIMGCDCS_STATUS_SUCCESS;
}

static nvimgcdcsStatus_t example_create_decode_state(
    nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state)
{
    *decode_state = new nvimgcdcsDecodeState();
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t example_destroy_decode_state(nvimgcdcsDecodeState_t decode_state)
{
    delete decode_state;
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t example_get_capabilities(
    nvimgcdcsDecoder_t decoder, const nvimgcdcsCapability_t** capabilities, size_t* size)
{
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

static nvimgcdcsStatus_t example_decoder_decode(nvimgcdcsDecoder_t decoder,
    nvimgcdcsDecodeState_t decode_state, nvimgcdcsCodeStreamDesc_t code_stream,
    nvimgcdcsImageDesc_t image, nvimgcdcsDecodeParams_t* params)
{
    std::cout << "example_ decoder_decode" << std::endl;
    nvimgcdcsImageInfo_t image_info;
    image->getImageInfo(image->instance, &image_info);
    size_t size                       = 0;
    size_t output_size                = 0;
    nvimgcdcsIoStreamDesc_t io_stream = code_stream->io_stream;
    io_stream->size(io_stream->instance, &size);
    code_stream->parse_state->buffer.resize(size);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, &code_stream->parse_state->buffer[0], size);
    if (output_size != size) {
        return NVIMGCDCS_STATUS_BAD_CODESTREAM;
    }

    static constexpr int kHeaderStart = 14;
    unsigned char* host_buffer;
    size_t host_buffer_size;
    image->getHostBuffer(image->instance, (void**)&host_buffer, &host_buffer_size);
    for (size_t c = 0; c < image_info.num_components; c++) {
        for (size_t y = 0; y < image_info.image_height; y++) {
            for (size_t x = 0; x < image_info.image_width; x++) {
                host_buffer[(image_info.num_components - c - 1) * image_info.image_height *
                                image_info.image_width +
                            (image_info.image_height - y - 1) * image_info.image_width + x] =
                    code_stream->parse_state
                        ->buffer[kHeaderStart + code_stream->parse_state->header_size +
                                 image_info.num_components * (y * image_info.image_width + x) + c];
            }
        }
    }
    image->imageReady(image->instance, NVIMGCDCS_PROCESSING_STATUS_SUCCESS);
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsDecoderDesc_t example_decoder = {
    NULL,               // instance    
    "example_decoder",  //id
    0x00000100,         // version
    "bmp",              //  codec_type 
    example_can_decode,
    example_decoder_create,
    example_decoder_destroy, 
    example_create_decode_state, 
    example_destroy_decode_state,
    example_get_capabilities,
    example_decoder_decode
};
// clang-format on