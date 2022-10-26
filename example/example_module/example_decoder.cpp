#include <nvimgcdcs_module.h>

static nvimgcdcsDecoderStatus_t example_can_decode(void* instance, bool* result,
    nvimgcdcsCodeStreamDesc_t code_stream, nvimgcdcsDecodeParams_t* params)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}

static nvimgcdcsDecoderStatus_t example_decoder_create(
    void* instance, nvimgcdcsDecoder_t* decoder, nvimgcdcsDecodeParams_t* params)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}

static nvimgcdcsDecoderStatus_t example_decoder_destroy(nvimgcdcsDecoder_t decoder)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}

static nvimgcdcsDecoderStatus_t example_create_decode_state(
    nvimgcdcsDecoder_t decoder, nvimgcdcsDecodeState_t* decode_state)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}

nvimgcdcsDecoderStatus_t example_destroy_decode_state(nvimgcdcsDecodeState_t decode_state)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}

static nvimgcdcsDecoderStatus_t example_decoder_decode(nvimgcdcsDecoder_t decoder,
    nvimgcdcsDecodeState_t decode_state, nvimgcdcsCodeStreamDesc_t code_stream,
    nvimgcdcsImageDesc_t image, nvimgcdcsDecodeParams_t* params)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
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
    example_decoder_decode
};
// clang-format on