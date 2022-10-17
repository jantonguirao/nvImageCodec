#include <nvimgcdcs_module.h>

static const char* example_decoder_get_name(void* instance)
{
    return "example_encoder";
}

static nvimgcdcsDecoderStatus_t example_decoder_create(
    void* instance, nvimgcdcsData_t* params, nvimgcdcsDecoder_t* decoder)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}

static void example_decoder_destroy(nvimgcdcsDecoder_t* decoder)
{
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

static nvimgcdcsDecoderStatus_t example_decoder_decode(
    nvimgcdcsDecodeState_t decoder_state, char* input_buffer, char* output_image)
{
    return NVIMGCDCS_DECODER_STATUS_SUCCESS;
}
// clang-format off
nvimgcdcsDecoderDesc_t example_decoder = {
    NULL,              // instance    
    "example_decoder",  //id
    0x00000100,         // version
    "raw",              //  codec_type 
    example_decoder_get_name, 
    example_decoder_create,
    example_decoder_destroy, 
    example_create_decode_state, 
    example_destroy_decode_state,
    example_decoder_decode
};
// clang-format on