
#include <nvimgcdcs_module.h>

static nvimgcdcsEncoderStatus_t example_encoder_create(
    void* instance, nvimgcdcsData_t* params, nvimgcdcsEncoder_t* encoder)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_encoder_destroy(nvimgcdcsEncoder_t encoder)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_create_encode_state(
    nvimgcdcsEncoder_t decoder, nvimgcdcsEncodeState_t* encode_state)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

nvimgcdcsEncoderStatus_t example_destroy_encde_state(nvimgcdcsEncodeState_t encode_state)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static nvimgcdcsEncoderStatus_t example_encoder_encode(
    nvimgcdcsEncodeState_t encode_state, char* input_image, char* output_buffer)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsEncoderDesc example_encoder = {
    NULL,               // instance    
    "example_encoder",  //id
     0x00000100,        // version
    "raw",              //  codec_type 
    example_encoder_create,
    example_encoder_destroy, 
    example_create_encode_state, 
    example_destroy_encde_state,
    example_encoder_encode
};
// clang-format on    
