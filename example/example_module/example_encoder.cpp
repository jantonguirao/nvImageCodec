
#include <nvimgcdcs_module.h>



static const char* example_encoder_get_name(void* instance)
{
    return "example_encoder";
}

static nvimgcdcsEncoderStatus_t example_encoder_create(
    void* instance, nvimgcdcsData_t* params, nvimgcdcsEncoder_t* encoder)
{
    return NVIMGCDCS_ENCODER_STATUS_SUCCESS;
}

static void example_encoder_destroy(void* instance, nvimgcdcsEncoder_t* encoder)
{
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

nvimgcdcsEncoderDesc example_encoder = {
    /* instance     */ NULL,
    /* id           */ "example_encoder",
    /* version       */ 0x00000100,
    /* codec_type   */ "raw", example_encoder_get_name, example_encoder_create,
    example_encoder_destroy, example_create_encode_state, example_destroy_encde_state,
    example_encoder_encode};
