#include <nvimgcdcs_module.h>

NVIMGCDCS_EXTENSION_MODULE()

extern nvimgcdcsEncoderDesc example_encoder;
extern nvimgcdcsDecoderDesc example_decoder;

nvimgcdcsModuleStatus_t nvimgcdcsModuleLoad(nvimgcdcsFrameworkDesc_t* framework)
{
    framework->registerEncoder(framework->instance, &example_encoder);
    framework->registerDecoder(framework->instance, &example_decoder);

    return NVIMGCDCS_MODULE_STATUS_SUCCESS;
}
