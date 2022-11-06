#include <nvimgcdcs_module.h>

NVIMGCDCS_EXTENSION_MODULE()

extern nvimgcdcsParserDesc example_parser;
extern nvimgcdcsEncoderDesc example_encoder;
extern nvimgcdcsDecoderDesc example_decoder;

nvimgcdcsStatus_t nvimgcdcsModuleLoad(nvimgcdcsFrameworkDesc_t* framework)
{
    framework->registerParser(framework->instance, &example_parser);
    framework->registerEncoder(framework->instance, &example_encoder);
    framework->registerDecoder(framework->instance, &example_decoder);

    return NVIMGCDCS_STATUS_SUCCESS;
}
