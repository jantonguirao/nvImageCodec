#include <nvimgcdcs_module.h>
#include "log.h"

NVIMGCDCS_EXTENSION_MODULE()

extern nvimgcdcsParserDesc nvbmp_parser;
extern nvimgcdcsEncoderDesc nvbmp_encoder;
extern nvimgcdcsDecoderDesc nvbmp_decoder;

nvimgcdcsStatus_t nvimgcdcsExtModuleLoad(
    nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsExtModule_t* module)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);

    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtModuleLoad");
    framework->registerParser(framework->instance, &nvbmp_parser);
    framework->registerEncoder(framework->instance, &nvbmp_encoder);
    framework->registerDecoder(framework->instance, &nvbmp_decoder);

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvimgcdcsExtModuleUnload(
    nvimgcdcsFrameworkDesc_t* framework, nvimgcdcsExtModule_t module)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtModuleUnload");
    Logger::get().unregisterLogFunc();
    return NVIMGCDCS_STATUS_SUCCESS;
}