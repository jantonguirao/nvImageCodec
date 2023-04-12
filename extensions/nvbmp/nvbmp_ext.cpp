#include <nvimgcodecs.h>
#include "log.h"

extern nvimgcdcsParserDesc nvbmp_parser;
extern nvimgcdcsEncoderDesc nvbmp_encoder;
extern nvimgcdcsDecoderDesc nvbmp_decoder;

nvimgcdcsStatus_t nvimgcdcsExtensionCreate(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);

    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionCreate");
    framework->registerParser(framework->instance, &nvbmp_parser);
    framework->registerEncoder(framework->instance, &nvbmp_encoder);
    framework->registerDecoder(framework->instance, &nvbmp_decoder);

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvimgcdcsExtensionDestroy(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionDestroy");
    Logger::get().unregisterLogFunc();
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvbmp_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    "nvbmp_extension",  // id
     0x00000100,        // version

    nvimgcdcsExtensionCreate,
    nvimgcdcsExtensionDestroy
};
// clang-format on  

nvimgcdcsStatus_t get_nvbmp_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvbmp_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
