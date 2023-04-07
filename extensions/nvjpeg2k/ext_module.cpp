#include <nvimgcodecs.h>
#include "cuda_decoder.h"
#include "cuda_encoder.h"
#include "error_handling.h"
#include "log.h"
#include "parser.h"

namespace nvjpeg2k {

class NvJpeg2kImgCodecsExtension
{
  public:
    explicit NvJpeg2kImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
        , jpeg2k_decoder_(framework)
        , jpeg2k_encoder_(framework)
        , jpeg2k_parser_(framework)
    {
        framework->registerParser(framework->instance, jpeg2k_parser_.getParserDesc());
        framework->registerEncoder(framework->instance, jpeg2k_encoder_.getEncoderDesc());
        framework->registerDecoder(framework->instance, jpeg2k_decoder_.getDecoderDesc());
    }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    NvJpeg2kDecoderPlugin jpeg2k_decoder_;
    NvJpeg2kEncoderPlugin jpeg2k_encoder_;
    NvJpeg2kParserPlugin jpeg2k_parser_;
};

} // namespace nvjpeg2k

struct nvimgcdcsExtension
{
    explicit nvimgcdcsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : nvJpeg2kExtension_(framework)
    {
    }
    nvjpeg2k::NvJpeg2kImgCodecsExtension nvJpeg2kExtension_;
};

nvimgcdcsStatus_t nvimgcdcsNvJ2kExtensionCreate(const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionCreate");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = new nvimgcdcsExtension(framework);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvimgcdcsNvJ2kExtensionDestroy(const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionDestroy");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        delete extension;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvjpeg2k_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    "nvjpeg2k_extension",  // id
     0x00000100,        // version

    nvimgcdcsNvJ2kExtensionCreate,
    nvimgcdcsNvJ2kExtensionDestroy
};
// clang-format on

nvimgcdcsStatus_t nvimgcdcsExtensionModuleEntry(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvjpeg2k_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}