#include <nvimgcodecs.h>
#include "cuda_decoder.h"
#include "cuda_encoder.h"
#include "error_handling.h"
#include "log.h"
#include "parser.h"

namespace nvjpeg2k {

struct NvJpeg2kImgCodecsExtension
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

nvimgcdcsStatus_t nvjpeg2k_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvjpeg2k_extension_create");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new nvjpeg2k::NvJpeg2kImgCodecsExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvjpeg2k_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvjpeg2k_extension_destroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvjpeg2k::NvJpeg2kImgCodecsExtension*>(extension);
        delete ext_handle;
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

    NULL,
    "nvjpeg2k_extension",  // id
     0x00000100,        // version

    nvjpeg2k_extension_create,
    nvjpeg2k_extension_destroy
};
// clang-format on

nvimgcdcsStatus_t get_nvjpeg2k_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
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
