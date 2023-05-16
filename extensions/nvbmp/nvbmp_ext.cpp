#include <nvimgcodecs.h>
#include "log.h"
#include "error_handling.h"

extern nvimgcdcsEncoderDesc nvbmp_encoder;
extern nvimgcdcsDecoderDesc nvbmp_decoder;

namespace nvimgcdcs {

struct BmpImgCodecsExtension
{
  public:
    explicit BmpImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
    {
        framework->registerEncoder(framework->instance, &nvbmp_encoder, NVIMGCDCS_PRIORITY_VERY_LOW);
        framework->registerDecoder(framework->instance, &nvbmp_decoder, NVIMGCDCS_PRIORITY_VERY_LOW);
    }
    ~BmpImgCodecsExtension()
    {
        framework_->unregisterEncoder(framework_->instance, &nvbmp_encoder);
        framework_->unregisterDecoder(framework_->instance, &nvbmp_decoder);
    }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvimgcdcs

nvimgcdcsStatus_t nvbmp_extension_create(void* instance, nvimgcdcsExtension_t* extension,
    const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvbmp_extension_create");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension =
            reinterpret_cast<nvimgcdcsExtension_t>(new nvimgcdcs::BmpImgCodecsExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvbmp_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvbmp_extension_destroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::BmpImgCodecsExtension*>(extension);
        delete ext_handle;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvbmp_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvbmp_extension",  // id
     0x00000100,        // version

    nvbmp_extension_create,
    nvbmp_extension_destroy
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
