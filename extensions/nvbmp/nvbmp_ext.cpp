#include <nvimgcodecs.h>
#include "error_handling.h"
#include "log.h"
#include "decoder.h"
#include "encoder.h"

namespace nvbmp {

struct BmpImgCodecsExtension
{
  public:
    explicit BmpImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , nvbmp_encoder_(framework)
        , nvbmp_decoder_(framework)

    {
        framework->registerEncoder(framework->instance, nvbmp_encoder_.getEncoderDesc(), NVIMGCDCS_PRIORITY_VERY_LOW);
        framework->registerDecoder(framework->instance, nvbmp_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_VERY_LOW);
    }
    ~BmpImgCodecsExtension()
    {
        framework_->unregisterEncoder(framework_->instance, nvbmp_encoder_.getEncoderDesc());
        framework_->unregisterDecoder(framework_->instance, nvbmp_decoder_.getDecoderDesc());
    }

    static nvimgcdcsStatus_t nvbmp_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "nvbmp_ext", "nvbmp_extension_create");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcdcsExtension_t>(new BmpImgCodecsExtension(framework));

        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t nvbmp_extension_destroy(nvimgcdcsExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<BmpImgCodecsExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "nvbmp_ext", "nvbmp_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    NvBmpEncoderPlugin nvbmp_encoder_;
    NvBmpDecoderPlugin nvbmp_decoder_;
};

} // namespace nvbmp


// clang-format off
nvimgcdcsExtensionDesc_t nvbmp_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvbmp_extension",  
    NVIMGCDCS_VER,       
    NVIMGCDCS_EXT_API_VER,

    nvbmp::BmpImgCodecsExtension::nvbmp_extension_create,
    nvbmp::BmpImgCodecsExtension::nvbmp_extension_destroy
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
