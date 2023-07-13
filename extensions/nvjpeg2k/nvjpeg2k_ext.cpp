#include <nvimgcodecs.h>
#include "cuda_decoder.h"
#include "cuda_encoder.h"
#include "error_handling.h"
#include "log.h"

namespace nvjpeg2k {

struct NvJpeg2kImgCodecsExtension
{
  public:
    explicit NvJpeg2kImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg2k_decoder_(framework)
        , jpeg2k_encoder_(framework)
    {
        framework->registerEncoder(framework->instance, jpeg2k_encoder_.getEncoderDesc(), NVIMGCDCS_PRIORITY_HIGH);
        framework->registerDecoder(framework->instance, jpeg2k_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_HIGH);
    }
    ~NvJpeg2kImgCodecsExtension()
    {
        framework_->unregisterEncoder(framework_->instance, jpeg2k_encoder_.getEncoderDesc());
        framework_->unregisterDecoder(framework_->instance, jpeg2k_decoder_.getDecoderDesc());
    }

    static nvimgcdcsStatus_t nvjpeg2k_extension_create(
        void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
    {
        NVIMGCDCS_LOG_TRACE(framework, "nvjpeg2k-module", "nvjpeg2k_extension_create");
        try {
            XM_CHECK_NULL(framework)
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcdcsExtension_t>(new nvjpeg2k::NvJpeg2kImgCodecsExtension(framework));
        } catch (const NvJpeg2kException& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t nvjpeg2k_extension_destroy(nvimgcdcsExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<nvjpeg2k::NvJpeg2kImgCodecsExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "nvjpeg2k-module", "nvjpeg2k_extension_destroy");
            delete ext_handle;
        } catch (const NvJpeg2kException& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    NvJpeg2kDecoderPlugin jpeg2k_decoder_;
    NvJpeg2kEncoderPlugin jpeg2k_encoder_;
};

} // namespace nvjpeg2k

// clang-format off
nvimgcdcsExtensionDesc_t nvjpeg2k_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvjpeg2k_extension",  
    NVIMGCDCS_VER,           
    NVIMGCDCS_EXT_API_VER,
    
    nvjpeg2k::NvJpeg2kImgCodecsExtension::nvjpeg2k_extension_create,
    nvjpeg2k::NvJpeg2kImgCodecsExtension::nvjpeg2k_extension_destroy
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
