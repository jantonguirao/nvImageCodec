#include <nvimgcodecs.h>
#include "libtiff_decoder.h"
#include "log.h"
#include "error_handling.h"

namespace libtiff {

struct LibtiffImgCodecsExtension
{
  public:
    explicit LibtiffImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , tiff_decoder_(framework)
    {
        framework->registerDecoder(framework->instance, tiff_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~LibtiffImgCodecsExtension() { framework_->unregisterDecoder(framework_->instance, tiff_decoder_.getDecoderDesc()); }

    static nvimgcdcsStatus_t libtiffExtensionCreate(
        void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "libtiff_ext", "nvimgcdcsExtensionCreate");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcdcsExtension_t>(new libtiff::LibtiffImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t libtiffExtensionDestroy(nvimgcdcsExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<libtiff::LibtiffImgCodecsExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "libtiff_ext", "nvimgcdcsExtensionDestroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    LibtiffDecoderPlugin tiff_decoder_;
};

} // namespace libtiff

// clang-format off
nvimgcdcsExtensionDesc_t libtiff_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "libtiff_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    libtiff::LibtiffImgCodecsExtension::libtiffExtensionCreate,
    libtiff::LibtiffImgCodecsExtension::libtiffExtensionDestroy
};
// clang-format on

nvimgcdcsStatus_t get_libtiff_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = libtiff_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
