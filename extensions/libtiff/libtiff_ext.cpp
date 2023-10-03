#include <nvimgcodec.h>
#include "libtiff_decoder.h"
#include "log.h"
#include "error_handling.h"

namespace libtiff {

struct LibtiffImgCodecsExtension
{
  public:
    explicit LibtiffImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , tiff_decoder_(framework)
    {
        framework->registerDecoder(framework->instance, tiff_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~LibtiffImgCodecsExtension() { framework_->unregisterDecoder(framework_->instance, tiff_decoder_.getDecoderDesc()); }

    static nvimgcodecStatus_t libtiffExtensionCreate(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "libtiff_ext", "nvimgcodecExtensionCreate");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new libtiff::LibtiffImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t libtiffExtensionDestroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<libtiff::LibtiffImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "libtiff_ext", "nvimgcodecExtensionDestroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    LibtiffDecoderPlugin tiff_decoder_;
};

} // namespace libtiff

// clang-format off
nvimgcodecExtensionDesc_t libtiff_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "libtiff_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_EXT_API_VER,

    libtiff::LibtiffImgCodecsExtension::libtiffExtensionCreate,
    libtiff::LibtiffImgCodecsExtension::libtiffExtensionDestroy
};
// clang-format on

nvimgcodecStatus_t get_libtiff_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = libtiff_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
