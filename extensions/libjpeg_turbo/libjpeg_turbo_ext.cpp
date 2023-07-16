#include <nvimgcodecs.h>
#include "libjpeg_turbo_decoder.h"
#include "log.h"
#include "error_handling.h"

namespace libjpeg_turbo {

struct LibjpegTurboImgCodecsExtension
{
  public:
    explicit LibjpegTurboImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg_decoder_(framework)
    {
        framework->registerDecoder(framework->instance, jpeg_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~LibjpegTurboImgCodecsExtension() { framework_->unregisterDecoder(framework_->instance, jpeg_decoder_.getDecoderDesc()); }
    static nvimgcdcsStatus_t libjpegTurboExtensionCreate(
        void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "libjpeg_turbo_ext", "nvimgcdcsExtensionCreate");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcdcsExtension_t>(new libjpeg_turbo::LibjpegTurboImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t libjpegTurboExtensionDestroy(nvimgcdcsExtension_t extension)
    {

        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<libjpeg_turbo::LibjpegTurboImgCodecsExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "libjpeg_turbo_ext", "nvimgcdcsExtensionDestroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    LibjpegTurboDecoderPlugin jpeg_decoder_;
};

} // namespace libjpeg_turbo


// clang-format off
nvimgcdcsExtensionDesc_t libjpeg_turbo_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "libjpeg_turbo_extension",
    NVIMGCDCS_VER,       
    NVIMGCDCS_EXT_API_VER,

    libjpeg_turbo::LibjpegTurboImgCodecsExtension::libjpegTurboExtensionCreate,
    libjpeg_turbo::LibjpegTurboImgCodecsExtension::libjpegTurboExtensionDestroy
};
// clang-format on

nvimgcdcsStatus_t get_libjpeg_turbo_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = libjpeg_turbo_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
