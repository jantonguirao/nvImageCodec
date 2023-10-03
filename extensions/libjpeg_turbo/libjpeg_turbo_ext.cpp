#include <nvimgcodec.h>
#include "libjpeg_turbo_decoder.h"
#include "log.h"
#include "error_handling.h"

namespace libjpeg_turbo {

struct LibjpegTurboImgCodecsExtension
{
  public:
    explicit LibjpegTurboImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg_decoder_(framework)
    {
        framework->registerDecoder(framework->instance, jpeg_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~LibjpegTurboImgCodecsExtension() { framework_->unregisterDecoder(framework_->instance, jpeg_decoder_.getDecoderDesc()); }
    static nvimgcodecStatus_t libjpegTurboExtensionCreate(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "libjpeg_turbo_ext", "nvimgcodecExtensionCreate");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new libjpeg_turbo::LibjpegTurboImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t libjpegTurboExtensionDestroy(nvimgcodecExtension_t extension)
    {

        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<libjpeg_turbo::LibjpegTurboImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "libjpeg_turbo_ext", "nvimgcodecExtensionDestroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    LibjpegTurboDecoderPlugin jpeg_decoder_;
};

} // namespace libjpeg_turbo


// clang-format off
nvimgcodecExtensionDesc_t libjpeg_turbo_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "libjpeg_turbo_extension",
    NVIMGCODEC_VER,       
    NVIMGCODEC_EXT_API_VER,

    libjpeg_turbo::LibjpegTurboImgCodecsExtension::libjpegTurboExtensionCreate,
    libjpeg_turbo::LibjpegTurboImgCodecsExtension::libjpegTurboExtensionDestroy
};
// clang-format on

nvimgcodecStatus_t get_libjpeg_turbo_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = libjpeg_turbo_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
