#include <nvimgcodecs.h>
#include "libjpeg_turbo_decoder.h"
#include "log.h"

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

namespace libjpeg_turbo {

struct LibjpegTurboImgCodecsExtension
{
  public:
    explicit LibjpegTurboImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
        , jpeg_decoder_(framework)
    {
        framework->registerDecoder(framework->instance, jpeg_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_NORMAL);
    }
    ~LibjpegTurboImgCodecsExtension() { framework_->unregisterDecoder(framework_->instance, jpeg_decoder_.getDecoderDesc()); }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    LibjpegTurboDecoderPlugin jpeg_decoder_;
};

} // namespace libjpeg_turbo

nvimgcdcsStatus_t libjpegTurboExtensionCreate(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionCreate");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new libjpeg_turbo::LibjpegTurboImgCodecsExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t libjpegTurboExtensionDestroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionDestroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<libjpeg_turbo::LibjpegTurboImgCodecsExtension*>(extension);
        delete ext_handle;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t libjpeg_turbo_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "libjpeg_turbo_extension",
    NVIMGCDCS_VER,       
    NVIMGCDCS_EXT_API_VER,

    libjpegTurboExtensionCreate,
    libjpegTurboExtensionDestroy
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
