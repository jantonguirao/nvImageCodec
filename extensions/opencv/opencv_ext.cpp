#include <nvimgcodecs.h>
#include "opencv_decoder.h"
#include "log.h"

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

namespace opencv {

struct OpenCVImgCodecsExtension
{
  public:
    explicit OpenCVImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
        , jpeg_decoder_("jpeg", framework)
        , jpeg2k_decoder_("jpeg2k", framework)
        , png_decoder_("png", framework)
        , bmp_decoder_("bmp", framework)
        , pnm_decoder_("pnm", framework)
        , tiff_decoder_("tiff", framework)
        , webp_decoder_("webp", framework)
    {
        framework->registerDecoder(framework->instance, jpeg_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
        framework->registerDecoder(framework->instance, jpeg2k_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
        framework->registerDecoder(framework->instance, png_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
        framework->registerDecoder(framework->instance, bmp_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
        framework->registerDecoder(framework->instance, pnm_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
        framework->registerDecoder(framework->instance, tiff_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
        framework->registerDecoder(framework->instance, webp_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_LOW);
    }

    ~OpenCVImgCodecsExtension()
    {
        framework_->unregisterDecoder(framework_->instance, jpeg_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, jpeg2k_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, png_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, bmp_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, pnm_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, tiff_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, webp_decoder_.getDecoderDesc());
    }
  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    OpenCVDecoderPlugin jpeg_decoder_;
    OpenCVDecoderPlugin jpeg2k_decoder_;
    OpenCVDecoderPlugin png_decoder_;
    OpenCVDecoderPlugin bmp_decoder_;
    OpenCVDecoderPlugin pnm_decoder_;
    OpenCVDecoderPlugin tiff_decoder_;
    OpenCVDecoderPlugin webp_decoder_;
};

} // namespace opencv



nvimgcdcsStatus_t opencvExtensionCreate(void *instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionCreate");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new opencv::OpenCVImgCodecsExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t opencvExtensionDestroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionDestroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<opencv::OpenCVImgCodecsExtension*>(extension);
        delete ext_handle;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t opencv_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "opencv_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    opencvExtensionCreate,
    opencvExtensionDestroy
};
// clang-format on

nvimgcdcsStatus_t get_opencv_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = opencv_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
