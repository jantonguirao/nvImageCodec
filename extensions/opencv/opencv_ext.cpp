#include <nvimgcodecs.h>
#include "opencv_decoder.h"
#include "log.h"
#include "error_handling.h"

namespace opencv {

struct OpenCVImgCodecsExtension
{
  public:
    explicit OpenCVImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)
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

    static nvimgcdcsStatus_t opencvExtensionCreate(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "opencv_ext", "nvimgcdcsExtensionCreate");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcdcsExtension_t>(new opencv::OpenCVImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t opencvExtensionDestroy(nvimgcdcsExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<opencv::OpenCVImgCodecsExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "opencv_ext", "nvimgcdcsExtensionDestroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    OpenCVDecoderPlugin jpeg_decoder_;
    OpenCVDecoderPlugin jpeg2k_decoder_;
    OpenCVDecoderPlugin png_decoder_;
    OpenCVDecoderPlugin bmp_decoder_;
    OpenCVDecoderPlugin pnm_decoder_;
    OpenCVDecoderPlugin tiff_decoder_;
    OpenCVDecoderPlugin webp_decoder_;
};

} // namespace opencv


// clang-format off
nvimgcdcsExtensionDesc_t opencv_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "opencv_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    opencv::OpenCVImgCodecsExtension::opencvExtensionCreate,
    opencv::OpenCVImgCodecsExtension::opencvExtensionDestroy
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
