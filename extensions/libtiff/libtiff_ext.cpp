#include <nvimgcodecs.h>
#include "libtiff_decoder.h"
#include "log.h"

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

namespace libtiff {

struct LibtiffImgCodecsExtension
{
  public:
    explicit LibtiffImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
        , tiff_decoder_(framework)
    {
        framework->registerDecoder(framework->instance, tiff_decoder_.getDecoderDesc());
    }
    ~LibtiffImgCodecsExtension()
    {
      framework_->unregisterDecoder(framework_->instance, tiff_decoder_.getDecoderDesc());   
    }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    LibtiffDecoderPlugin tiff_decoder_;
};

} // namespace libtiff

nvimgcdcsStatus_t libtiffExtensionCreate(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionCreate");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new libtiff::LibtiffImgCodecsExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t libtiffExtensionDestroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionDestroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<libtiff::LibtiffImgCodecsExtension*>(extension);
        delete ext_handle;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t libtiff_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "libtiff_extension",  // id
     0x00000100,        // version

    libtiffExtensionCreate,
    libtiffExtensionDestroy
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
