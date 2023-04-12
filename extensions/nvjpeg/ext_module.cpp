#include <nvimgcodecs.h>
#include "cuda_decoder.h"
#include "hw_decoder.h"
#include "cuda_encoder.h"
#include "parser.h"
#include "errors_handling.h"
#include "log.h"

extern nvimgcdcsParserDesc jpeg_parser;

namespace nvjpeg {

class NvJpegImgCodecsExtension
{
  public:
    explicit NvJpegImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
        , jpeg_parser_(framework)
        , jpeg_hw_decoder_(framework)
        , jpeg_cuda_decoder_(framework)
        , jpeg_cuda_encoder_(framework)
    {
        framework->registerParser(framework->instance, jpeg_parser_.getParserDesc());
        framework->registerEncoder(framework->instance, jpeg_cuda_encoder_.getEncoderDesc());
        if (jpeg_hw_decoder_.isPlatformSupported())
            framework->registerDecoder(framework->instance, jpeg_hw_decoder_.getDecoderDesc());
        framework->registerDecoder(framework->instance, jpeg_cuda_decoder_.getDecoderDesc());
    }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
    NvJpegParserPlugin jpeg_parser_;
    NvJpegHwDecoderPlugin jpeg_hw_decoder_;
    NvJpegCudaDecoderPlugin jpeg_cuda_decoder_;
    NvJpegCudaEncoderPlugin jpeg_cuda_encoder_;
};

} // namespace nvjpeg

struct nvimgcdcsExtension
{
    nvimgcdcsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : nvJpegExtension_(framework)
    {
    }
    nvjpeg::NvJpegImgCodecsExtension nvJpegExtension_;
};

nvimgcdcsStatus_t nvimgcdcsExtensionCreate(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t* extension)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionCreate");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = new nvimgcdcsExtension(framework);
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvimgcdcsExtensionDestroy(
    const nvimgcdcsFrameworkDesc_t framework, nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvimgcdcsExtensionDestroy");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        delete extension;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvjpeg_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    "nvjpeg_extension",  // id
     0x00000100,        // version

    nvimgcdcsExtensionCreate,
    nvimgcdcsExtensionDestroy
};
// clang-format on  

nvimgcdcsStatus_t nvimgcdcsExtensionModuleEntry(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if ( ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC){
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvjpeg_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
