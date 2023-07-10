/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <nvimgcodecs.h>
#include "cuda_decoder.h"
#if NVJPEG_LOSSLESS_SUPPORTED
    #include "lossless_decoder.h"
#endif
#include "cuda_encoder.h"
#include "errors_handling.h"
#include "hw_decoder.h"
#include "log.h"

namespace nvjpeg {

struct NvJpegImgCodecsExtension
{
  public:  
    explicit NvJpegImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)    
        : framework_(framework)
        , jpeg_hw_decoder_(framework)
        , jpeg_cuda_decoder_(framework)        
        , jpeg_cuda_encoder_(framework)
        #if NVJPEG_LOSSLESS_SUPPORTED
        , jpeg_lossless_decoder_(framework)
        #endif
    {
        framework->registerEncoder(framework->instance, jpeg_cuda_encoder_.getEncoderDesc(), NVIMGCDCS_PRIORITY_HIGH);
        if (jpeg_hw_decoder_.isPlatformSupported())
            framework->registerDecoder(framework->instance, jpeg_hw_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_VERY_HIGH);
        framework->registerDecoder(framework->instance, jpeg_cuda_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_HIGH);
#if NVJPEG_LOSSLESS_SUPPORTED     
        framework->registerDecoder(framework->instance, jpeg_lossless_decoder_.getDecoderDesc(), NVIMGCDCS_PRIORITY_HIGH);
#endif
    }
    ~NvJpegImgCodecsExtension(){
        framework_->unregisterEncoder(framework_->instance, jpeg_cuda_encoder_.getEncoderDesc());
        if (jpeg_hw_decoder_.isPlatformSupported())
            framework_->unregisterDecoder(framework_->instance, jpeg_hw_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, jpeg_cuda_decoder_.getDecoderDesc());
#if NVJPEG_LOSSLESS_SUPPORTED      
        framework_->unregisterDecoder(framework_->instance, jpeg_lossless_decoder_.getDecoderDesc());
#endif
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    NvJpegHwDecoderPlugin jpeg_hw_decoder_;
    NvJpegCudaDecoderPlugin jpeg_cuda_decoder_;    
    NvJpegCudaEncoderPlugin jpeg_cuda_encoder_;
#if NVJPEG_LOSSLESS_SUPPORTED
    NvJpegLosslessDecoderPlugin jpeg_lossless_decoder_;    
#endif
};
} // namespace nvjpeg

nvimgcdcsStatus_t nvjpeg_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvjpeg_extension_create");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new nvjpeg::NvJpegImgCodecsExtension(framework));
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR("could not create nvjpeg extension " << e.info());
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvjpeg_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvjpeg_extension_destroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvjpeg::NvJpegImgCodecsExtension*>(extension);
        delete ext_handle;
        Logger::get().unregisterLogFunc();
    } catch (const NvJpegException& e) {
        NVIMGCDCS_LOG_ERROR("could not destroy nvimgcodecs extension " << e.info());
        return e.nvimgcdcsStatus();
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvjpeg_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvjpeg_extension",
    NVIMGCDCS_VER,    
    NVIMGCDCS_EXT_API_VER,

    nvjpeg_extension_create,
    nvjpeg_extension_destroy
};
// clang-format on  

nvimgcdcsStatus_t get_nvjpeg_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvjpeg_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}
