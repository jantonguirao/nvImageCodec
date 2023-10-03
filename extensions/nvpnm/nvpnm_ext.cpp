/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "nvpnm_ext.h"
#include "error_handling.h"
#include "log.h"
#include "encoder.h"

namespace nvpnm {

struct PnmImgCodecsExtension
{
  public:
    explicit PnmImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , nvpnm_encoder_(framework)
    {
        framework->registerEncoder(framework->instance, nvpnm_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_VERY_LOW);
    }
    ~PnmImgCodecsExtension() { framework_->unregisterEncoder(framework_->instance, nvpnm_encoder_.getEncoderDesc()); }

    static nvimgcodecStatus_t nvpnm_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvpnm_ext", "nvpnm_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new PnmImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvpnm_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<PnmImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvpnm_ext", "nvpnm_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    NvPnmEncoderPlugin nvpnm_encoder_;
};

} // namespace nvpnm

  // clang-format off
nvimgcodecExtensionDesc_t nvpnm_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvpnm_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_EXT_API_VER,

    nvpnm::PnmImgCodecsExtension::nvpnm_extension_create,
    nvpnm::PnmImgCodecsExtension::nvpnm_extension_destroy
};
// clang-format on  

nvimgcodecStatus_t get_nvpnm_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvpnm_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

