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
    explicit PnmImgCodecsExtension(const nvimgcdcsFrameworkDesc_t* framework)
        : framework_(framework)
        , nvpnm_encoder_(framework)
    {
        framework->registerEncoder(framework->instance, nvpnm_encoder_.getEncoderDesc(), NVIMGCDCS_PRIORITY_VERY_LOW);
    }
    ~PnmImgCodecsExtension() { framework_->unregisterEncoder(framework_->instance, nvpnm_encoder_.getEncoderDesc()); }

    static nvimgcdcsStatus_t nvpnm_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCDCS_LOG_TRACE(framework, "nvpnm_ext", "nvpnm_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcdcsExtension_t>(new PnmImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

    static nvimgcdcsStatus_t nvpnm_extension_destroy(nvimgcdcsExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<PnmImgCodecsExtension*>(extension);
            NVIMGCDCS_LOG_TRACE(ext_handle->framework_, "nvpnm_ext", "nvpnm_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCDCS_STATUS_SUCCESS;
    }

  private:
    const nvimgcdcsFrameworkDesc_t* framework_;
    NvPnmEncoderPlugin nvpnm_encoder_;
};

} // namespace nvpnm

  // clang-format off
nvimgcdcsExtensionDesc_t nvpnm_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvpnm_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    nvpnm::PnmImgCodecsExtension::nvpnm_extension_create,
    nvpnm::PnmImgCodecsExtension::nvpnm_extension_destroy
};
// clang-format on  

nvimgcdcsStatus_t get_nvpnm_extension_desc(nvimgcdcsExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->type != NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvpnm_extension;
    return NVIMGCDCS_STATUS_SUCCESS;
}

