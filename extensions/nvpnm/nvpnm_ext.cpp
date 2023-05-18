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

extern nvimgcdcsEncoderDesc nvpnm_encoder;

namespace nvimgcdcs {

struct PnmImgCodecsExtension
{
  public:
    explicit PnmImgCodecsExtension(const nvimgcdcsFrameworkDesc_t framework)
        : framework_(framework)
    {
        framework->registerEncoder(framework->instance, &nvpnm_encoder, NVIMGCDCS_PRIORITY_VERY_LOW);
    }
    ~PnmImgCodecsExtension() { framework_->unregisterEncoder(framework_->instance, &nvpnm_encoder); }

  private:
    const nvimgcdcsFrameworkDesc_t framework_;
};

} // namespace nvimgcdcs

nvimgcdcsStatus_t nvpnm_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvpnm_extension_create");
    try {
        XM_CHECK_NULL(framework)
        XM_CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcdcsExtension_t>(new nvimgcdcs::PnmImgCodecsExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvpnm_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvpnm_extension_destroy");
    try {
        XM_CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcdcs::PnmImgCodecsExtension*>(extension);
        delete ext_handle;
        Logger::get().unregisterLogFunc();
    } catch (const std::runtime_error& e) {
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvpnm_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvpnm_extension",
    NVIMGCDCS_VER,
    NVIMGCDCS_EXT_API_VER,

    nvpnm_extension_create,
    nvpnm_extension_destroy
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

