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
#include "log.h"

extern nvimgcdcsEncoderDesc nvpnm_encoder;

nvimgcdcsStatus_t nvpnm_extension_create(void* instance, nvimgcdcsExtension_t* extension, const nvimgcdcsFrameworkDesc_t framework)
{
    Logger::get().registerLogFunc(framework->instance, framework->log);
    NVIMGCDCS_LOG_TRACE("nvpnm_extension_create");

    framework->registerEncoder(framework->instance, &nvpnm_encoder);

    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t nvpnm_extension_destroy(nvimgcdcsExtension_t extension)
{
    NVIMGCDCS_LOG_TRACE("nvpnm_extension_destroy");
    Logger::get().unregisterLogFunc();

    return NVIMGCDCS_STATUS_SUCCESS;
}

// clang-format off
nvimgcdcsExtensionDesc_t nvpnm_extension = {
    NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC,
    NULL,

    NULL,
    "nvpnm_extension",  // id
     0x00000100,        // version

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

