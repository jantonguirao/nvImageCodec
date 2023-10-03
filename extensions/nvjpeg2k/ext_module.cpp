/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <nvimgcodec.h>
#include "nvjpeg2k_ext.h"

nvimgcodecStatus_t nvimgcodecExtensionModuleEntry(nvimgcodecExtensionDesc_t* ext_desc)
{
    return get_nvjpeg2k_extension_desc(ext_desc);
}
