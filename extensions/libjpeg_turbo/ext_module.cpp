/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <nvimgcodecs.h>
#include "libjpeg_turbo_ext.h"

nvimgcdcsStatus_t nvimgcdcsExtensionModuleEntry(nvimgcdcsExtensionDesc_t* ext_desc)
{
    return get_libjpeg_turbo_extension_desc(ext_desc);
}
