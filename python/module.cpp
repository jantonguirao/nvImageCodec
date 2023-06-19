/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "module.h"
#include "image.h"

namespace nvimgcdcs {

Module::Module()
{
    nvimgcdcsInstanceCreateInfo_t instance_create_info{NVIMGCDCS_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0};
    instance_create_info.load_builtin_modules = true;
    instance_create_info.load_extension_modules = true;
    instance_create_info.default_debug_messenger = false;
    instance_create_info.message_severity =
        NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_FATAL | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_ERROR | NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_WARNING;
    instance_create_info.message_type = NVIMGCDCS_DEBUG_MESSAGE_TYPE_ALL;
    nvimgcdcsInstanceCreate(&instance_, instance_create_info);
}

Module ::~Module()
{
    nvimgcdcsInstanceDestroy(instance_);
}

void Module::exportToPython(py::module& m, nvimgcdcsInstance_t instance)
{
    m.def("as_image", [instance](py::handle src) -> Image { return Image(instance, src.ptr()); });
}

} // namespace nvimgcdcs
