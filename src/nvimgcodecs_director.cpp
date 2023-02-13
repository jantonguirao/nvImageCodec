/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "nvimgcodecs_director.h"
#include "image_generic_decoder.h"

namespace nvimgcdcs {

NvImgCodecsDirector::NvImgCodecsDirector(nvimgcdcsInstanceCreateInfo_t create_info)
    : device_allocator_(create_info.device_allocator)
    , pinned_allocator_(create_info.pinned_allocator)
    , debug_messenger_(create_info.message_severity, create_info.message_type)
    , registrator_(&debug_messenger_)
    , codec_registry_()
    , plugin_framework_(&codec_registry_, std::move(std::make_unique<DirectoryScaner>()),
          std::move(std::make_unique<LibraryLoader>()))
{
}

NvImgCodecsDirector::~NvImgCodecsDirector()
{
}

std::unique_ptr<IImageDecoder> NvImgCodecsDirector::createGenericDecoder()
{
    return std::make_unique<ImageGenericDecoder>(&codec_registry_);
}

} // namespace nvimgcdcs
