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
#include "directory_scaner.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "library_loader.h"
#include "default_executor.h"
#include "user_executor.h"
#include "default_image_processor.h"

namespace nvimgcdcs {

static std::unique_ptr<IExecutor> GetExecutor(nvimgcdcsInstanceCreateInfo_t create_info) {
    std::unique_ptr<IExecutor> exec;
    if (create_info.executor)
        exec = std::make_unique<UserExecutor>(create_info.executor);
    else
        exec = std::make_unique<DefaultExecutor>(create_info.num_cpu_threads);
    return exec;
}

NvImgCodecsDirector::NvImgCodecsDirector(nvimgcdcsInstanceCreateInfo_t create_info)
    : device_allocator_(create_info.device_allocator)
    , pinned_allocator_(create_info.pinned_allocator)
    , debug_messenger_(create_info.message_severity, create_info.message_type)
    , registrator_(&debug_messenger_)
    , codec_registry_()
    , plugin_framework_(&codec_registry_, std::move(std::make_unique<DirectoryScaner>()),
          std::move(std::make_unique<LibraryLoader>()),
          std::move(GetExecutor(create_info)),
          std::move(std::make_unique<DefaultImageProcessor>()),
          device_allocator_, pinned_allocator_)
{
}

NvImgCodecsDirector::~NvImgCodecsDirector()
{
}

std::unique_ptr<IImageDecoder> NvImgCodecsDirector::createGenericDecoder()
{
    return std::make_unique<ImageGenericDecoder>(&codec_registry_);
}

std::unique_ptr<IImageEncoder> NvImgCodecsDirector::createGenericEncoder()
{
    return std::make_unique<ImageGenericEncoder>(&codec_registry_);
}

} // namespace nvimgcdcs
