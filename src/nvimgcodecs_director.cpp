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
#include "builtin_modules.h"
#include "code_stream.h"
#include "default_executor.h"
#include "directory_scaner.h"
#include "environment.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "iostream_factory.h"
#include "library_loader.h"
#include "user_executor.h"

namespace nvimgcdcs {

static std::unique_ptr<IExecutor> GetExecutor(nvimgcdcsInstanceCreateInfo_t create_info, ILogger* logger)
{
    std::unique_ptr<IExecutor> exec;
    if (create_info.executor)
        exec = std::make_unique<UserExecutor>(create_info.executor);
    else
        exec = std::make_unique<DefaultExecutor>(logger, create_info.num_cpu_threads);
    return exec;
}

NvImgCodecsDirector::NvImgCodecsDirector(nvimgcdcsInstanceCreateInfo_t create_info)
    : logger_()
    , device_allocator_(create_info.device_allocator)
    , pinned_allocator_(create_info.pinned_allocator)
    , default_debug_messenger_manager_(
          &logger_, create_info.message_severity, create_info.message_category, create_info.default_debug_messenger)
    , codec_registry_(&logger_)
    , plugin_framework_(&logger_, &codec_registry_, std::move(std::make_unique<Environment>()),
          std::move(std::make_unique<DirectoryScaner>()), std::move(std::make_unique<LibraryLoader>()),
          std::move(GetExecutor(create_info, &logger_)), device_allocator_, pinned_allocator_,
          create_info.extension_modules_path ? create_info.extension_modules_path : "")
{
    if (create_info.load_builtin_modules) {
        for (auto builtin_ext : get_builtin_modules())
            plugin_framework_.registerExtension(nullptr, &builtin_ext);
    }

    if (create_info.load_extension_modules) {
        plugin_framework_.discoverAndLoadExtModules();
    }
}

NvImgCodecsDirector::~NvImgCodecsDirector()
{
}

std::unique_ptr<CodeStream> NvImgCodecsDirector::createCodeStream()
{
    return std::make_unique<CodeStream>(&codec_registry_, std::make_unique<IoStreamFactory>());
}

std::unique_ptr<ImageGenericDecoder> NvImgCodecsDirector::createGenericDecoder(
    int device_id, int num_backends, const nvimgcdcsBackend_t* backends, const char* options)
{
    return std::make_unique<ImageGenericDecoder>(&logger_, device_id, num_backends, backends, options);
}

std::unique_ptr<ImageGenericEncoder> NvImgCodecsDirector::createGenericEncoder(int device_id, int num_backends, const nvimgcdcsBackend_t* backends, const char* options)
{
    return std::make_unique<ImageGenericEncoder>(&logger_, device_id, num_backends, backends, options);
}

void NvImgCodecsDirector::registerDebugMessenger(IDebugMessenger* messenger)
{
    logger_.registerDebugMessenger(messenger);
}

void NvImgCodecsDirector::unregisterDebugMessenger(IDebugMessenger* messenger)
{
    logger_.unregisterDebugMessenger(messenger);
}

} // namespace nvimgcdcs
