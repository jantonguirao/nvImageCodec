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
#include "directory_scaner.h"
#include "environment.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "iostream_factory.h"
#include "library_loader.h"

namespace nvimgcdcs {

NvImgCodecsDirector::NvImgCodecsDirector(const nvimgcdcsInstanceCreateInfo_t* create_info)
    : logger_()
    , default_debug_messenger_manager_(
          &logger_, create_info->message_severity, create_info->message_category, create_info->default_debug_messenger)
    , codec_registry_(&logger_)
    , plugin_framework_(&logger_, &codec_registry_, std::move(std::make_unique<Environment>()),
          std::move(std::make_unique<DirectoryScaner>()), std::move(std::make_unique<LibraryLoader>()), create_info->extension_modules_path ? create_info->extension_modules_path : "")
{
    if (create_info->load_builtin_modules) {
        for (auto builtin_ext : get_builtin_modules())
            plugin_framework_.registerExtension(nullptr, &builtin_ext);
    }

    if (create_info->load_extension_modules) {
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
    const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    return std::make_unique<ImageGenericDecoder>(&logger_, exec_params, options);
}

std::unique_ptr<ImageGenericEncoder> NvImgCodecsDirector::createGenericEncoder(
    const nvimgcdcsExecutionParams_t* exec_params, const char* options)
{
    return std::make_unique<ImageGenericEncoder>(&logger_, exec_params, options);
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
