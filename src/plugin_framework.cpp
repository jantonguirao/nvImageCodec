/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "plugin_framework.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "codec.h"
#include "codec_registry.h"
#include "image_decoder_factory.h"
#include "image_encoder.h"
#include "image_encoder_factory.h"
#include "image_parser_factory.h"
#include "log.h"

namespace fs = std::filesystem;

namespace nvimgcdcs {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
constexpr std::string_view defaultModuleDir = "/usr/lib/nvimgcodecs/plugins";
#elif defined(_WIN32) || defined(_WIN64)
constexpr std::string_view defaultModuleDir = "C:/Program Files/nvimgcodecs/plugins";
#endif

PluginFramework::PluginFramework(ICodecRegistry* codec_registry,
    std::unique_ptr<IDirectoryScaner> directory_scaner,
    std::unique_ptr<ILibraryLoader> library_loader, std::unique_ptr<IExecutor> executor,
    nvimgcdcsDeviceAllocator_t* device_allocator, nvimgcdcsPinnedAllocator_t* pinned_allocator)
    : directory_scaner_(std::move(directory_scaner))
    , library_loader_(std::move(library_loader))
    , executor_(std::move(executor))
    , framework_desc_{NVIMGCDCS_STRUCTURE_TYPE_FRAMEWORK_DESC, nullptr, "nvImageCodecs", 0x000100,
          this, device_allocator, pinned_allocator, &static_register_encoder,
          &static_register_decoder, &static_register_parser, &static_get_executor, &static_log}
    , codec_registry_(codec_registry)
    , plugin_dirs_{defaultModuleDir}
{
}

PluginFramework::~PluginFramework()
{
    unregisterAllExtensions();
}

nvimgcdcsStatus_t PluginFramework::static_register_encoder(
    void* instance, const struct nvimgcdcsEncoderDesc* desc)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerEncoder(desc);
}

nvimgcdcsStatus_t PluginFramework::static_register_decoder(
    void* instance, const struct nvimgcdcsDecoderDesc* desc)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerDecoder(desc);
}

nvimgcdcsStatus_t PluginFramework::static_register_parser(
    void* instance, const struct nvimgcdcsParserDesc* desc)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerParser(desc);
}

nvimgcdcsStatus_t PluginFramework::static_get_executor(
    void* instance, nvimgcdcsExecutorDesc_t* result)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->getExecutor(result);
}

nvimgcdcsStatus_t PluginFramework::static_log(void* instance,
    const nvimgcdcsDebugMessageSeverity_t message_severity,
    const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* data)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->log(message_severity, message_type, data);
}

nvimgcdcsStatus_t PluginFramework::registerExtension(nvimgcdcsExtension_t* extension,
    const nvimgcdcsExtensionDesc_t* extension_desc, const Module& module)
{
    if (extension_desc == nullptr) {
        NVIMGCDCS_LOG_ERROR("Extension description cannot be null");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (extension_desc->id == nullptr) {
        NVIMGCDCS_LOG_ERROR("Extension id cannot be null");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    NVIMGCDCS_LOG_INFO(
        "Registering extension " << extension_desc->id << " version:" << extension_desc->version);

    if (extension_desc->create == nullptr) {
        NVIMGCDCS_LOG_ERROR("Could not find  'create' function in extension module");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (extension_desc->destroy == nullptr) {
        NVIMGCDCS_LOG_ERROR("Could not find  'destroy' function in extension module");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    PluginFramework::Extension internal_extension;
    internal_extension.desc_ = *extension_desc;
    internal_extension.module_ = module;
    nvimgcdcsStatus_t status =
        internal_extension.desc_.create(&framework_desc_, &internal_extension.handle_);
    if (status != NVIMGCDCS_STATUS_SUCCESS) {
        NVIMGCDCS_LOG_ERROR("Could not create extension");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    *extension = internal_extension.handle_;

    extensions_.push_back(internal_extension);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerExtension(
    nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc)
{
    Module module;
    module.lib_handle_ = nullptr;

    return registerExtension(extension, extension_desc, module);
}

nvimgcdcsStatus_t PluginFramework::unregisterExtension(std::vector<Extension>::const_iterator it)
{
    it->desc_.destroy(&framework_desc_, it->handle_);

    if (it->module_.lib_handle_ != nullptr) {
        NVIMGCDCS_LOG_INFO("Unloading extension module:" << it->module_.path_);
        library_loader_->unloadLibrary(it->module_.lib_handle_);
    }
    extensions_.erase(it);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::unregisterExtension(nvimgcdcsExtension_t extension)
{
    auto it = std::find_if(extensions_.begin(), extensions_.end(),
        [&](auto e) -> bool { return e.handle_ == extension; });
    if (it == extensions_.end()) {
        NVIMGCDCS_LOG_WARNING("Could not find extension to unregister ");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    return unregisterExtension(it);
}

void PluginFramework::unregisterAllExtensions()
{
    while (!extensions_.empty()) {
        unregisterExtension(extensions_.begin());
    }
}

void PluginFramework::discoverAndLoadExtModules()
{
    for (const auto& dir : plugin_dirs_) {
        directory_scaner_->start(dir);
        while (directory_scaner_->hasMore()) {
            fs::path dir_entry_path = directory_scaner_->next();
            auto status = directory_scaner_->symlinkStatus(dir_entry_path);
            if (fs::is_regular_file(status)) {
                //TODO check and filter out entries
                const std::string module_path(dir_entry_path.string());
                loadExtModule(module_path);
            }
        }
    }
}

void PluginFramework::loadExtModule(const std::string& modulePath)
{
    NVIMGCDCS_LOG_INFO("Loading extension module: " << modulePath);
    PluginFramework::Module module;
    module.path_ = modulePath;
    try {
        module.lib_handle_ = library_loader_->loadLibrary(modulePath);
    } catch (...) {
        NVIMGCDCS_LOG_ERROR("Could not load extension module library: " << modulePath);
        return;
    }

    NVIMGCDCS_LOG_TRACE("Getting extension module entry func");
    try {
        module.extension_entry_ = reinterpret_cast<nvimgcdcsExtensionModuleEntryFunc_t>(
            library_loader_->getFuncAddress(module.lib_handle_, "nvimgcdcsExtensionModuleEntry"));

    } catch (...) {
        NVIMGCDCS_LOG_ERROR("Could not find extension module entry function: " << modulePath);
        NVIMGCDCS_LOG_INFO("Unloading extension module:" << modulePath);
        library_loader_->unloadLibrary(module.lib_handle_);

        return;
    }
    nvimgcdcsExtensionDesc_t extension_desc;
    memset(&extension_desc, 0, sizeof(extension_desc));
    extension_desc.type = NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC;
    nvimgcdcsStatus_t status = module.extension_entry_(&extension_desc);
    if (status != NVIMGCDCS_STATUS_SUCCESS) {
        NVIMGCDCS_LOG_ERROR("Could not get extension module description");
        NVIMGCDCS_LOG_INFO("Unloading extension module:" << modulePath);
        library_loader_->unloadLibrary(module.lib_handle_);

        return;
    }
    nvimgcdcsExtension_t extension;
    status = registerExtension(&extension, &extension_desc, module);
    if (status != NVIMGCDCS_STATUS_SUCCESS) {
        NVIMGCDCS_LOG_INFO("Unloading extension module:" << modulePath);
        library_loader_->unloadLibrary(module.lib_handle_);
    }
}

ICodec* PluginFramework::ensureExistsAndRetrieveCodec(const char* codec_name)
{
    ICodec* codec = codec_registry_->getCodecByName(codec_name);
    if (codec == nullptr) {
        NVIMGCDCS_LOG_INFO(
            "Codec " << codec_name << " not yet registered, registering for first time");
        std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(codec_name);
        codec_registry_->registerCodec(std::move(new_codec));
        codec = codec_registry_->getCodecByName(codec_name);
    }
    return codec;
}

nvimgcdcsStatus_t PluginFramework::registerEncoder(const struct nvimgcdcsEncoderDesc* desc)
{
    NVIMGCDCS_LOG_INFO("Framework is registering encoder");
    NVIMGCDCS_LOG_INFO(" - id:" << desc->id);
    NVIMGCDCS_LOG_INFO(" - codec:" << desc->codec);
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    NVIMGCDCS_LOG_INFO("Registering " << desc->id);
    std::unique_ptr<IImageEncoderFactory> encoder_factory =
        std::make_unique<ImageEncoderFactory>(desc);
    codec->registerEncoderFactory(std::move(encoder_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerDecoder(const struct nvimgcdcsDecoderDesc* desc)
{
    NVIMGCDCS_LOG_INFO("Framework is regisering decoder");
    NVIMGCDCS_LOG_INFO(" - id:" << desc->id);
    NVIMGCDCS_LOG_INFO(" - codec:" << desc->codec);
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    NVIMGCDCS_LOG_INFO("Registering " << desc->id);
    std::unique_ptr<IImageDecoderFactory> decoder_factory =
        std::make_unique<ImageDecoderFactory>(desc);
    codec->registerDecoderFactory(std::move(decoder_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerParser(const struct nvimgcdcsParserDesc* desc)
{
    NVIMGCDCS_LOG_INFO("Framework is regisering parser");
    NVIMGCDCS_LOG_INFO(" - id:" << desc->id);
    NVIMGCDCS_LOG_INFO(" - codec:" << desc->codec);
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    NVIMGCDCS_LOG_INFO("Registering " << desc->id);
    std::unique_ptr<IImageParserFactory> parser_factory =
        std::make_unique<ImageParserFactory>(desc);
    codec->registerParserFactory(std::move(parser_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::getExecutor(nvimgcdcsExecutorDesc_t* result)
{
    *result = executor_->getExecutorDesc();
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::log(const nvimgcdcsDebugMessageSeverity_t message_severity,
    const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* data)
{
    Logger::get().log(message_severity, message_type, data);
    return NVIMGCDCS_STATUS_SUCCESS;
}
} // namespace nvimgcdcs