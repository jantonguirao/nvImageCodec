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
#include <sstream>
#include <nvimgcdcs_version.h>
#include "codec.h"
#include "codec_registry.h"
#include "ienvironment.h"
#include "image_decoder_factory.h"
#include "image_encoder.h"
#include "image_encoder_factory.h"
#include "image_parser_factory.h"
#include "log.h"

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace nvimgcdcs {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)

std::string GetDefaultExtensionsPath() {
    Dl_info info;
    if (dladdr((const void*)GetDefaultExtensionsPath, &info)) {
        std::string path(info.dli_fname);
        // If this comes from a shared_object in the installation dir, 
        // we trim the nvimgcodecs dir and add "extensions" to the path
        // Examples:
        // /opt/nvidia/nvimgcodecs/lib64/libnvimgcodecs.so -> /opt/nvidia/nvimgcodecs/extensions
        // ~/.local/lib/python3.8/site-packages/nvidia/nvimgcodecs/libnvimgcodecs.so ->
        //      ~/.local/lib/python3.8/site-packages/nvidia/nvimgcodecs/extensions
        auto pos = path.find("nvimgcodecs/");
        if (pos != std::string::npos) {
           return path.substr(0, pos + strlen("nvimgcodecs/")) + "extensions";
        }
    }
    return "/opt/nvidia/nvimgcodecs/extensions";
}

char GetPathSeparator() {
    return ':';
}

#elif defined(_WIN32) || defined(_WIN64)

std::string GetDefaultExtensionsPath() {
    return "C:/Program Files/nvimgcodecs/extensions";
}

char GetPathSeparator() {
    return ';';
}

#endif


PluginFramework::PluginFramework(ILogger* logger, ICodecRegistry* codec_registry, std::unique_ptr<IEnvironment> env,
    std::unique_ptr<IDirectoryScaner> directory_scaner, std::unique_ptr<ILibraryLoader> library_loader, std::unique_ptr<IExecutor> executor,
    nvimgcdcsDeviceAllocator_t* device_allocator, nvimgcdcsPinnedAllocator_t* pinned_allocator, const std::string& extensions_path)
    : logger_(logger)
    , env_(std::move(env))
    , directory_scaner_(std::move(directory_scaner))
    , library_loader_(std::move(library_loader))
    , executor_(std::move(executor))
    , framework_desc_{NVIMGCDCS_STRUCTURE_TYPE_FRAMEWORK_DESC, nullptr, this, "nvImageCodecs", NVIMGCDCS_VER, NVIMGCDCS_EXT_API_VER,
          CUDART_VERSION, device_allocator, pinned_allocator, &static_log, &static_register_encoder, &static_unregister_encoder,
          &static_register_decoder, &static_unregister_decoder, &static_register_parser, &static_unregister_parser, &static_get_executor
         }
    , codec_registry_(codec_registry)
    , extension_paths_{}
{

    std::string effective_ext_path = extensions_path;
    if (effective_ext_path.empty()) {
        std::string env_extensions_path = env_->getVariable("NVIMGCODECS_EXTENSIONS_PATH");
        effective_ext_path = env_extensions_path.empty() ? GetDefaultExtensionsPath() : std::string(env_extensions_path);
    }
    std::stringstream ss(effective_ext_path);
    std::string current_path;
    while (getline(ss, current_path, GetPathSeparator())) {
        NVIMGCDCS_LOG_DEBUG(logger_, "Using extension path [" << current_path << "]");
        extension_paths_.push_back(current_path);
    }
}

PluginFramework::~PluginFramework()
{
    unregisterAllExtensions();
}

nvimgcdcsStatus_t PluginFramework::static_register_encoder(void* instance, const nvimgcdcsEncoderDesc_t* desc, float priority)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerEncoder(desc, priority);
}

nvimgcdcsStatus_t PluginFramework::static_register_decoder(void* instance, const nvimgcdcsDecoderDesc_t* desc, float priority)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerDecoder(desc, priority);
}

nvimgcdcsStatus_t PluginFramework::static_register_parser(void* instance, const nvimgcdcsParserDesc_t* desc, float priority)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->registerParser(desc, priority);
}

nvimgcdcsStatus_t PluginFramework::static_unregister_encoder(void* instance, const nvimgcdcsEncoderDesc_t* desc)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->unregisterEncoder(desc);
}

nvimgcdcsStatus_t PluginFramework::static_unregister_decoder(void* instance, const nvimgcdcsDecoderDesc_t* desc)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->unregisterDecoder(desc);
}

nvimgcdcsStatus_t PluginFramework::static_unregister_parser(void* instance, const nvimgcdcsParserDesc_t* desc)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->unregisterParser(desc);
}

nvimgcdcsStatus_t PluginFramework::static_get_executor(void* instance, nvimgcdcsExecutorDesc_t** result)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->getExecutor(result);
}

nvimgcdcsStatus_t PluginFramework::static_log(void* instance, const nvimgcdcsDebugMessageSeverity_t message_severity,
    const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* data)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->log(message_severity, message_type, data);
}

nvimgcdcsStatus_t PluginFramework::registerExtension(
    nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc, const Module& module)
{
    if (extension_desc == nullptr) {
        NVIMGCDCS_LOG_ERROR(logger_, "Extension description cannot be null");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (extension_desc->id == nullptr) {
        NVIMGCDCS_LOG_ERROR(logger_, "Extension id cannot be null");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (extension_desc->create == nullptr) {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not find  'create' function in extension");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (extension_desc->destroy == nullptr) {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not find  'destroy' function in extension");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }

    if (extension_desc->ext_api_version > NVIMGCDCS_EXT_API_VER) {
        NVIMGCDCS_LOG_WARNING(logger_, "Could not register extension " << extension_desc->id << " version:" << NVIMGCDCS_STREAM_VER(extension_desc->version)
                                                              << " Extension API version: " << NVIMGCDCS_STREAM_VER(extension_desc->ext_api_version)
                                                              << " newer than framework API version: " << NVIMGCDCS_EXT_API_VER);
        return NVIMGCDCS_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
    auto it = extensions_.find(extension_desc->id);
    if (it != extensions_.end()) {
        if (it->second.desc_.version == extension_desc->version) {
            NVIMGCDCS_LOG_WARNING(logger_, "Could not register extension " << extension_desc->id << " version:" << NVIMGCDCS_STREAM_VER(extension_desc->version)
                                                                  << " Extension with the same id and version already registered");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        } else if (it->second.desc_.version > extension_desc->version) {
            NVIMGCDCS_LOG_WARNING(logger_, "Could not register extension " << extension_desc->id << " version:" << NVIMGCDCS_STREAM_VER(extension_desc->version)
                                                                  << " Extension with the same id and newer version: "
                                                                  << NVIMGCDCS_STREAM_VER(it->second.desc_.version) << " already registered");
            return NVIMGCDCS_STATUS_INVALID_PARAMETER;
        } else if (it->second.desc_.version < extension_desc->version) {
            NVIMGCDCS_LOG_WARNING(logger_, "Extension with the same id:" << extension_desc->id << " and older version: " << NVIMGCDCS_STREAM_VER(it->second.desc_.version)
                                                                << " already registered and will be unregistered");
            unregisterExtension(it);
        }
    }

    NVIMGCDCS_LOG_INFO(logger_, "Registering extension " << extension_desc->id << " version:" << NVIMGCDCS_STREAM_VER(extension_desc->version));
    PluginFramework::Extension internal_extension;
    internal_extension.desc_ = *extension_desc;
    internal_extension.module_ = module;
    nvimgcdcsStatus_t status =
        internal_extension.desc_.create(internal_extension.desc_.instance, &internal_extension.handle_, &framework_desc_);
    if (status != NVIMGCDCS_STATUS_SUCCESS) {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not create extension");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    if (extension)
        *extension = internal_extension.handle_;

    extensions_.emplace(extension_desc->id, internal_extension);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerExtension(nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc)
{
    Module module;
    module.lib_handle_ = nullptr;

    return registerExtension(extension, extension_desc, module);
}

nvimgcdcsStatus_t PluginFramework::unregisterExtension(std::map<std::string, Extension>::const_iterator it)
{
    NVIMGCDCS_LOG_INFO(logger_, "Unregistering extension " << it->second.desc_.id << " version:" << NVIMGCDCS_STREAM_VER(it->second.desc_.version));
    it->second.desc_.destroy(it->second.handle_);

    if (it->second.module_.lib_handle_ != nullptr) {
        NVIMGCDCS_LOG_INFO(logger_, "Unloading extension module:" << it->second.module_.path_);
        library_loader_->unloadLibrary(it->second.module_.lib_handle_);
    }
    extensions_.erase(it);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::unregisterExtension(nvimgcdcsExtension_t extension)
{
    auto it = std::find_if(extensions_.begin(), extensions_.end(), [&](const auto& e) -> bool { return e.second.handle_ == extension; });
    if (it == extensions_.end()) {
        NVIMGCDCS_LOG_WARNING(logger_, "Could not find extension to unregister ");
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

bool is_extension_disabled(fs::path dir_entry_path)
{
    return dir_entry_path.filename().string().front() == '~';
}

void PluginFramework::discoverAndLoadExtModules()
{
    for (const auto& dir : extension_paths_) {

        if (!directory_scaner_->exists(dir)) {
            NVIMGCDCS_LOG_DEBUG(logger_, "Plugin dir does not exists [" << dir << "]");
            continue;
        }
        directory_scaner_->start(dir);
        while (directory_scaner_->hasMore()) {
            fs::path dir_entry_path = directory_scaner_->next();
            auto status = directory_scaner_->symlinkStatus(dir_entry_path);
            if (fs::is_regular_file(status)) {
                if (is_extension_disabled(dir_entry_path)) {
                    continue;
                }
                const std::string module_path(dir_entry_path.string());
                loadExtModule(module_path);
            }
        }
    }
}

void PluginFramework::loadExtModule(const std::string& modulePath)
{
    NVIMGCDCS_LOG_INFO(logger_, "Loading extension module: " << modulePath);
    PluginFramework::Module module;
    module.path_ = modulePath;
    try {
        module.lib_handle_ = library_loader_->loadLibrary(modulePath);
    } catch (const std::runtime_error& e)
    {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not load extension module library. Error: " << e.what());
        return;
    } catch (...) {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not load extension module library: " << modulePath);
        return;
    }

    NVIMGCDCS_LOG_TRACE(logger_, "Getting extension module entry func");
    try {
        module.extension_entry_ = reinterpret_cast<nvimgcdcsExtensionModuleEntryFunc_t>(
            library_loader_->getFuncAddress(module.lib_handle_, "nvimgcdcsExtensionModuleEntry"));

    } catch (...) {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not find extension module entry function: " << modulePath);
        NVIMGCDCS_LOG_INFO(logger_, "Unloading extension module:" << modulePath);
        library_loader_->unloadLibrary(module.lib_handle_);

        return;
    }
    nvimgcdcsExtensionDesc_t extension_desc{NVIMGCDCS_STRUCTURE_TYPE_EXTENSION_DESC, 0};
    nvimgcdcsStatus_t status = module.extension_entry_(&extension_desc);
    if (status != NVIMGCDCS_STATUS_SUCCESS) {
        NVIMGCDCS_LOG_ERROR(logger_, "Could not get extension module description");
        NVIMGCDCS_LOG_INFO(logger_, "Unloading extension module:" << modulePath);
        library_loader_->unloadLibrary(module.lib_handle_);

        return;
    }
    nvimgcdcsExtension_t extension;
    status = registerExtension(&extension, &extension_desc, module);
    if (status != NVIMGCDCS_STATUS_SUCCESS) {
        NVIMGCDCS_LOG_INFO(logger_, "Unloading extension module:" << modulePath);
        library_loader_->unloadLibrary(module.lib_handle_);
    }
}

ICodec* PluginFramework::ensureExistsAndRetrieveCodec(const char* codec_name)
{
    ICodec* codec = codec_registry_->getCodecByName(codec_name);
    if (codec == nullptr) {
        NVIMGCDCS_LOG_INFO(logger_, "Codec " << codec_name << " not yet registered, registering for first time");
        std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(logger_, codec_name);
        codec_registry_->registerCodec(std::move(new_codec));
        codec = codec_registry_->getCodecByName(codec_name);
    }
    return codec;
}

nvimgcdcsStatus_t PluginFramework::registerEncoder(const nvimgcdcsEncoderDesc_t* desc, float priority)
{
    NVIMGCDCS_LOG_INFO(logger_, "Framework is registering encoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    std::unique_ptr<IImageEncoderFactory> encoder_factory = std::make_unique<ImageEncoderFactory>(desc);
    codec->registerEncoderFactory(std::move(encoder_factory), priority);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::unregisterEncoder(const nvimgcdcsEncoderDesc_t* desc)
{
    NVIMGCDCS_LOG_INFO(logger_, "Framework is unregistering encoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        NVIMGCDCS_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    codec->unregisterEncoderFactory(desc->id);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerDecoder(const nvimgcdcsDecoderDesc_t* desc, float priority)
{
    NVIMGCDCS_LOG_INFO(logger_, "Framework is registering decoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    std::unique_ptr<IImageDecoderFactory> decoder_factory = std::make_unique<ImageDecoderFactory>(desc);
    codec->registerDecoderFactory(std::move(decoder_factory), priority);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::unregisterDecoder(const nvimgcdcsDecoderDesc_t* desc)
{
    NVIMGCDCS_LOG_INFO(logger_, "Framework is unregistering decoder (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        NVIMGCDCS_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    codec->unregisterDecoderFactory(desc->id);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerParser(const nvimgcdcsParserDesc_t* desc, float priority)
{
    NVIMGCDCS_LOG_INFO(logger_, "Framework is registering parser (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = ensureExistsAndRetrieveCodec(desc->codec);
    std::unique_ptr<IImageParserFactory> parser_factory = std::make_unique<ImageParserFactory>(desc);
    codec->registerParserFactory(std::move(parser_factory), priority);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::unregisterParser(const nvimgcdcsParserDesc_t* desc)
{
    NVIMGCDCS_LOG_INFO(logger_, "Framework is unregistering parser (id:" << desc->id << " codec:" << desc->codec << ")");
    ICodec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        NVIMGCDCS_LOG_WARNING(logger_, "Codec " << desc->codec << " not registered");
        return NVIMGCDCS_STATUS_INVALID_PARAMETER;
    }
    codec->unregisterParserFactory(desc->id);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::getExecutor(nvimgcdcsExecutorDesc_t** result)
{
    *result = executor_->getExecutorDesc();
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::log(const nvimgcdcsDebugMessageSeverity_t message_severity,
    const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* data)
{
    logger_->log(message_severity, message_type, data);
    return NVIMGCDCS_STATUS_SUCCESS;
}
} // namespace nvimgcdcs
