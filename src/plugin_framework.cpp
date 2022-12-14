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

#include <filesystem>
#include <iostream>
#include "codec.h"
#include "codec_registry.h"
#include "image_encoder.h"
#include "log.h"
#include "thread_safe_queue.h"
#include "image_parser_factory.h"
#include "image_encoder_factory.h"
#include "image_decoder_factory.h"

namespace fs = std::filesystem;

namespace nvimgcdcs {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
constexpr std::string_view defaultModuleDir = "/usr/lib/nvimgcodecs/plugins";
#elif defined(_WIN32) || defined(_WIN64)
constexpr std::string_view defaultModuleDir = "C:/Program Files/nvimgcodecs/plugins";
#endif

PluginFramework::PluginFramework(ICodecRegistry* codec_registry)
    : framework_desc_{NVIMGCDCS_STRUCTURE_TYPE_FRAMEWORK_DESC, nullptr, "nvImageCodecs", 0x000100,
          this, &static_register_encoder, &static_register_decoder, &static_register_parser,
          &static_log}
    , codec_registry_(codec_registry)
    , plugin_dirs_{defaultModuleDir}
{
}

PluginFramework::~PluginFramework()
{
    unloadAllExtModules();
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

nvimgcdcsStatus_t PluginFramework::static_log(void* instance,
    const nvimgcdcsDebugMessageSeverity_t message_severity,
    const nvimgcdcsDebugMessageType_t message_type,
    const nvimgcdcsDebugMessageData_t* data)
{
    PluginFramework* handle = reinterpret_cast<PluginFramework*>(instance);
    return handle->log(message_severity, message_type, data);
}

void PluginFramework::discoverAndLoadExtModules()
{
    for (const auto& dir : plugin_dirs_)
        for (const auto& entry : fs::directory_iterator(dir)) {
            const std::string modulePath(entry.path().string());
            loadExtModule(modulePath);
        }
}

void PluginFramework::loadExtModule(const std::string& modulePath)
{
    NVIMGCDCS_LOG_INFO("Loading extension module:" << modulePath);
    PluginFramework::Module module;
    module.lib_handle_ = nvimgcdcsLoadModule(modulePath);
    NVIMGCDCS_LOG_TRACE("Getting module version func");
    module.getVersion = nvimgcdcsGetFuncAddress<nvimgcdcsModuleVersion_t>(
        module.lib_handle_, "nvimgcdcsExtModuleGetVersion");
    uint32_t version = module.getVersion();
    NVIMGCDCS_LOG_INFO("Extension module:" << modulePath << " version:" << version);
    NVIMGCDCS_LOG_TRACE("Getting module load func");
    module.load = nvimgcdcsGetFuncAddress<nvimgcdcsExtModuleLoad_t>(
        module.lib_handle_, "nvimgcdcsExtModuleLoad");
    NVIMGCDCS_LOG_TRACE("Getting module unload func");
    module.unload = nvimgcdcsGetFuncAddress<nvimgcdcsExtModuleUnload_t>(
        module.lib_handle_, "nvimgcdcsExtModuleUnload");

    module.load(&framework_desc_, &module.module_handle_);
    modules_.push_back(module);
}

void PluginFramework::unloadAllExtModules()
{
    for (const auto& module : modules_) {
        module.unload(&framework_desc_, module.module_handle_);
        nvimgcdcsUnloadModule(module.lib_handle_);
    }
    modules_.clear();
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
    std::unique_ptr<IImageParserFactory> parser_factory = std::make_unique<ImageParserFactory>(desc);
    codec->registerParserFactory(std::move(parser_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::log(const nvimgcdcsDebugMessageSeverity_t message_severity,
    const nvimgcdcsDebugMessageType_t message_type,
    const nvimgcdcsDebugMessageData_t* data)
{
    Logger::get().log(message_severity, message_type, data);
    return NVIMGCDCS_STATUS_SUCCESS;
}
} // namespace nvimgcdcs