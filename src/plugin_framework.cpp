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
#include "thread_safe_queue.h"

namespace fs = std::filesystem;

namespace nvimgcdcs {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
constexpr std::string_view defaultModuleDir = "/usr/lib/nvimgcodecs/plugins";
#elif defined(_WIN32) || defined(_WIN64)
constexpr std::string_view defaultModuleDir = "C:/Program Files/nvimgcodecs/plugins";
#endif

PluginFramework::PluginFramework(CodecRegistry* codec_registry)
    : framework_desc_{NVIMGCDCS_STRUCTURE_TYPE_FRAMEWORK_DESC, nullptr, "nvImageCodecs", 0x000100, this,
          &static_register_encoder, &static_register_decoder, &static_register_parser}
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
    std::cout << "Loading extension module:" << modulePath  << std::endl;
    PluginFramework::Module module;
    module.lib_handle_ = nvimgcdcsLoadModule(modulePath);
    std::cout << "Getting module version" << std::endl;
    module.getVersion = nvimgcdcsGetFuncAddress<nvimgcdcsModuleVersion_t>(
        module.lib_handle_, "nvimgcdcsExtModuleGetVersion");
    uint32_t version = module.getVersion();
    std::cout << "Extension module:" << modulePath << " version:" << version
              << std::endl;
    module.load = nvimgcdcsGetFuncAddress<nvimgcdcsExtModuleLoad_t>(
        module.lib_handle_, "nvimgcdcsExtModuleLoad");
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

nvimgcdcsStatus_t PluginFramework::registerEncoder(const struct nvimgcdcsEncoderDesc* desc)
{
    std::cout << "Framework side register_encoder" << std::endl;
    std::cout << " - id:" << desc->id << std::endl;
    std::cout << " - codec:" << desc->codec << std::endl;
    Codec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        std::cout << "Codec " << desc->codec << " not found, creating new one" << std::endl;
        std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(desc->codec);
        codec_registry_->registerCodec(std::move(new_codec));
        codec = codec_registry_->getCodecByName(desc->codec);
    } else {
        std::cout << "Codec " << desc->codec << " found" << std::endl;
    }
    std::cout << "Creating new encoder factory " << std::endl;
    std::unique_ptr<ImageEncoderFactory> encoder_factory =
        std::make_unique<ImageEncoderFactory>(desc);
    codec->registerEncoder(std::move(encoder_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerDecoder(const struct nvimgcdcsDecoderDesc* desc)
{
    std::cout << "Framework side register_decoder" << std::endl;
    std::cout << " - id:" << desc->id << std::endl;
    std::cout << " - codec:" << desc->codec << std::endl;
    Codec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        std::cout << "Codec " << desc->codec << " not found, creating new one" << std::endl;
        std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(desc->codec);
        codec_registry_->registerCodec(std::move(new_codec));
        codec = codec_registry_->getCodecByName(desc->codec);
    } else {
        std::cout << "Codec " << desc->codec << " found" << std::endl;
    }
    std::cout << "Creating new decoder factory " << std::endl;
    std::unique_ptr<ImageDecoderFactory> decoder_factory =
        std::make_unique<ImageDecoderFactory>(desc);
    codec->registerDecoder(std::move(decoder_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}

nvimgcdcsStatus_t PluginFramework::registerParser(const struct nvimgcdcsParserDesc* desc)
{
    std::cout << "Framework side register parser" << std::endl;
    std::cout << " - id:" << desc->id << std::endl;
    std::cout << " - codec:" << desc->codec << std::endl;
    Codec* codec = codec_registry_->getCodecByName(desc->codec);
    if (codec == nullptr) {
        std::cout << "Codec " << desc->codec << " not found, creating new one" << std::endl;
        std::unique_ptr<Codec> new_codec = std::make_unique<Codec>(desc->codec);
        codec_registry_->registerCodec(std::move(new_codec));
        codec = codec_registry_->getCodecByName(desc->codec);
    } else {
        std::cout << "Codec " << desc->codec << " found" << std::endl;
    }
    std::cout << "Creating new parser factory " << std::endl;
    std::unique_ptr<ImageParserFactory> parser_factory = std::make_unique<ImageParserFactory>(desc);
    codec->registerParser(std::move(parser_factory), 1);
    return NVIMGCDCS_STATUS_SUCCESS;
}


} // namespace nvimgcdcs