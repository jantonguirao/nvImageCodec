/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodecs.h>
#include <map>
#include <string>
#include <vector>
#include "idirectory_scaner.h"
#include "iexecutor.h"
#include "ilibrary_loader.h"

namespace nvimgcdcs {

class ICodecRegistry;
class ICodec;
class IEnvironment;

std::string GetDefaultExtensionsPath();
char GetPathSeparator();
class PluginFramework
{
  public:
    explicit PluginFramework(ICodecRegistry* codec_registry, std::unique_ptr<IEnvironment> env,
        std::unique_ptr<IDirectoryScaner> directory_scaner, std::unique_ptr<ILibraryLoader> library_loader,
        std::unique_ptr<IExecutor> executor, nvimgcdcsDeviceAllocator_t* device_allocator, nvimgcdcsPinnedAllocator_t* pinned_allocator,
        const std::string& extensions_path);
    ~PluginFramework();
    nvimgcdcsStatus_t registerExtension(nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc);
    nvimgcdcsStatus_t unregisterExtension(nvimgcdcsExtension_t extension);
    void unregisterAllExtensions();

    void discoverAndLoadExtModules();
    void loadExtModule(const std::string& modulePath);

  private:
    struct Module
    {
        std::string path_;
        ILibraryLoader::LibraryHandle lib_handle_;
        nvimgcdcsExtensionModuleEntryFunc_t extension_entry_;
    };

    struct Extension
    {
        nvimgcdcsExtension_t handle_;
        nvimgcdcsExtensionDesc_t desc_;
        Module module_;
    };

    nvimgcdcsStatus_t registerExtension(
        nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc, const Module& module);
    nvimgcdcsStatus_t unregisterExtension(std::map<std::string, Extension>::const_iterator it);

    ICodec* ensureExistsAndRetrieveCodec(const char* codec_name);

    nvimgcdcsStatus_t registerEncoder(const nvimgcdcsEncoderDesc_t desc, float priority);
    nvimgcdcsStatus_t unregisterEncoder(const nvimgcdcsEncoderDesc_t desc);
    nvimgcdcsStatus_t registerDecoder(const nvimgcdcsDecoderDesc_t desc, float priority);
    nvimgcdcsStatus_t unregisterDecoder(const nvimgcdcsDecoderDesc_t desc);
    nvimgcdcsStatus_t registerParser(const nvimgcdcsParserDesc_t* desc, float priority);
    nvimgcdcsStatus_t unregisterParser(const nvimgcdcsParserDesc_t* desc);

    nvimgcdcsStatus_t getExecutor(nvimgcdcsExecutorDesc_t** result);
    nvimgcdcsStatus_t log(const nvimgcdcsDebugMessageSeverity_t message_severity, const nvimgcdcsDebugMessageType_t message_type,
        const nvimgcdcsDebugMessageData_t* callback_data);

    static nvimgcdcsStatus_t static_register_encoder(void* instance, const nvimgcdcsEncoderDesc_t desc, float priority);
    static nvimgcdcsStatus_t static_unregister_encoder(void* instance, const nvimgcdcsEncoderDesc_t desc);
    static nvimgcdcsStatus_t static_register_decoder(void* instance, const nvimgcdcsDecoderDesc_t desc, float priority);
    static nvimgcdcsStatus_t static_unregister_decoder(void* instance, const nvimgcdcsDecoderDesc_t desc);
    static nvimgcdcsStatus_t static_register_parser(void* instance, const nvimgcdcsParserDesc_t* desc, float priority);
    static nvimgcdcsStatus_t static_unregister_parser(void* instance, const nvimgcdcsParserDesc_t* desc);

    static nvimgcdcsStatus_t static_get_executor(void* instance, nvimgcdcsExecutorDesc_t** result);
    static nvimgcdcsStatus_t static_log(void* instance, const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* callback_data);

    std::unique_ptr<IEnvironment> env_;
    std::unique_ptr<IDirectoryScaner> directory_scaner_;
    std::unique_ptr<ILibraryLoader> library_loader_;
    std::map<std::string, Extension> extensions_;
    std::unique_ptr<IExecutor> executor_;
    struct nvimgcdcsFrameworkDesc framework_desc_;
    ICodecRegistry* codec_registry_;
    std::vector<std::string> extension_paths_;
};
} // namespace nvimgcdcs
