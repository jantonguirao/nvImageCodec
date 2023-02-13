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
#include "ilibrary_loader.h"

namespace nvimgcdcs {

class ICodecRegistry;
class ICodec;

class PluginFramework
{
  public:
    explicit PluginFramework(ICodecRegistry* codec_registry,
        std::unique_ptr<IDirectoryScaner> directory_scaner,
        std::unique_ptr<ILibraryLoader> library_loader);
    ~PluginFramework();
    nvimgcdcsStatus_t registerExtension(
        nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc);
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
        nvimgcdcsExtension_t* extension, const nvimgcdcsExtensionDesc_t* extension_desc,
        const Module& module);
    nvimgcdcsStatus_t unregisterExtension(std::vector<Extension>::const_iterator it);

    ICodec* ensureExistsAndRetrieveCodec(const char* codec_name);

    nvimgcdcsStatus_t registerEncoder(const struct nvimgcdcsEncoderDesc* desc);
    nvimgcdcsStatus_t registerDecoder(const struct nvimgcdcsDecoderDesc* desc);
    nvimgcdcsStatus_t registerParser(const struct nvimgcdcsParserDesc* desc);
    nvimgcdcsStatus_t log(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type,
        const nvimgcdcsDebugMessageData_t* callback_data);

    //TODO define statics with macro
    static nvimgcdcsStatus_t static_register_encoder(
        void* instance, const struct nvimgcdcsEncoderDesc* desc);
    static nvimgcdcsStatus_t static_register_decoder(
        void* instance, const struct nvimgcdcsDecoderDesc* desc);
    static nvimgcdcsStatus_t static_register_parser(
        void* instance, const struct nvimgcdcsParserDesc* desc);
    static nvimgcdcsStatus_t static_log(void* instance,
        const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type,
        const nvimgcdcsDebugMessageData_t* callback_data);

    std::unique_ptr<IDirectoryScaner> directory_scaner_;
    std::unique_ptr<ILibraryLoader> library_loader_;
    std::vector<Extension> extensions_;
    nvimgcdcsFrameworkDesc_t framework_desc_;
    ICodecRegistry* codec_registry_;
    std::vector<std::string_view> plugin_dirs_;
};
} // namespace nvimgcdcs
