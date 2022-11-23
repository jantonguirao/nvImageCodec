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
#include <nvimgcdcs_module.h>
#include <nvimgcodecs.h>
#include <string>
#include <vector>
#include <map>
#include "nvimgcdcs_module_load.h"
#include "thread_safe_queue.h"

namespace nvimgcdcs {
class CodecRegistry;
class Image;
class PluginFramework
{
  public:
    explicit PluginFramework(CodecRegistry* codec_registry);
    ~PluginFramework();
    void discoverAndLoadExtModules();
    void loadExtModule(const std::string& modulePath);
    void unloadAllExtModules();
  private:
    nvimgcdcsStatus_t registerEncoder(const struct nvimgcdcsEncoderDesc* desc);
    nvimgcdcsStatus_t registerDecoder(const struct nvimgcdcsDecoderDesc* desc);
    nvimgcdcsStatus_t registerParser(const struct nvimgcdcsParserDesc* desc);
    //TODO define statics with macro
    static nvimgcdcsStatus_t static_register_encoder(
        void* instance, const struct nvimgcdcsEncoderDesc* desc);
    static nvimgcdcsStatus_t static_register_decoder(
        void* instance, const struct nvimgcdcsDecoderDesc* desc);
    static nvimgcdcsStatus_t static_register_parser(
        void* instance, const struct nvimgcdcsParserDesc* desc);

    std::vector<nvimgcdcsModuleHandle> modules_;
    nvimgcdcsFrameworkDesc framework_desc_;
    CodecRegistry* codec_registry_;
    std::vector<std::string_view> plugin_dirs_;
};
} // namespace nvimgcdcs
