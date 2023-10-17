
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

#include <nvimgcodec.h>

#include "code_stream.h"
#include "codec_registry.h"
#include "debug_messenger.h"
#include "default_debug_messenger.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "log.h"
#include "logger.h"
#include "plugin_framework.h"

namespace nvimgcodec {

class IDebugMessenger;

class NvImgCodecDirector
{
  public:
    struct DefaultDebugMessengerManager
    {
        DefaultDebugMessengerManager(ILogger* logger, const nvimgcodecInstanceCreateInfo_t* create_info)
            : logger_(logger)
        {
            if (create_info->create_debug_messenger) {
                if (create_info->debug_messenger_desc) {
                    dbg_messenger_ = std::make_unique<DebugMessenger>(create_info->debug_messenger_desc);
                } else {
                    dbg_messenger_ = std::make_unique<DefaultDebugMessenger>(create_info->message_severity, create_info->message_category);
                }

                logger_->registerDebugMessenger(dbg_messenger_.get());
            }
        };
        ~DefaultDebugMessengerManager()
        {
            if (dbg_messenger_) {
                logger_->unregisterDebugMessenger(dbg_messenger_.get());
            }
        };
        ILogger* logger_;
        std::unique_ptr<IDebugMessenger> dbg_messenger_;
    };

    explicit NvImgCodecDirector(const nvimgcodecInstanceCreateInfo_t* create_info);
    ~NvImgCodecDirector();

    std::unique_ptr<CodeStream> createCodeStream();
    std::unique_ptr<ImageGenericDecoder> createGenericDecoder(const nvimgcodecExecutionParams_t* exec_params, const char* options);
    std::unique_ptr<ImageGenericEncoder> createGenericEncoder(const nvimgcodecExecutionParams_t* exec_params, const char* options);
    void registerDebugMessenger(IDebugMessenger* messenger);
    void unregisterDebugMessenger(IDebugMessenger* messenger);


    Logger logger_;
    DefaultDebugMessengerManager default_debug_messenger_manager_;
    CodecRegistry codec_registry_;
    PluginFramework plugin_framework_;
};

} // namespace nvimgcodec
