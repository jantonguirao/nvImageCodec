
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

#include "codec_registry.h"
#include "debug_messenger.h"
#include "default_debug_messenger.h"
#include "default_executor.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "log.h"
#include "plugin_framework.h"

namespace nvimgcdcs {

class NvImgCodecsDirector
{
  public:
    struct DefaultDebugMessengerManager
    {
        DefaultDebugMessengerManager(uint32_t message_severity, uint32_t message_type, bool register_messenger)
        {
            if (register_messenger) {
                dbg_messenger_ = std::make_unique<DefaultDebugMessenger>(message_severity, message_type);
                Logger::get().registerDebugMessenger(dbg_messenger_.get());
            }
        };
        ~DefaultDebugMessengerManager()
        {
            if (dbg_messenger_) {
                Logger::get().unregisterDebugMessenger(dbg_messenger_.get());
            }
        };
        std::unique_ptr<DefaultDebugMessenger> dbg_messenger_;
    };

    explicit NvImgCodecsDirector(nvimgcdcsInstanceCreateInfo_t create_info);
    ~NvImgCodecsDirector();

    std::unique_ptr<ImageGenericDecoder> createGenericDecoder(int device_id, int num_backends, const nvimgcdcsBackend_t* backends, const char* options);
    std::unique_ptr<ImageGenericEncoder> createGenericEncoder(int device_id, int num_backends, const nvimgcdcsBackend_t* backends, const char* options);

    nvimgcdcsDeviceAllocator_t* device_allocator_;
    nvimgcdcsPinnedAllocator_t* pinned_allocator_;
    DefaultDebugMessengerManager default_debug_messenger_manager_;
    CodecRegistry codec_registry_;
    PluginFramework plugin_framework_;
};

} // namespace nvimgcdcs
