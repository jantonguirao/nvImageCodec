
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

#include "codec_registry.h"
#include "debug_messenger.h"
#include "default_debug_messenger.h"
#include "directory_scaner.h"
#include "library_loader.h"
#include "log.h"
#include "plugin_framework.h"

namespace nvimgcdcs {

class NvImgCodecsDirector
{
  public:
    struct DefaultDebugMessengerRegistrator
    {
        DefaultDebugMessengerRegistrator(DefaultDebugMessenger* dbg_messenger)
            : debug_messenger_(dbg_messenger->getDesc())
        {
            Logger::get().registerDebugMessenger(&debug_messenger_);
        };
        ~DefaultDebugMessengerRegistrator()
        {
            Logger::get().unregisterDebugMessenger(&debug_messenger_);
        };
        DebugMessenger debug_messenger_;
    };

    explicit NvImgCodecsDirector(nvimgcdcsInstanceCreateInfo_t create_info);
    ~NvImgCodecsDirector();

    std::unique_ptr<IImageDecoder> createGenericDecoder();

    nvimgcdcsDeviceAllocator_t* device_allocator_;
    nvimgcdcsPinnedAllocator_t* pinned_allocator_;
    DefaultDebugMessenger debug_messenger_;
    DefaultDebugMessengerRegistrator registrator_;
    CodecRegistry codec_registry_;
    PluginFramework plugin_framework_;
};

} // namespace nvimgcdcs
