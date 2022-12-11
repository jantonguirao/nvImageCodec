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

namespace nvimgcdcs {

class DefaultDebugMessenger
{
  public:
    DefaultDebugMessenger(uint32_t message_severity, uint32_t message_type);

    const nvimgcdcsDebugMessengerDesc_t* getDesc() { return &desc_; }

  private:
    bool debugCallback(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type,
        const nvimgcdcsDebugMessageData_t* callback_data);

    static bool static_debug_callback(nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type,
        const nvimgcdcsDebugMessageData_t* callback_data,
        void* user_data
    );

    const nvimgcdcsDebugMessengerDesc_t desc_;
};

} //namespace nvimgcdcs