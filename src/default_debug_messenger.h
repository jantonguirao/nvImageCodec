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
#include "idebug_messenger.h"

namespace nvimgcdcs {

class DefaultDebugMessenger : public IDebugMessenger
{
  public:
    DefaultDebugMessenger(uint32_t message_severity = NVIMGCDCS_DEBUG_MESSAGE_SEVERITY_DEFAULT, 
        uint32_t message_category = NVIMGCDCS_DEBUG_MESSAGE_CATEGORY_ALL);

    const nvimgcdcsDebugMessengerDesc_t* getDesc() override { return &desc_; }

  private:
    int debugCallback(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageCategory_t message_category,
        const nvimgcdcsDebugMessageData_t* callback_data);

    static int static_debug_callback(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageCategory_t message_category,
        const nvimgcdcsDebugMessageData_t* callback_data,
        void* user_data
    );

    const nvimgcdcsDebugMessengerDesc_t desc_;
};

} //namespace nvimgcdcs