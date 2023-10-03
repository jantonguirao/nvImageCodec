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
#include "idebug_messenger.h"

namespace nvimgcodec {

class DefaultDebugMessenger : public IDebugMessenger
{
  public:
    DefaultDebugMessenger(uint32_t message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT, 
        uint32_t message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL);

    const nvimgcodecDebugMessengerDesc_t* getDesc() override { return &desc_; }

  private:
    int debugCallback(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category,
        const nvimgcodecDebugMessageData_t* callback_data);

    static int static_debug_callback(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category,
        const nvimgcodecDebugMessageData_t* callback_data,
        void* user_data
    );

    const nvimgcodecDebugMessengerDesc_t desc_;
};

} //namespace nvimgcodec