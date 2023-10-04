/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvimgcodec.h>

namespace nvimgcodec {

class IDebugMessenger;

class ILogger
{
  public:
    virtual ~ILogger() = default;
    virtual void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const std::string& message) = 0;
    virtual void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data) = 0;
    virtual void registerDebugMessenger(IDebugMessenger* messenger) = 0;
    virtual void unregisterDebugMessenger(IDebugMessenger* messenger) = 0;
};

} //namespace nvimgcodec