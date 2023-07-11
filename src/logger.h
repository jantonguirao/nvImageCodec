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
#include <vector>
#include <string>
#include "ilogger.h"

namespace nvimgcdcs {

class IDebugMessenger;
class Logger : public ILogger
{
  public:
    Logger() = default;
    Logger(IDebugMessenger* messenger);
    static ILogger* get();
    void log(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type, const std::string& message) override ;
    void log(const nvimgcdcsDebugMessageSeverity_t message_severity,
        const nvimgcdcsDebugMessageType_t message_type, const nvimgcdcsDebugMessageData_t* data) override;
    void registerDebugMessenger(IDebugMessenger* messenger) override;
    void unregisterDebugMessenger(IDebugMessenger* messenger) override;

  private:
    std::vector<IDebugMessenger*> messengers_;
};

} //namespace nvimgcdcs