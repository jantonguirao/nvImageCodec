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
#include <vector>
#include <string>
#include "ilogger.h"

namespace nvimgcodec {

class IDebugMessenger;
class Logger : public ILogger
{
  public:
    Logger() = default;
    Logger(IDebugMessenger* messenger);
    static ILogger* get();
    void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const std::string& message) override ;
    void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data) override;
    void registerDebugMessenger(IDebugMessenger* messenger) override;
    void unregisterDebugMessenger(IDebugMessenger* messenger) override;

  private:
    std::vector<IDebugMessenger*> messengers_;
};

} //namespace nvimgcodec