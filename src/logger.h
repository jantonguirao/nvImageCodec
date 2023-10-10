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
#include <string>
#include <vector>

#include "default_debug_messenger.h"
#include "idebug_messenger.h"
#include "ilogger.h"

namespace nvimgcodec {

class Logger : public ILogger
{
  public:
    Logger(const std::string& name, IDebugMessenger* messenger = nullptr)
        : name_(name)
    {
        if (messenger != nullptr)
            messengers_.push_back(messenger);
    }

    static ILogger* get()
    {
        static DefaultDebugMessenger default_debug_messanger;
        static Logger instance("nvimgcodec", &default_debug_messanger);

        return &instance;
    }

    void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const std::string& message) override 
    {
        nvimgcodecDebugMessageData_t data{
            NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, nullptr, message.c_str(), 0, nullptr, name_.c_str(), 0};

        log(message_severity, message_category, &data);
    }

    void log(const nvimgcodecDebugMessageSeverity_t message_severity,
        const nvimgcodecDebugMessageCategory_t message_category, const nvimgcodecDebugMessageData_t* data) override
    {
        for (auto dbgmsg : messengers_) {
            if ((dbgmsg->getDesc()->message_severity & message_severity) && (dbgmsg->getDesc()->message_category & message_category)) {
                dbgmsg->getDesc()->user_callback(message_severity, message_category, data, dbgmsg->getDesc()->user_data);
            }
        }
    }

    void registerDebugMessenger(IDebugMessenger* messenger) override
    { 
      messengers_.push_back(messenger); 
    }

    void unregisterDebugMessenger(IDebugMessenger* messenger) override
    {
        auto it = std::find(messengers_.begin(), messengers_.end(), messenger);
        if (it != messengers_.end()) {
            messengers_.erase(it);
        }
    }

  private:
    std::vector<IDebugMessenger*> messengers_;
    std::string name_;
};

} //namespace nvimgcodec