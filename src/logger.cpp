/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "logger.h"
#include <nvimgcodecs.h>
#include <algorithm>
#include "idebug_messenger.h"
#include "default_debug_messenger.h"

namespace nvimgcdcs {

Logger::Logger(IDebugMessenger* messenger)
{
    messengers_.push_back(messenger);
}

ILogger* Logger::get()
{
    static DefaultDebugMessenger default_debug_messanger;
    static Logger instance(&default_debug_messanger);
    
    return &instance;
}

void Logger::log(
    const nvimgcdcsDebugMessageSeverity_t message_severity, const nvimgcdcsDebugMessageCategory_t message_category, const std::string& message)
{
    nvimgcdcsDebugMessageData_t data{NVIMGCDCS_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, nullptr, message.c_str(), 0, nullptr, "nvimgcodecs", 0};

    log(message_severity, message_category, &data);
}

void Logger::log(const nvimgcdcsDebugMessageSeverity_t message_severity, const nvimgcdcsDebugMessageCategory_t message_category,
    const nvimgcdcsDebugMessageData_t* data)
{
    for (auto dbgmsg : messengers_) {
        if ((dbgmsg->getDesc()->message_severity & message_severity) && (dbgmsg->getDesc()->message_category & message_category)) {
            dbgmsg->getDesc()->user_callback(message_severity, message_category, data, dbgmsg->getDesc()->user_data);
        }
    }
}

void Logger::registerDebugMessenger(IDebugMessenger* messenger)
{
    messengers_.push_back(messenger);
}

void Logger::unregisterDebugMessenger(IDebugMessenger* messenger)
{
    auto it = std::find(messengers_.begin(), messengers_.end(), messenger);
    if (it != messengers_.end()) {
        messengers_.erase(it);
    }
}

} //namespace nvimgcdcs