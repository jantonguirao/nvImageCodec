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

namespace nvimgcdcs {

Logger& Logger::get()
{
    static Logger instance;
    return instance;
}

void Logger::log(
    const nvimgcdcsDebugMessageSeverity_t message_severity, const nvimgcdcsDebugMessageType_t message_type, const std::string& message)
{
    nvimgcdcsDebugMessageData_t data{NVIMGCDCS_STRUCTURE_TYPE_DEBUG_MESSAGE_DATA, nullptr, message.c_str(), 0, nullptr, "nvimgcodecs", 0};

    log(message_severity, message_type, &data);
}

void Logger::log(const nvimgcdcsDebugMessageSeverity_t message_severity, const nvimgcdcsDebugMessageType_t message_type,
    const nvimgcdcsDebugMessageData_t* data)
{
    for (auto dbgmsg : messengers_) {
        if ((dbgmsg->getDesc()->message_severity & message_severity) && (dbgmsg->getDesc()->message_type & message_type)) {
            dbgmsg->getDesc()->user_callback(message_severity, message_type, data, dbgmsg->getDesc()->userData);
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