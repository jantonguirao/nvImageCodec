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

#include <gmock/gmock.h>
#include "../src/ilogger.h"

namespace nvimgcodec {

class MockLogger : public ILogger
{
  public:
    MOCK_METHOD(void, log,
        (const nvimgcodecDebugMessageSeverity_t message_severity, const nvimgcodecDebugMessageCategory_t message_type,
            const std::string& message),
        (override));
    MOCK_METHOD(void, log,
        (const nvimgcodecDebugMessageSeverity_t message_severity, const nvimgcodecDebugMessageCategory_t message_type,
            const nvimgcodecDebugMessageData_t* data),
        (override));
    MOCK_METHOD(void, registerDebugMessenger, (IDebugMessenger * messenger), (override));
    MOCK_METHOD(void, unregisterDebugMessenger, (IDebugMessenger * messenger), (override));
};

} // namespace nvimgcodec