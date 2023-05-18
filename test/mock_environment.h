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
#include "../src/ienvironment.h"

namespace nvimgcdcs {

class MockEnvironment : public IEnvironment
{
  public:
    MOCK_METHOD(std::string, getVariable, (const std::string&), (override));
};

} // namespace nvimgcdcs
