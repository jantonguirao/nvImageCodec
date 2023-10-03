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

#include <string>

namespace nvimgcodec {

class IEnvironment
{
  public:
    virtual ~IEnvironment() = default;

    virtual std::string getVariable(const std::string& env_var) = 0;
};

} // namespace nvimgcodec
