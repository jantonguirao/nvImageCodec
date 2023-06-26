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


namespace nvimgcdcs {

class IDebugMessenger
{
  public:
    virtual ~IDebugMessenger() = default;
    virtual const nvimgcdcsDebugMessengerDesc_t* getDesc() = 0;
};
} // namespace nvimgcdcs
