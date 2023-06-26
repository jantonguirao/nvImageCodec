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

#include "idebug_messenger.h"

namespace nvimgcdcs {

class DebugMessenger : public IDebugMessenger
{
  public:
    DebugMessenger(const nvimgcdcsDebugMessengerDesc_t* desc): desc_(*desc){}
    const nvimgcdcsDebugMessengerDesc_t* getDesc() override { return &desc_; }
  private:
    const nvimgcdcsDebugMessengerDesc_t desc_;
};

} //namespace nvimgcdcs