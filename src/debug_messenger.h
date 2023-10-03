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

#include "idebug_messenger.h"

namespace nvimgcodec {

class DebugMessenger : public IDebugMessenger
{
  public:
    DebugMessenger(const nvimgcodecDebugMessengerDesc_t* desc): desc_(*desc){}
    const nvimgcodecDebugMessengerDesc_t* getDesc() override { return &desc_; }
  private:
    const nvimgcodecDebugMessengerDesc_t desc_;
};

} //namespace nvimgcodec