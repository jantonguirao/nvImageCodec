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
#include <nvimgcdcs_module.h>

namespace nvimgcdcs {
class EncodeState
{
  public:
    EncodeState(
        const struct nvimgcdcsEncoderDesc* encoder_desc, nvimgcdcsEncoder_t encoder);
    ~EncodeState();
    nvimgcdcsEncodeState_t getInternalEncodeState();

  private:
    const struct nvimgcdcsEncoderDesc* encoder_desc_;
    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsEncodeState_t encode_state_;
};
} // namespace nvimgcdcs