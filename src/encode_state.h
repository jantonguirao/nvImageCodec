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
#include <memory>
#include "iencode_state.h"

namespace nvimgcdcs {

class EncodeState : public IEncodeState
{
  public:
    EncodeState(const nvimgcdcsEncoderDesc_t encoder_desc, nvimgcdcsEncoder_t encoder);
    ~EncodeState() override;
    void setPromise(const ProcessingResultsPromise& promise) override;
    const ProcessingResultsPromise& getPromise() override;
    nvimgcdcsEncodeState_t getInternalEncodeState() override;

  private:
    const nvimgcdcsEncoderDesc_t encoder_desc_;
    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsEncodeState_t encode_state_;
    std::unique_ptr<ProcessingResultsPromise> promise_;
};

} // namespace nvimgcdcs
