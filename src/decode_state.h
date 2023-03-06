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
#include "idecode_state.h"

namespace nvimgcdcs {
class DecodeState : public IDecodeState
{
  public:
    DecodeState(const nvimgcdcsDecoderDesc_t decoder_desc, nvimgcdcsDecoder_t decoder);
    ~DecodeState() override;
    void setPromise(const ProcessingResultsPromise& promise) override;
    const ProcessingResultsPromise& getPromise() override;
    nvimgcdcsDecodeState_t getInternalDecodeState() override;

  private:
    const nvimgcdcsDecoderDesc_t decoder_desc_;
    nvimgcdcsDecoder_t decoder_;
    nvimgcdcsDecodeState_t decode_state_;
    std::unique_ptr<ProcessingResultsPromise> promise_;
};
} // namespace nvimgcdcs
