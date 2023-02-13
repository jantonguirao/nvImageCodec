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
#include "iencode_state.h"

namespace nvimgcdcs {

class EncodeStateBatch : public IEncodeState
{
  public:
    EncodeStateBatch(const struct nvimgcdcsEncoderDesc* encoder_desc, nvimgcdcsEncoder_t encoder,
        cudaStream_t cuda_stream);
    ~EncodeStateBatch() override;
    void setPromise(std::unique_ptr<ProcessingResultsPromise> promise) override;
    ProcessingResultsPromise* getPromise() override;
    nvimgcdcsEncodeState_t getInternalEncodeState() override;

  private:
    const struct nvimgcdcsEncoderDesc* encoder_desc_;
    nvimgcdcsEncoder_t encoder_;
    nvimgcdcsEncodeState_t encode_state_;
    cudaStream_t cuda_stream_;
    std::unique_ptr<ProcessingResultsPromise> promise_;
};

} // namespace nvimgcdcs