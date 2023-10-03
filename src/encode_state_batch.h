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
#include "iencode_state.h"

namespace nvimgcodec {

class EncodeStateBatch : public IEncodeState
{
  public:
    EncodeStateBatch() = default;
    ~EncodeStateBatch() override = default;
    void setPromise(const ProcessingResultsPromise& promise) override;
    const ProcessingResultsPromise& getPromise() override;
  private:
    std::unique_ptr<ProcessingResultsPromise> promise_;
};

} // namespace nvimgcodec
