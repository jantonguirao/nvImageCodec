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
#include "processing_results.h"

namespace nvimgcdcs {

class IEncodeState
{
  public:
    virtual ~IEncodeState() = default;
    virtual void setPromise(std::unique_ptr<ProcessingResultsPromise> promise) = 0;
    virtual ProcessingResultsPromise* getPromise() = 0;
    virtual nvimgcdcsEncodeState_t getInternalEncodeState() = 0;
};

} // namespace nvimgcdcs