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
#include <memory>
#include "processing_results.h"

namespace nvimgcodec {

class ProcessingResultsPromise;

class IDecodeState
{
  public:
    virtual ~IDecodeState() = default;
    virtual void setPromise(const ProcessingResultsPromise& promise) = 0;
    virtual const ProcessingResultsPromise& getPromise() = 0;
};
} // namespace nvimgcodec
