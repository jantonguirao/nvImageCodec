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

#include <memory>
#include "processing_results.h"
#include "work.h"

namespace nvimgcodec {

template<typename T>
class IWorkManager
{
  public:
    virtual ~IWorkManager() = default;
    virtual std::unique_ptr<Work<T>> createNewWork(
        const ProcessingResultsPromise& results, const void* params) = 0;
    virtual void recycleWork(std::unique_ptr<Work<T>> work) = 0;
    virtual void combineWork(
        Work<T>* target, std::unique_ptr<Work<T>> source) = 0;
};

} // namespace nvimgcodec
