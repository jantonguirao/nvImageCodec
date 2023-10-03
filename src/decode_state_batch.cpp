/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "decode_state_batch.h"

namespace nvimgcodec {

void DecodeStateBatch::setPromise(const ProcessingResultsPromise& promise)
{
    promise_ = std::make_unique<ProcessingResultsPromise>(promise);
}

const ProcessingResultsPromise& DecodeStateBatch::getPromise()
{
    return *promise_;
}

} // namespace nvimgcodec
