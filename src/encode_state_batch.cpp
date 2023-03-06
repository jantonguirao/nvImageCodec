/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <memory>
#include "encode_state_batch.h"

namespace nvimgcdcs {

EncodeStateBatch::EncodeStateBatch(const nvimgcdcsEncoderDesc_t encoder_desc,
    nvimgcdcsEncoder_t encoder)
    : encoder_desc_(encoder_desc)
    , encoder_(encoder)
    , encode_state_(nullptr)
{
    if (encoder_desc_)
        if (encoder_desc_->createEncodeStateBatch) {
            encoder_desc_->createEncodeStateBatch(encoder_, &encode_state_);
        }
}

EncodeStateBatch::~EncodeStateBatch()
{
    if (encode_state_) {
        encoder_desc_->destroyEncodeState(encode_state_);
    }
}

void EncodeStateBatch::setPromise(const ProcessingResultsPromise& promise)
{
    promise_ = std::make_unique<ProcessingResultsPromise>(promise);
}

const ProcessingResultsPromise& EncodeStateBatch::getPromise()
{
    return *promise_;
}

nvimgcdcsEncodeState_t EncodeStateBatch::getInternalEncodeState()
{
    return encode_state_;
}
} // namespace nvimgcdcs
