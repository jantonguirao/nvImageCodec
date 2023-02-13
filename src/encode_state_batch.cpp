/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "encode_state_batch.h"

namespace nvimgcdcs {

EncodeStateBatch::EncodeStateBatch(const struct nvimgcdcsEncoderDesc* encoder_desc,
    nvimgcdcsEncoder_t encoder, cudaStream_t cuda_stream)
    : encoder_desc_(encoder_desc)
    , encoder_(encoder)
    , encode_state_(nullptr)
    , cuda_stream_(cuda_stream)
{
    if (encoder_desc_->createEncodeStateBatch) {
        encoder_desc_->createEncodeStateBatch(encoder_, &encode_state_, cuda_stream);
    }
}

EncodeStateBatch::~EncodeStateBatch()
{
    if (encode_state_) {
        encoder_desc_->destroyEncodeState(encode_state_);
    }
}

void EncodeStateBatch::setPromise(std::unique_ptr<ProcessingResultsPromise> promise)
{
    promise_ = std::move(promise);
}

ProcessingResultsPromise* EncodeStateBatch::getPromise()
{
    return promise_.get();
}

nvimgcdcsEncodeState_t EncodeStateBatch::getInternalEncodeState()
{
    return encode_state_;
}
} // namespace nvimgcdcs