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

namespace nvimgcdcs {

DecodeStateBatch::DecodeStateBatch(const struct nvimgcdcsDecoderDesc* decoder_desc,
    nvimgcdcsDecoder_t decoder, cudaStream_t cuda_stream)
    : decoder_desc_(decoder_desc)
    , decoder_(decoder)
    , decode_state_(nullptr)
    , cuda_stream_(cuda_stream)
{
    if (decoder_desc_)
        if (decoder_desc_->createDecodeStateBatch) {
            decoder_desc_->createDecodeStateBatch(decoder_, &decode_state_, cuda_stream);
        }
}

DecodeStateBatch::~DecodeStateBatch()
{
    if (decode_state_) {
        decoder_desc_->destroyDecodeState(decode_state_);
    }
}

void DecodeStateBatch::setPromise(std::unique_ptr<ProcessingResultsPromise> promise)
{
    promise_ = std::move(promise);
}

ProcessingResultsPromise* DecodeStateBatch::getPromise()
{
    return promise_.get();
}

nvimgcdcsDecodeState_t DecodeStateBatch::getInternalDecodeState()
{
    return decode_state_;
}
} // namespace nvimgcdcs