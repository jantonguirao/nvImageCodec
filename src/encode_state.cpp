/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "encode_state.h"

namespace nvimgcdcs {

EncodeState::EncodeState(const struct nvimgcdcsEncoderDesc* encoder_desc,
    nvimgcdcsEncoder_t encoder, cudaStream_t cuda_stream)
    : encoder_desc_(encoder_desc)
    , encoder_(encoder)
{
    encoder_desc_->createEncodeState(encoder_, &encode_state_, cuda_stream);
}

EncodeState::~EncodeState()
{
    encoder_desc_->destroyEncodeState(encode_state_);
}

nvimgcdcsEncodeState_t EncodeState::getInternalEncodeState()
{
    return encode_state_;
}
} // namespace nvimgcdcs