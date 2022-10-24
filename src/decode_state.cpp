/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "decode_state.h"

namespace nvimgcdcs {

DecodeState::DecodeState(
    const struct nvimgcdcsDecoderDesc* decoder_desc, nvimgcdcsDecoder_t decoder)
    : decoder_desc_(decoder_desc)
    , decoder_(decoder)
{
    decoder_desc_->createDecodeState(decoder_, &decode_state_);
}

DecodeState::~DecodeState()
{
    decoder_desc_->destroyDecodeState(decode_state_);
}

nvimgcdcsDecodeState_t DecodeState::getInternalDecodeState()
{
    return decode_state_;
}
} // namespace nvimgcdcs