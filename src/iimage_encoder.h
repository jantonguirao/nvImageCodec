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
#include <string>
#include <vector>
#include "processing_results.h"

namespace nvimgcodec {

class IEncodeState;
class IImage;
class ICodeStream;

class IImageEncoder
{
  public:
    virtual ~IImageEncoder() = default;
    virtual nvimgcodecBackendKind_t getBackendKind() const = 0;
    virtual std::unique_ptr<IEncodeState> createEncodeStateBatch() const = 0;
    virtual void canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
        const nvimgcodecEncodeParams_t* params, std::vector<bool>* result, std::vector<nvimgcodecProcessingStatus_t>* status) const = 0;
    virtual std::unique_ptr<ProcessingResultsFuture> encode(IEncodeState* encode_state_batch,
        const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
        const nvimgcodecEncodeParams_t* params) = 0;
};

} // namespace nvimgcodec